import argparse
from functools import partial

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .diff_head import DiffHead
from .layers import TransformerBlock, get_2d_pos, precompute_freqs_cis_2d
from .vae import VAE


def get_model_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, choices=list(SphereAR_models.keys()), default="SphereAR-L"
    )
    parser.add_argument("--vae-only", action="store_true", help="only train vae")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--patch-size", type=int, default=16, choices=[16])
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cls-token-num", type=int, default=16)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--diff-batch-mul", type=int, default=4)
    parser.add_argument("--grad-checkpointing", action="store_true")
    return parser


def create_model(args, device):
    model = SphereAR_models[args.model](
        resolution=args.image_size,
        patch_size=args.patch_size,
        latent_dim=args.latent_dim,
        vae_only=args.vae_only,
        diff_batch_mul=args.diff_batch_mul,
        cls_token_num=args.cls_token_num,
        num_classes=args.num_classes,
        grad_checkpointing=args.grad_checkpointing,
    ).to(device, memory_format=torch.channels_last)
    return model


class SphereAR(nn.Module):

    def __init__(
        self,
        dim,
        n_layer,
        n_head,
        diff_layers,
        diff_dim,
        diff_adanln_layers,
        latent_dim,
        patch_size,
        resolution,
        diff_batch_mul,
        vae_only=False,
        grad_checkpointing=False,
        cls_token_num=16,
        num_classes: int = 1000,
        class_dropout_prob: float = 0.1,
    ):
        super().__init__()

        self.n_layer = n_layer
        self.resolution = resolution
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.cls_token_num = cls_token_num
        self.class_dropout_prob = class_dropout_prob
        self.latent_dim = latent_dim

        self.vae = VAE(
            latent_dim=latent_dim, image_size=resolution, patch_size=patch_size
        )
        self.vae_only = vae_only
        self.grad_checkpointing = grad_checkpointing

        if not vae_only:
            self.cls_embedding = nn.Embedding(num_classes + 1, dim * self.cls_token_num)
            self.proj_in = nn.Linear(latent_dim, dim, bias=True)
            self.emb_norm = nn.RMSNorm(dim, eps=1e-6, elementwise_affine=True)
            self.h, self.w = resolution // patch_size, resolution // patch_size
            self.total_tokens = self.h * self.w + self.cls_token_num

            self.layers = torch.nn.ModuleList()
            for layer_id in range(n_layer):
                self.layers.append(
                    TransformerBlock(
                        dim,
                        n_head,
                        causal=True,
                    )
                )

            self.norm = nn.RMSNorm(dim, eps=1e-6, elementwise_affine=True)
            self.pos_for_diff = nn.Embedding(self.h * self.w, dim)
            self.head = DiffHead(
                ch_target=latent_dim,
                ch_cond=dim,
                ch_latent=diff_dim,
                depth_latent=diff_layers,
                depth_adanln=diff_adanln_layers,
                grad_checkpointing=grad_checkpointing,
            )
            self.diff_batch_mul = diff_batch_mul

            patch_2d_pos = get_2d_pos(resolution, patch_size)

            self.register_buffer(
                "freqs_cis",
                precompute_freqs_cis_2d(
                    patch_2d_pos,
                    dim // n_head,
                    10000,
                    cls_token_num=self.cls_token_num,
                )[:-1],
                persistent=False,
            )
            self.freeze_vae()

        self.initialize_weights()

    def non_decay_keys(self):
        return ["proj_in", "cls_embedding"]

    def freeze_module(self, module: nn.Module):
        for param in module.parameters():
            param.requires_grad = False

    def freeze_vae(self):
        self.freeze_module(self.vae)
        self.vae.eval()

    def initialize_weights(self):
        # Initialize nn.Linear and nn.Embedding
        self.apply(self.__init_weights)
        if not self.vae_only:
            self.head.initialize_weights()
        self.vae.initialize_weights()

    def __init_weights(self, module):
        std = 0.02
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)

    def drop_label(self, class_id):
        if self.class_dropout_prob > 0.0 and self.training:
            is_drop = (
                torch.rand(class_id.shape, device=class_id.device)
                < self.class_dropout_prob
            )
            class_id = torch.where(is_drop, self.num_classes, class_id)
        return class_id

    def forward(
        self,
        images,
        class_id,
    ):

        vae_latent, kl_loss = self.vae.encode(images)

        if not self.vae_only:
            x = vae_latent.detach()
            x = self.proj_in(x[:, :-1, :])
            class_id = self.drop_label(class_id)
            bsz = x.shape[0]
            c = self.cls_embedding(class_id).view(bsz, self.cls_token_num, -1)
            x = torch.cat([c, x], dim=1)
            x = self.emb_norm(x)

            if self.grad_checkpointing and self.training:
                for layer in self.layers:
                    block = partial(layer.forward, freqs_cis=self.freqs_cis)
                    x = checkpoint(block, x, use_reentrant=False)
            else:
                for layer in self.layers:
                    x = layer(x, self.freqs_cis)

            x = x[:, -self.h * self.w :, :]
            x = self.norm(x)
            x = x + self.pos_for_diff.weight

            target = vae_latent.detach()
            x = x.view(-1, x.shape[-1])
            target = target.view(-1, target.shape[-1])

            x = x.repeat(self.diff_batch_mul, 1)
            target = target.repeat(self.diff_batch_mul, 1)
            loss = self.head(target, x)
            recon = None
        else:
            loss = torch.tensor(0.0, device=images.device, dtype=images.dtype)
            recon = self.vae.decode(vae_latent)

        return loss, kl_loss, recon

    def enable_kv_cache(self, bsz):
        for layer in self.layers:
            layer.attention.enable_kv_cache(bsz, self.total_tokens)

    @torch.compile()
    def forward_model(self, x, start_pos, end_pos):
        x = self.emb_norm(x)
        for layer in self.layers:
            x = layer.forward_onestep(
                x, self.freqs_cis[start_pos:end_pos,], start_pos, end_pos
            )
        x = self.norm(x)
        return x

    def head_sample(self, x, diff_pos, sample_steps, cfg_scale, cfg_schedule="linear"):
        x = x + self.pos_for_diff.weight[diff_pos : diff_pos + 1, :]
        x = x.view(-1, x.shape[-1])
        seq_len = self.h * self.w
        if cfg_scale > 1.0:
            if cfg_schedule == "constant":
                cfg_iter = cfg_scale
            elif cfg_schedule == "linear":
                start = 1.0
                cfg_iter = start + (cfg_scale - start) * diff_pos / seq_len
            else:
                raise NotImplementedError(f"unknown cfg_schedule {cfg_schedule}")
        else:
            cfg_iter = 1.0
        pred = self.head.sample(x, num_sampling_steps=sample_steps, cfg=cfg_iter)
        pred = pred.view(-1, 1, pred.shape[-1])
        # Important: normalize here, for both next-token prediction and vae decoding
        pred = self.vae.normalize(pred)
        return pred

    @torch.no_grad()
    def sample(self, cond, sample_steps, cfg_scale=1.0, cfg_schedule="linear"):
        self.eval()
        if cfg_scale > 1.0:
            cond_null = torch.ones_like(cond) * self.num_classes
            cond_combined = torch.cat([cond, cond_null])
        else:
            cond_combined = cond
        bsz = cond_combined.shape[0]
        act_bsz = bsz // 2 if cfg_scale > 1.0 else bsz
        self.enable_kv_cache(bsz)

        c = self.cls_embedding(cond_combined).view(bsz, self.cls_token_num, -1)
        last_pred = None
        all_preds = []
        for i in range(self.h * self.w):
            if i == 0:
                x = self.forward_model(c, 0, self.cls_token_num)
            else:
                x = self.proj_in(last_pred)
                x = self.forward_model(
                    x, i + self.cls_token_num - 1, i + self.cls_token_num
                )
            last_pred = self.head_sample(
                x[:, -1:, :],
                i,
                sample_steps,
                cfg_scale,
                cfg_schedule,
            )
            all_preds.append(last_pred)

        x = torch.cat(all_preds, dim=-2)[:act_bsz]
        recon = self.vae.decode(x)
        return recon

    def get_fsdp_wrap_module_list(self):
        return list(self.layers)


def SphereAR_H(**kwargs):
    return SphereAR(
        n_layer=40,
        n_head=20,
        dim=1280,
        diff_layers=12,
        diff_dim=1280,
        diff_adanln_layers=3,
        **kwargs,
    )


def SphereAR_L(**kwargs):
    return SphereAR(
        n_layer=32,
        n_head=16,
        dim=1024,
        diff_layers=8,
        diff_dim=1024,
        diff_adanln_layers=2,
        **kwargs,
    )


def SphereAR_B(**kwargs):
    return SphereAR(
        n_layer=24,
        n_head=12,
        dim=768,
        diff_layers=6,
        diff_dim=768,
        diff_adanln_layers=2,
        **kwargs,
    )


SphereAR_models = {
    "SphereAR-B": SphereAR_B,
    "SphereAR-L": SphereAR_L,
    "SphereAR-H": SphereAR_H,
}

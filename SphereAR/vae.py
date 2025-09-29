# Modified from:
#   taming-transformers: https://github.com/CompVis/taming-transformers
#   maskgit: https://github.com/google-research/maskgit
#   llamagen: https://github.com/FoundationVision/LlamaGen

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import TransformerBlock, get_2d_pos, precompute_freqs_cis_2d
from .psd import PowerSphericalDistribution, l2_norm


class VAE(nn.Module):

    def __init__(
        self,
        latent_dim=16,
        image_size=256,
        patch_size=16,
        z_channels=512,
        cnn_chs=[64, 64, 128, 256, 512],
        encoder_vit_layers=6,
        decoder_vit_layers=12,
    ):
        super().__init__()
        self.z_channels = z_channels
        n_head = z_channels // 64
        self.encoder = ViTEncoder(
            n_layers=encoder_vit_layers,
            d_model=z_channels,
            n_heads=n_head,
            cnn_chs=cnn_chs,
            image_size=image_size,
            patch_size=patch_size,
        )
        self.decoder = ViTDecoder(
            n_layers=decoder_vit_layers,
            d_model=z_channels,
            n_heads=n_head,
            cnn_chs=cnn_chs[::-1],
            image_size=image_size,
            patch_size=patch_size,
        )
        self.latent_dim = latent_dim
        self.quant_proj = nn.Linear(z_channels, latent_dim + 1, bias=True)
        self.post_quant_proj = nn.Linear(latent_dim, z_channels, bias=False)

    def initialize_weights(self):

        self.quant_proj.reset_parameters()
        self.post_quant_proj.reset_parameters()
        self.encoder.output.reset_parameters()

    def normalize(self, x):
        x = l2_norm(x)
        x = x * (self.latent_dim**0.5)
        return x

    def encode(self, x):
        x = self.encoder(x)
        x = self.quant_proj(x)
        mu = x[..., :-1]
        kappa = x[..., -1]
        mu = l2_norm(mu)
        kappa = F.softplus(kappa) + 1.0
        qz = PowerSphericalDistribution(mu, kappa)
        loss = qz.kl_to_uniform()
        x = qz.rsample()
        x = x * (self.latent_dim**0.5)
        return x, loss.mean()

    def decode(self, x):
        x = self.post_quant_proj(x)
        dec = self.decoder(x)
        return dec


class ResDownBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(min(32, in_ch // 4), in_ch, eps=1e-6, affine=True),
            nn.SiLU(),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(min(32, out_ch // 4), out_ch, eps=1e-6, affine=True),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
        )
        self.shortcut = nn.Conv2d(
            in_ch, out_ch, kernel_size=2, stride=2, padding=0, bias=False
        )

    def forward(self, x):
        return self.shortcut(x) + self.block(x)


class PatchifyNet(nn.Module):

    def __init__(self, chs):
        super().__init__()
        layers = []
        for i in range(len(chs) - 1):
            layers.append(ResDownBlock(chs[i], chs[i + 1]))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = self.net(x)
        return x


class NCHW_to_NLC(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        n, c, h, w = x.shape
        x = x.permute(0, 2, 3, 1).view(n, h * w, c)
        return x


class NLC_to_NCHW(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        n, l, c = x.shape
        h = w = int(l**0.5)
        x = x.view(n, h, w, c).permute(0, 3, 1, 2)
        return x


class ResUpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        num_groups = min(32, out_ch // 4)
        self.block = nn.Sequential(
            nn.GroupNorm(min(32, in_ch // 4), in_ch, eps=1e-6, affine=True),
            nn.SiLU(),
            nn.ConvTranspose2d(
                in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False
            ),
            nn.GroupNorm(num_groups, out_ch, eps=1e-6, affine=True),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
        )
        self.shortcut = nn.ConvTranspose2d(
            in_ch, out_ch, kernel_size=2, stride=2, bias=False
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(num_groups, out_ch, eps=1e-6, affine=True),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(num_groups, out_ch, eps=1e-6, affine=True),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
        )

    def forward(self, x):
        h = self.block(x)
        x = self.shortcut(x)
        x = x + h
        x = x + self.block2(x)
        return x


class UnpatchifyNet(nn.Module):

    def __init__(self, chs):
        super().__init__()
        layers = []
        for i in range(len(chs) - 1):
            layers.append(ResUpBlock(chs[i], chs[i + 1]))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = self.net(x)
        return x


class ViTEncoder(nn.Module):

    def __init__(
        self,
        n_layers=6,
        n_heads=8,
        d_model=512,
        cnn_chs=[64, 64, 128, 256, 512],
        image_size=256,
        patch_size=16,
    ):
        super().__init__()
        self.conv_in = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.patchify = nn.Sequential(
            PatchifyNet(cnn_chs),
            NCHW_to_NLC(),
        )
        assert cnn_chs[-1] == d_model
        self.register_num_tokens = 4
        self.register_token = nn.Embedding(self.register_num_tokens, d_model)
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(
                TransformerBlock(
                    d_model,
                    n_heads,
                    0.0,
                    0.0,
                    drop_path=0.0,
                    causal=False,
                )
            )
        self.norm = nn.RMSNorm(d_model, eps=1e-6)
        self.output = nn.Linear(d_model, d_model, bias=False)
        raw_2d_pos = get_2d_pos(image_size, patch_size)
        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis_2d(
                raw_2d_pos, d_model // n_heads, cls_token_num=self.register_num_tokens
            ).clone(),
        )

    def forward(self, image):
        x = self.conv_in(image)
        x = self.patchify(x)
        x_null = self.register_token.weight.view(1, -1, x.shape[-1]).expand(
            x.shape[0], -1, -1
        )
        x = torch.cat([x_null, x], dim=1)
        for layer in self.layers:
            x = layer(x, self.freqs_cis)
        x = x[:, self.register_num_tokens :, :]
        x = self.output(self.norm(x))
        return x


class ViTDecoder(nn.Module):
    def __init__(
        self,
        n_layers=12,
        n_heads=8,
        d_model=512,
        cnn_chs=[512, 256, 128, 64, 64],
        image_size=256,
        patch_size=16,
    ):
        super().__init__()
        self.d_model = d_model
        assert d_model == cnn_chs[0]
        self.conv_in = nn.Sequential(
            NLC_to_NCHW(),
            nn.Conv2d(d_model, d_model, kernel_size=3, stride=1, padding=1, bias=False),
            NCHW_to_NLC(),
        )
        self.register_num_tokens = 4
        self.register_token = nn.Embedding(self.register_num_tokens, d_model)
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(
                TransformerBlock(
                    d_model,
                    n_heads,
                    0.0,
                    0.0,
                    drop_path=0.0,
                    causal=False,
                )
            )
        self.unpatchify = nn.Sequential(
            NLC_to_NCHW(),
            UnpatchifyNet(chs=cnn_chs),
        )
        self.conv_out = nn.Sequential(
            nn.GroupNorm(16, cnn_chs[-1], eps=1e-6, affine=True),
            nn.SiLU(),
            nn.Conv2d(cnn_chs[-1], 3, kernel_size=3, stride=1, padding=1, bias=False),
        )

        raw_2d_pos = get_2d_pos(image_size, patch_size)
        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis_2d(
                raw_2d_pos, d_model // n_heads, cls_token_num=self.register_num_tokens
            ).clone(),
        )

    @torch.compile()
    def forward(self, x):
        x = self.conv_in(x)
        x_null = self.register_token.weight.view(1, -1, x.shape[-1]).expand(
            x.shape[0], -1, -1
        )
        x = torch.cat([x_null, x], dim=1)
        for layer in self.layers:
            x = layer(x, self.freqs_cis)
        x = x[:, self.register_num_tokens :, :]
        x = self.unpatchify(x)
        x = self.conv_out(x)
        return x

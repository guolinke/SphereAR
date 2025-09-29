# Modified from:
#   taming-transformers:  https://github.com/CompVis/taming-transformers
#   muse-maskgit-pytorch: https://github.com/lucidrains/muse-maskgit-pytorch/blob/main/muse_maskgit_pytorch/vqgan_vae.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch._dynamo as dynamo

from .lpips import LPIPS
from .discriminator_patchgan import (
    NLayerDiscriminator as PatchGANDiscriminator,
)
from .discriminator_stylegan import (
    Discriminator as StyleGANDiscriminator,
)


def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1.0 - logits_real))
    loss_fake = torch.mean(F.relu(1.0 + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def vanilla_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.softplus(-logits_real))
    loss_fake = torch.mean(F.softplus(logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def non_saturating_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(
        F.binary_cross_entropy_with_logits(torch.ones_like(logits_real), logits_real)
    )
    loss_fake = torch.mean(
        F.binary_cross_entropy_with_logits(torch.zeros_like(logits_fake), logits_fake)
    )
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def hinge_gen_loss(logit_fake):
    return -torch.mean(logit_fake)


def non_saturating_gen_loss(logit_fake):
    return torch.mean(
        F.binary_cross_entropy_with_logits(torch.ones_like(logit_fake), logit_fake)
    )


def adopt_weight(weight, global_step, threshold=0, value=0.0):
    if global_step < threshold:
        weight = value
    return weight


class GANLoss(nn.Module):
    def __init__(
        self,
        disc_start,
        disc_loss="hinge",
        disc_dim=64,
        disc_type="patchgan",
        image_size=256,
        disc_num_layers=3,
        disc_in_channels=3,
        disc_weight=0.5,
        gen_adv_loss="hinge",
        reconstruction_loss="l2",
    ):
        super().__init__()
        # discriminator loss
        assert disc_type in ["patchgan", "stylegan"]
        assert disc_loss in ["hinge", "vanilla", "non-saturating"]
        if disc_type == "patchgan":
            self.discriminator = PatchGANDiscriminator(
                input_nc=disc_in_channels,
                n_layers=disc_num_layers,
                ndf=disc_dim,
            )
        elif disc_type == "stylegan":
            self.discriminator = StyleGANDiscriminator(
                input_nc=disc_in_channels,
                image_size=image_size,
            )
        else:
            raise ValueError(f"Unknown GAN discriminator type '{disc_type}'.")
        if disc_loss == "hinge":
            self.disc_loss = hinge_d_loss
        elif disc_loss == "vanilla":
            self.disc_loss = vanilla_d_loss
        elif disc_loss == "non-saturating":
            self.disc_loss = non_saturating_d_loss
        else:
            raise ValueError(f"Unknown GAN discriminator loss '{disc_loss}'.")
        self.discriminator_iter_start = disc_start
        self.disc_weight = disc_weight

        assert gen_adv_loss in ["hinge", "non-saturating"]
        # gen_adv_loss
        if gen_adv_loss == "hinge":
            self.gen_adv_loss = hinge_gen_loss
        elif gen_adv_loss == "non-saturating":
            self.gen_adv_loss = non_saturating_gen_loss
        else:
            raise ValueError(f"Unknown GAN generator loss '{gen_adv_loss}'.")

        # perceptual loss
        self.perceptual_loss = LPIPS().eval()

        # reconstruction loss
        if reconstruction_loss == "l1":
            self.rec_loss = F.l1_loss
        elif reconstruction_loss == "l2":
            self.rec_loss = F.mse_loss
        else:
            raise ValueError(f"Unknown rec loss '{reconstruction_loss}'.")

    def forward(
        self,
        inputs,
        reconstructions,
        optimizer_idx,
        global_step,
    ):
        # inputs = inputs.contiguous()
        # reconstructions = reconstructions.contiguous()
        # generator update
        if optimizer_idx == 0:
            # reconstruction loss
            rec_loss = self.rec_loss(inputs, reconstructions)

            # perceptual loss
            p_loss = self.perceptual_loss(inputs, reconstructions)
            p_loss = torch.mean(p_loss)

            # discriminator loss
            logits_fake = self.discriminator(reconstructions)
            generator_adv_loss = self.gen_adv_loss(logits_fake)

            disc_weight = adopt_weight(
                self.disc_weight, global_step, threshold=self.discriminator_iter_start
            )

            return (
                rec_loss,
                p_loss,
                disc_weight * generator_adv_loss,
            )

        # discriminator update
        if optimizer_idx == 1:
            logits_real = self.discriminator(inputs.detach())
            logits_fake = self.discriminator(reconstructions.detach())

            disc_weight = adopt_weight(
                self.disc_weight, global_step, threshold=self.discriminator_iter_start
            )
            d_adversarial_loss = disc_weight * self.disc_loss(logits_real, logits_fake)

            return d_adversarial_loss

import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
from ..lpips import LPIPS, LPIPS_Pseudo3D
from ..med3dnet import LPIPS3D
from .util import *
from torch.amp import autocast
import time
from typing import List
import numpy as np


def random_crop_3d(x: List[torch.Tensor], crop_size=64):
    """
    Randomly crop a 3D patch of size `crop_size` from each tensor in the list x.
    crop_size can be an int or a list/tuple of 1 or 3 ints (D, H, W).
    """
    _, _, D, H, W = x[0].shape
    # Normalize crop_size to list of 3 ints
    if isinstance(crop_size, int):
        crop_size = [crop_size, crop_size, crop_size]
    elif isinstance(crop_size, (list, tuple)):
        if len(crop_size) == 1:
            crop_size = [crop_size[0]] * 3
        elif len(crop_size) == 3:
            crop_size = list(crop_size)
        else:
            raise ValueError("crop_size must be int or a list/tuple of 1 or 3 ints")
    else:
        raise ValueError("crop_size must be int or a list/tuple of 1 or 3 ints")
    crop_d, crop_h, crop_w = crop_size

    d0 = torch.randint(0, D - crop_d + 1, (1,)).item()
    h0 = torch.randint(0, H - crop_h + 1, (1,)).item()
    w0 = torch.randint(0, W - crop_w + 1, (1,)).item()

    return [x[i][:, :, d0:d0+crop_d, h0:h0+crop_h, w0:w0+crop_w] for i in range(len(x))]

class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator that can handle both 2D and 3D inputs"""
    def __init__(self, input_nc=3, ndf=64, n_layers=3, use_actnorm=False, kw=4, padw=1, dims=2):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            use_actnorm    -- whether to use ActNorm instead of BatchNorm
            dims (int)     -- number of dimensions (2 for 2D, 3 for 3D)
        """
        super(NLayerDiscriminator, self).__init__()
        self.dims = dims
        norm_layer = get_norm_layer(dims, use_actnorm)
        conv_layer = get_conv_layer(dims)
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func != nn.BatchNorm2d and norm_layer.func != nn.BatchNorm3d
        else:
            use_bias = norm_layer != nn.BatchNorm2d and norm_layer != nn.BatchNorm3d

        self.kw = kw
        self.padw = padw
        sequence = [conv_layer(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                conv_layer(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            conv_layer(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [
            conv_layer(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.main = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.main(input)

class VQLPIPSWithDiscriminator(nn.Module):
    def __init__(self, disc_start, 
                 disc_loss="hinge",
                 disc_ndf=64,
                 disc_crop_size=None,
                 disc_kw=4,
                 disc_padw=1,
                 disc_num_layers=3, 
                 disc_in_channels=3, 
                 disc_weight=1.0,
                 disc_adaptive_weight=False,
                 disc_warmup=None,
                 use_actnorm=False, 
                 
                 gen_adv_loss="hinge",
                 
                 reconstruction_loss="l1",
                 reconstruction_weight=1.0,
                 codebook_weight=1.0,
                 
                 perceptual_loss="vgg",
                 perceptual_loss_slice_ratio=0.1,
                 perceptual_weight=1.0, 
                 perceptual_intermediate_loss=False,
                 perceptual_resize=False, 
                 perceptual_warmup=None,
                 
                 lecam_loss_weight=None, 
                 disc_cr_loss_weight=0.0,
                 use_diff_aug=False,  
                 kl_weight=1e-6,         
                 dims=2):
        super().__init__()
        assert disc_loss in ["hinge", "vanilla", "non-saturating"]
        assert gen_adv_loss in ["hinge", "non-saturating"]
        
        ## Initialize all the parameters
        self.disc_start = disc_start
        self.disc_loss = disc_loss
        self.gen_adv_loss = gen_adv_loss
        self.reconstruction_loss = reconstruction_loss
        self.reconstruction_weight = reconstruction_weight
        self.codebook_weight = codebook_weight
        self.perceptual_loss = perceptual_loss
        self.perceptual_loss_slice_ratio = perceptual_loss_slice_ratio
        self.perceptual_weight = perceptual_weight
        self.perceptual_intermediate_loss = perceptual_intermediate_loss
        self.perceptual_resize = perceptual_resize
        self.perceptual_warmup = perceptual_warmup
        self.lecam_loss_weight = lecam_loss_weight
        self.disc_cr_loss_weight = disc_cr_loss_weight
        self.use_diff_aug = use_diff_aug
        self.dims = dims
        self.disc_ndf = disc_ndf
        self.disc_kw = disc_kw
        self.disc_padw = disc_padw
        self.disc_num_layers = disc_num_layers
        self.disc_in_channels = disc_in_channels
        self.disc_weight = disc_weight
        self.disc_adaptive_weight = disc_adaptive_weight
        self.disc_warmup = disc_warmup
        self.disc_crop_size = disc_crop_size

        self.use_actnorm = use_actnorm
        self.kl_weight = kl_weight
        self.discriminator = NLayerDiscriminator(input_nc=disc_in_channels,
                                               n_layers=disc_num_layers,
                                               use_actnorm=use_actnorm,
                                               ndf=disc_ndf,
                                               kw=disc_kw,
                                               padw=disc_padw,
                                               dims=dims
                                               ).apply(self.weights_init) if disc_weight is not None else None
        
        ## Discriminator loss
        if disc_loss == "hinge":
            self.disc_loss = hinge_d_loss
        elif disc_loss == "vanilla":
            self.disc_loss = vanilla_d_loss
        elif disc_loss == "non-saturating":
            self.disc_loss = non_saturating_d_loss
        else:
            raise ValueError(f"Unknown GAN loss '{disc_loss}'.")

        ## Generator loss
        if gen_adv_loss == "hinge":
            self.gen_adv_loss = hinge_gen_loss
        elif gen_adv_loss == "non-saturating":
            self.gen_adv_loss = non_saturating_gen_loss
        else:
            raise ValueError(f"Unknown GAN loss '{gen_adv_loss}'.")


        if dims == 3:
            if self.perceptual_loss == "vgg_triplane":
                self.perceptual_loss = LPIPS_Pseudo3D(slice_ratio=self.perceptual_loss_slice_ratio)
            elif self.perceptual_loss == "med3dnet":
                self.perceptual_loss = LPIPS3D(model_name="resnet200", device="cuda")
            elif self.perceptual_loss is not None:
                raise ValueError(f"Unknown perceptual loss '{perceptual_loss}'.")
        elif dims == 2:
            if self.perceptual_loss == "vgg":
                self.perceptual_loss = LPIPS().eval()
            elif self.perceptual_loss is not None:
                raise ValueError(f"Unknown perceptual loss '{perceptual_loss}'.")
        else:
            raise ValueError(f"Unknown dimensions '{dims}'.")

        ## Reconstruction loss
        if reconstruction_loss == "l1":
            self.reconstruction_loss = nn.L1Loss()
        elif reconstruction_loss == "l2":
            self.reconstruction_loss = nn.MSELoss()
        else:
            raise ValueError(f"Unknown reconstruction loss '{reconstruction_loss}'.")

        if self.lecam_loss_weight is not None:
            self.lecam_loss = LeCAM_EMA()

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer):
        if last_layer is None:
            return torch.tensor(1.0, device=nll_loss.device)
        
        nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
        g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()

        return d_weight.detach()
    
    @autocast('cuda', enabled=False) # loss should be in float32
    def forward(self, codebook_loss, inputs, reconstructions, optimizer_idx, global_step, last_layer=None, posteriors=None, split="train"):
        # generator update
        if optimizer_idx == 0:

            # standard reconstruction loss
            rec_loss = self.reconstruction_loss(inputs.contiguous(), reconstructions.contiguous())
            rec_loss = self.reconstruction_weight * rec_loss


            # standard codebook loss from quantizer 
            codebook_loss = self.codebook_weight * codebook_loss.mean()


            # perceptual loss LPIPS or triplane for 3D 
            p_loss = self.perceptual_loss(inputs.contiguous(), reconstructions.contiguous()) if self.perceptual_loss is not None else torch.tensor(0.0)
            p_loss = torch.mean(p_loss)
            if self.perceptual_warmup is None:
                self.perceptual_weight_loss = self.perceptual_weight
            else:
                self.perceptual_weight_loss = min(1.0, global_step / self.perceptual_warmup) * self.perceptual_weight

            p_loss = self.perceptual_weight_loss * p_loss

            if self.discriminator is not None:
                # discriminator loss with DiffAug -> https://arxiv.org/pdf/2006.10738
                if self.use_diff_aug:
                    reconstructions = DiffAugment(reconstructions.contiguous(), policy='color,translation,cutout_0.2', prob=0.5)
                logits_fake = self.discriminator(reconstructions.contiguous()) if self.discriminator is not None else torch.tensor(0.0)
                generator_adv_loss = self.gen_adv_loss(logits_fake)
                # adaptive weighting from -> https://arxiv.org/pdf/2012.09841
                if self.disc_adaptive_weight:
                    null_loss = self.reconstruction_weight * rec_loss + self.perceptual_weight_loss * p_loss
                    disc_adaptive_weight = self.calculate_adaptive_weight(null_loss, generator_adv_loss, last_layer=last_layer)
                else:
                    disc_adaptive_weight = torch.tensor(1.0)
                disc_weight = adopt_weight(self.disc_weight, global_step, threshold=self.disc_start, warmup_steps=self.disc_warmup)
                
                generator_adv_loss = disc_adaptive_weight * disc_weight * generator_adv_loss
            else:
                generator_adv_loss = torch.tensor(0.0)
                disc_adaptive_weight = torch.tensor(1.0)
                disc_weight = torch.tensor(0.0)


            # KL div loss for posteriors VAEKL -> https://arxiv.org/pdf/1312.6114
            kl_loss = torch.zeros((), device=inputs.device)
            if posteriors is not None:
                # assume extra_result_dict contains posteriors with kl method
                if hasattr(posteriors, "kl"):
                    kl_loss = posteriors.kl()
                    kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
            kl_loss = self.kl_weight * kl_loss


            # total loss
            total_loss = rec_loss + p_loss + generator_adv_loss + codebook_loss + kl_loss
            
                        
            log_dict = {
                "Loss/generator/total_loss": total_loss.clone().detach().mean(),

                "Loss/generator/quantizer_loss": codebook_loss.detach().mean(),    # Quantizer Loss = Codebook and Commitment Loss
                "Loss/generator/codebook_weight": self.codebook_weight,

                "Loss/generator/reconstruction_loss": rec_loss.detach().mean(),    # Reconstruction Loss = Pixel-wise L1/L2 Loss
                "Loss/generator/reconstruction_weight": self.reconstruction_weight,

                "Loss/generator/perceptual_loss": p_loss.detach().mean(),    # Perceptual Loss = Featurewise Loss via LPIPS or Triplane
                "Loss/generator/perceptual_weight": self.perceptual_weight_loss,

                "Loss/generator/adversarial_loss": generator_adv_loss.detach().mean(),    # Generator Adversarial Loss = how much discriminator is fooled
                "Loss/generator/discriminator_adaptive_weight": disc_adaptive_weight if isinstance(disc_adaptive_weight, torch.Tensor) else torch.tensor(disc_adaptive_weight),
                "Loss/generator/discriminator_weight": disc_weight,

                "Loss/generator/kl_loss": kl_loss.detach().mean(),        # KL Loss = how much posterior is different from prior
                "Loss/generator/kl_weight": self.kl_weight,
            }
                    
            return total_loss, log_dict

        # discriminator update
        if optimizer_idx == 1:
            ### No calculations needed for discriminator update before disc_start
            if global_step < self.disc_start:
                # Return None for skipping optimization, but still log zeros for consistency
                return None, {
                    "Loss/discriminator/adversarial_loss": 0.0,
                    "Loss/discriminator/discriminator_weight": 0.0,
                    "Loss/discriminator/discriminator_cr_loss": 0.0,
                    "Loss/discriminator/discriminator_cr_loss_weight": 0.0,
                    "Loss/discriminator/discriminator_cr_weight": self.disc_cr_loss_weight,
                    "Loss/discriminator/lecam_loss": 0.0,
                    "Loss/discriminator/lecam_loss_weight": self.lecam_loss_weight,
                }

            if self.dims == 3 and self.disc_crop_size is not None:
                inputs, reconstructions = random_crop_3d([inputs, reconstructions], self.disc_crop_size)

            if self.use_diff_aug:
                logits_real = self.discriminator(DiffAugment(inputs.contiguous().detach(), policy='color,translation,cutout_0.2', prob=0.5))
                logits_fake = self.discriminator(DiffAugment(reconstructions.contiguous().detach(), policy='color,translation,cutout_0.2', prob=0.5))
            else:
                logits_real = self.discriminator(inputs.contiguous().detach())
                logits_fake = self.discriminator(reconstructions.contiguous().detach())

            disc_weight = adopt_weight(self.disc_weight, global_step, threshold=self.disc_start, warmup_steps=self.disc_warmup)

            lecam_loss = torch.tensor(0.0)
            if self.lecam_loss_weight is not None:
                self.lecam_loss.update(logits_real, logits_fake)
                lecam_loss = lecam_reg(logits_real, logits_fake, self.lecam_loss)
                non_saturate_d_loss = self.disc_loss(logits_real, logits_fake)
                d_adversarial_loss = disc_weight * (lecam_loss * self.lecam_loss_weight + non_saturate_d_loss)
            else:
                d_adversarial_loss = disc_weight * self.disc_loss(logits_real, logits_fake)    # Discriminator Adversarial Loss = how much discriminator is fooled

            
            if self.disc_cr_loss_weight:    # CR Loss = Contrastive Regularization Loss -> https://arxiv.org/pdf/2006.10738 = Discriminator should behave the same for contrast adaptions etc... 
                logits_real_s = self.discriminator(DiffAugment(inputs.contiguous().detach(), policy='color,translation,cutout_0.5', prob=1.0))
                logits_fake_s = self.discriminator(DiffAugment(reconstructions.contiguous().detach(), policy='color,translation,cutout_0.5', prob=1.0))
                disc_cr_loss_weight = self.disc_cr_loss_weight if global_step >= self.disc_start else 0.0
                d_cr = F.mse_loss(torch.cat([logits_real, logits_fake], dim=0), torch.cat([logits_real_s, logits_fake_s])) * disc_cr_loss_weight
                d_adversarial_loss += d_cr

            log_dict = {
                "Loss/discriminator/adversarial_loss": d_adversarial_loss,
                "Loss/discriminator/discriminator_weight": disc_weight,
                "Loss/discriminator/discriminator_cr_loss": d_cr,
                "Loss/discriminator/discriminator_cr_loss_weight": disc_cr_loss_weight,
                "Loss/discriminator/discriminator_cr_weight": self.disc_cr_loss_weight,
                "Loss/discriminator/lecam_loss": lecam_loss,
                "Loss/discriminator/lecam_loss_weight": self.lecam_loss_weight,
            }
            
            return d_adversarial_loss, log_dict


import torch
import torch.nn as nn
import torch.nn.functional as F
class ActNorm(nn.Module):
    def __init__(self, num_features, logdet=False, affine=True,
                 allow_reverse_init=False):
        assert affine
        super().__init__()
        self.logdet = logdet
        self.loc = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        self.scale = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.allow_reverse_init = allow_reverse_init

        self.register_buffer('initialized', torch.tensor(0, dtype=torch.uint8))

    def initialize(self, input):
        with torch.no_grad():
            flatten = input.permute(1, 0, 2, 3).contiguous().view(input.shape[1], -1)
            mean = (
                flatten.mean(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
                .permute(1, 0, 2, 3)
            )
            std = (
                flatten.std(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
                .permute(1, 0, 2, 3)
            )

            self.loc.data.copy_(-mean)
            self.scale.data.copy_(1 / (std + 1e-6))

    def forward(self, input, reverse=False):
        if reverse:
            return self.reverse(input)
        if len(input.shape) == 2:
            input = input[:,:,None,None]
            squeeze = True
        else:
            squeeze = False

        _, _, height, width = input.shape

        if self.training and self.initialized.item() == 0:
            self.initialize(input)
            self.initialized.fill_(1)

        h = self.scale * (input + self.loc)

        if squeeze:
            h = h.squeeze(-1).squeeze(-1)

        if self.logdet:
            log_abs = torch.log(torch.abs(self.scale))
            logdet = height*width*torch.sum(log_abs)
            logdet = logdet * torch.ones(input.shape[0]).to(input)
            return h, logdet

        return h

    def reverse(self, output):
        if self.training and self.initialized.item() == 0:
            if not self.allow_reverse_init:
                raise RuntimeError(
                    "Initializing ActNorm in reverse direction is "
                    "disabled by default. Use allow_reverse_init=True to enable."
                )
            else:
                self.initialize(output)
                self.initialized.fill_(1)

        if len(output.shape) == 2:
            output = output[:,:,None,None]
            squeeze = True
        else:
            squeeze = False

        h = output / self.scale - self.loc

        if squeeze:
            h = h.squeeze(-1).squeeze(-1)
        return h


class LeCAM_EMA(nn.Module):
    def __init__(self, init=0., decay=0.999, device='cuda'):
        super().__init__()
        self.decay = decay
        # Register buffers so they move with model and stay on correct device
        self.register_buffer('logits_real_ema', torch.tensor(init, device=device))
        self.register_buffer('logits_fake_ema', torch.tensor(init, device=device))

    def update(self, logits_real, logits_fake):
        # Ensure input tensors are on the same device
        logits_real = logits_real.to(self.logits_real_ema.device)
        logits_fake = logits_fake.to(self.logits_fake_ema.device)
        
        # Update EMA in-place
        self.logits_real_ema.mul_(self.decay).add_(torch.mean(logits_real).detach() * (1 - self.decay))
        self.logits_fake_ema.mul_(self.decay).add_(torch.mean(logits_fake).detach() * (1 - self.decay))

def lecam_reg(real_pred, fake_pred, lecam_ema):
    reg = torch.mean(F.relu(real_pred - lecam_ema.logits_fake_ema).pow(2)) + \
          torch.mean(F.relu(lecam_ema.logits_real_ema - fake_pred).pow(2))
    return reg

def get_conv_layer(dims):
    if dims == 2:
        return nn.Conv2d
    elif dims == 3:
        return nn.Conv3d
    else:
        raise ValueError(f"Unsupported number of dimensions: {dims}")


def get_norm_layer(dims, use_actnorm=False):
    if use_actnorm:
        return ActNorm
    if dims == 2:
        return nn.BatchNorm2d
    elif dims == 3:
        return nn.BatchNorm3d
    else:
        raise ValueError(f"Unsupported number of dimensions: {dims}")

# zero weight until threshold is reached
def adopt_weight(weight, global_step, threshold=0, value=0., warmup_steps=None):
    if global_step < threshold:
        weight = value
        # linear warmup for warmup_steps
    elif warmup_steps is not None:
        weight = weight * min(1.0, abs(global_step-threshold) / warmup_steps)

    return weight


def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def vanilla_d_loss(logits_real, logits_fake):
    d_loss = 0.5 * (
        torch.mean(torch.nn.functional.softplus(-logits_real)) +
        torch.mean(torch.nn.functional.softplus(logits_fake)))
    return d_loss

def non_saturating_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.binary_cross_entropy_with_logits(torch.ones_like(logits_real),  logits_real))
    loss_fake = torch.mean(F.binary_cross_entropy_with_logits(torch.zeros_like(logits_fake), logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss

def non_saturating_gen_loss(logit_fake):
    return torch.mean(F.binary_cross_entropy_with_logits(torch.ones_like(logit_fake),  logit_fake))

def hinge_gen_loss(logit_fake):
    return -torch.mean(logit_fake)



# Differentiable Augmentation for Data-Efficient GAN Training
# Shengyu Zhao, Zhijian Liu, Ji Lin, Jun-Yan Zhu, and Song Han
# https://arxiv.org/pdf/2006.10738
"""Adapted to also work with 3d"""
import torch
import torch.nn.functional as F

from functools import partial


def DiffAugment(x, policy='', prob=0.5, channels_first=True):
    if policy:
        if not channels_first:
            x = x.permute(0, 3, 1, 2)
        for p in policy.split(','):
            if torch.rand(1) > prob:
                continue
            
            for f in AUGMENT_FNS[p]:
                x = f(x)
            
        if not channels_first:
            x = x.permute(0, 2, 3, 1)
        x = x.contiguous()
    return x

def rand_brightness(x):
    # dims: [B, C, *spatial]
    return x + (torch.rand(x.size(0), 1, *([1] * (x.ndim - 2)), dtype=x.dtype, device=x.device) - 0.5)

def rand_saturation(x):
    # dims: [B, C, *spatial]
    x_mean = x.mean(dim=1, keepdim=True)
    return (x - x_mean) * (torch.rand(x.size(0), 1, *([1] * (x.ndim - 2)), dtype=x.dtype, device=x.device) * 2) + x_mean

def rand_contrast(x):
    # dims: [B, C, *spatial]
    x_mean = x.mean(dim=list(range(1, x.ndim)), keepdim=True)
    return (x - x_mean) * (torch.rand(x.size(0), 1, *([1] * (x.ndim - 2)), dtype=x.dtype, device=x.device) + 0.5) + x_mean

def rand_translation(x, ratio=0.125):
    # dims: [B, C, *spatial]
    spatial_dims = x.shape[2:]
    shifts = [int(s * ratio + 0.5) for s in spatial_dims]
    translations = [torch.randint(-sh, sh + 1, size=[x.size(0)], device=x.device) for sh in shifts]

    # pad 1 in each spatial dim
    pad = []
    for _ in spatial_dims[::-1]:
        pad.extend([1, 1])
    x_pad = F.pad(x, pad)

    # build grid
    grids = torch.meshgrid(
        [torch.arange(s, device=x.device) for s in [x.size(0), *spatial_dims]],
        indexing="ij"
    )
    batch_grid = grids[0]
    spatial_grids = list(grids[1:])

    spatial_grids = [
        (g + translations[i].view(-1, *([1] * (g.ndim - 1)))).clamp(0, spatial_dims[i] + 1)
        for i, g in enumerate(spatial_grids)
    ]

    x = x_pad.permute(0, *range(2, x.ndim), 1).contiguous()[batch_grid, *spatial_grids].permute(0, x.ndim - 1, *range(1, x.ndim - 1))
    return x

def rand_cutout(x, ratio=0.2):
    # dims: [B, C, *spatial]
    spatial_dims = x.shape[2:]
    cutout_sizes = [int(s * ratio + 0.5) for s in spatial_dims]

    offsets = [torch.randint(0, s + (1 - c % 2), size=[x.size(0)], device=x.device)
               for s, c in zip(spatial_dims, cutout_sizes)]

    # build grid
    mesh = torch.meshgrid(
        [torch.arange(x.size(0), device=x.device)] +
        [torch.arange(c, device=x.device) for c in cutout_sizes],
        indexing="ij"
    )
    batch_grid = mesh[0]
    spatial_grids = list(mesh[1:])

    spatial_grids = [
        (g + offsets[i].view(-1, *([1] * (g.ndim - 1))) - (cutout_sizes[i] // 2))
        .clamp(0, spatial_dims[i] - 1)
        for i, g in enumerate(spatial_grids)
    ]

    mask = torch.ones(x.size(0), *spatial_dims, dtype=x.dtype, device=x.device)
    mask[batch_grid, *spatial_grids] = 0
    return x * mask.unsqueeze(1)

AUGMENT_FNS = {
    'color': [rand_brightness, rand_saturation, rand_contrast],
    'translation': [rand_translation],
    'cutout_0.2': [partial(rand_cutout, ratio=0.2)],
    'cutout_0.5': [partial(rand_cutout, ratio=0.5)],
}


if __name__ == "__main__":
    weight = adopt_weight(0.1, 15000, threshold=10000, value=0.0, warmup_steps=5000)
    print(weight)
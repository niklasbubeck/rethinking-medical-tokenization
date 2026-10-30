from sklearn import metrics
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import pandas as pd
import os 
import torch.nn.functional as F
import torch_fidelity

from ..utils import *
from ..metrics.metrics import CompositeMetrics

from .base_evaluator import EvaluationTask


class DiTUncondGeneration(EvaluationTask):
    def __init__(self, name, metrics: list, input_key=None, save_as=['gif', 'nii.gz'], n_samples=None, diffusion_cfg=None, sampler='ddpm', clip_denoised=False, temperature=1.0):
        super().__init__(name, metrics, save_as, sampler=sampler, clip_denoised=clip_denoised, temperature=temperature)
        assert sampler in ["ddpm", "ddim"], f"Can only select out of 'ddpm' and 'ddim' or you have to define yourself!"
        self.name = name
        self.input_key = input_key
        self.n_samples = n_samples 
        self.clip_denoised = clip_denoised
        self.temperature = temperature
        self.diffusion = instantiate_from_config(diffusion_cfg)
        self.sampler_fcn = self.diffusion.p_sample_loop if sampler == "ddpm" else self.diffusion.ddim_sample_loop

    @torch.no_grad()
    def process_batch(self, model, batch, output_dir=None):
        model = self.unwrap_model(model)
        gts, labels = batch[self.input_key], batch['label']
        try:    
            gts_ = model.vae_encode(gts)
        except:
            gts_ = gts
        shape = gts_.shape

        cond = dict(y=labels)

        sample = self.sampler_fcn(model, shape, clip_denoised=self.clip_denoised, progress=False, model_kwargs=cond, temperature=self.temperature)
        try:
            sample = model.vae_decode(sample)
        except:
            sample = sample

        return gts, sample, torch.ones_like(gts)


class MedTokDiTCondGeneration(EvaluationTask):
    def __init__(self, name, metrics: list, input_key=None, save_as=['gif', 'nii.gz'], n_samples=None, diffusion_cfg=None, sampler='ddpm', clip_denoised=False, temperature=1.0, cfg_scale=5.0):
        super().__init__(name, metrics, save_as, sampler=sampler, clip_denoised=clip_denoised, temperature=temperature, cfg_scale=cfg_scale)
        assert sampler in ["ddpm", "ddim"], f"Can only select out of 'ddpm' and 'ddim' or you have to define yourself!"
        self.name = name
        self.input_key = input_key
        self.n_samples = n_samples 
        self.clip_denoised = clip_denoised
        self.temperature = temperature
        self.cfg_scale = cfg_scale
        print(f"cfg_scale: {self.cfg_scale}")
        self.diffusion = instantiate_from_config(diffusion_cfg)
        self.sampler_fcn = self.diffusion.p_sample_loop if sampler == "ddpm" else self.diffusion.ddim_sample_loop

    @torch.no_grad()
    def process_batch(self, model, batch, output_dir=None):
        model = self.unwrap_model(model)
        gts, labels, dataset_ids = batch['image'], batch['label'], batch['dataset_id']
        labels = labels.squeeze(1)
        gts_ = model.vae_encode(gts)

        shape = gts_.shape

        cond = dict(y=labels, dataset_id=dataset_ids, cfg_scale=self.cfg_scale, cfg_dataset=False)

        print(f"dataset_ids: {dataset_ids}")
        sample = self.sampler_fcn(model.generator.forward_with_cfg, shape, clip_denoised=self.clip_denoised, progress=False, model_kwargs=cond, temperature=self.temperature, device=gts.device)
        sample = model.vae_decode(sample)


        return gts, sample, torch.ones_like(gts)


class ViTADiTCondGeneration(EvaluationTask):
    def __init__(self, name, metrics: list, input_key=None, save_as=['gif', 'nii.gz'], n_samples=None, diffusion_cfg=None, sampler='ddpm', clip_denoised=False, temperature=1.0):
        super().__init__(name, metrics, save_as, sampler=sampler, clip_denoised=clip_denoised, temperature=temperature)
        assert sampler in ["ddpm", "ddim"], f"Can only select out of 'ddpm' and 'ddim' or you have to define yourself!"
        self.name = name
        self.input_key = input_key
        self.n_samples = n_samples 
        self.clip_denoised = clip_denoised
        print(f"clip_denoised: {self.clip_denoised}")
        self.temperature = temperature
        self.diffusion = instantiate_from_config(diffusion_cfg)
        self.sampler_fcn = self.diffusion.p_sample_loop if sampler == "ddpm" else self.diffusion.ddim_sample_loop

    @torch.no_grad()
    def process_batch(self, model, batch, output_dir=None):
        model = self.unwrap_model(model)
        gts, labels = batch[self.input_key], batch['label']
        gts_, cls_token, shape = model.vae_encode(gts)
        print(f"gts_: {gts_.min()}, {gts_.max()}, {gts_.mean()}, {gts_.std()}")
        shape = gts_.shape

        cond = dict(y=cls_token.squeeze(1))

        sample = self.sampler_fcn(model, shape, clip_denoised=False, progress=False, model_kwargs=cond)
        print(f"sample: {sample.min()}, {sample.max()}, {sample.mean()}, {sample.std()}")
        print(f"sample shape: {sample.shape}")
        sample = model.vae_decode(sample, cls_token, shape)

        return gts, sample, torch.ones_like(gts)

class DiTInpainting(EvaluationTask):
    def __init__(self, name, metrics: list, input_key=None, save_as=['gif', 'nii.gz'], n_samples=None, diffusion_cfg=None, mask_ratio=0.5, clip_denoised=False, temperature=1.0):
        super().__init__(name, metrics, save_as, input_key=input_key, mask_ratio=mask_ratio, clip_denoised=clip_denoised, temperature=temperature)
        self.name = name
        self.input_key = input_key
        self.n_samples = n_samples 
        self.mask_ratio = mask_ratio
        self.temperature = temperature
        self.clip_denoised = clip_denoised
        self.diffusion = instantiate_from_config(diffusion_cfg)
        self.sampler_fcn = self.diffusion.p_sample_loop_inpainting

    @torch.no_grad()
    def process_batch(self, model, batch, output_dir=None):
        model = self.unwrap_model(model)
        gts, labels = batch[self.input_key], batch['label']
        try:    
            gts_ = model.vae_encode(gts)
        except:
            gts_ = gts
        shape = gts_.shape
        b, c, h, w = gts_.shape

        mask_path = f"./masks/mask_{self.mask_ratio}.png"
        mask_orig = torch.from_numpy(np.array(Image.open(mask_path).convert('L')) > 128).float()
        print(mask_orig)
        # mask_orig = torch.rand((1, 1, h, w)) > self.mask_ratio
        mask = mask_orig.expand(b, c, h, w)
        cond = dict(y=labels)

        sample = self.sampler_fcn(model, shape, x_0=gts_, mask=mask.int(), clip_denoised=self.clip_denoised, progress=False, model_kwargs=cond, temperature=self.temperature)
        try:
            sample = model.vae_decode(sample)
        except:
            sample = sample

        mask_orig_interpolated = torch.nn.functional.interpolate(
            mask.float(), size=gts.shape[-2:], mode='nearest'
        ).int().to(gts.device)
        return gts, sample, mask_orig_interpolated[:, :3, :, :]

class DiTCondGeneration(EvaluationTask): 
    def __init__(self, *args, **kwargs):
        raise NotImplementedError

class DiTSuperresolution(EvaluationTask): 
    def __init__(self, *args, **kwargs): 
        raise NotImplementedError
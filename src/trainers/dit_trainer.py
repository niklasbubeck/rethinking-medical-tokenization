import torch 
import numpy as np
import matplotlib.pyplot as plt

from src.utils import instantiate_from_config, init_schedulers, init_optimizers
from .all_trainer import AllTrainer

class DiTTrainer(AllTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.diffusion = kwargs.get("diffusion", None)
        self.diffusion = self.accelerator.prepare(self.diffusion)


    def get_model_loss(self, model, batch, *args, **kwargs):
        # get data 
        x, labels = batch[self.input_key], batch['label']
        cond = dict(y=labels)

        # prepare time conditioning and diffusion noise
        t = torch.randint(0, self.diffusion.num_timesteps, (x.shape[0],), device=x.device)
        x_ = self.unwrap_model(model).vae_encode(x)

        with self.accelerator.autocast():
            terms = self.diffusion.training_losses(model, x_, t, model_kwargs=cond)

        loss_simple = terms["loss"].mean()

        return loss_simple

class ViTADiTTrainer(AllTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.diffusion = kwargs.get("diffusion", None)
        self.diffusion = self.accelerator.prepare(self.diffusion)

    def get_model_loss(self, model, batch, *args, **kwargs):
        # get data 
        x, labels = batch[self.input_key], batch['label']

        # prepare time conditioning and diffusion noise
        t = torch.randint(0, self.diffusion.num_timesteps, (x.shape[0],), device=x.device)
        x_, cls_token, shape = self.unwrap_model(model).vae_encode(x)
        cond = dict(y=cls_token.squeeze(1))
        # print(f"cls_token: {cls_token.shape}")

        with self.accelerator.autocast():
            terms = self.diffusion.training_losses(model, x_, t, model_kwargs=cond)

        loss_simple = terms["loss"].mean()

        return loss_simple

class MedTokDiTTrainer(AllTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.diffusion = kwargs.get("diffusion", None)
        self.diffusion = self.accelerator.prepare(self.diffusion)


    def get_model_loss(self, model, batch, *args, **kwargs):
        # get data 
        x, labels, dataset_ids = batch["image"], batch["label"], batch["dataset_id"]
        labels = labels.squeeze(1)
        cond = dict(y=labels, dataset_id=dataset_ids)

        # prepare time conditioning and diffusion noise
        t = torch.randint(0, self.diffusion.num_timesteps, (x.shape[0],), device=x.device)
        x_ = self.unwrap_model(model).vae_encode(x)
        with self.accelerator.autocast():
            terms = self.diffusion.training_losses(model, x_, t, model_kwargs=cond)

        loss_simple = terms["loss"].mean()

        return loss_simple
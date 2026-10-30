import torch
from .all_trainer import AllTrainer
from src.utils import instantiate_from_config
from tqdm import tqdm

class VQGANTrainer(AllTrainer):
    
    def get_model_loss(self, model, batch, loss_fn, *args, **kwargs):
        def get_last_layer():
            return self.unwrap_model(model).decoder.get_last_layer()
        
        x = batch[self.input_key]

        xrec, qloss = model(x)
        # Determine which network to update this step
        is_gen_step = self.steps % 2 == 0
        losses = {}
        
        if self.normalize:
            x = (x + 1) / 2
            xrec = (xrec + 1) / 2

        losses = {}

        # Always update generator if we're before discriminator start
        if self.steps % 2 == 0 or self.steps < self.unwrap_model(loss_fn).disc_start:
            # Generator loss
            aeloss, log_dict_ae = loss_fn(
                qloss, x.to(torch.float32), xrec.to(torch.float32), 0, self.steps,
                last_layer=get_last_layer(), posteriors=None, split="train"
            )
            losses['gen'] = aeloss.mean()
            losses['disc'] = None
            self.add_log.update(log_dict_ae)
        else:
            # Discriminator loss (only after disc_start)
            discloss, log_dict_disc = loss_fn(
                qloss, x.to(torch.float32), xrec.to(torch.float32), 1, self.steps,
                last_layer=get_last_layer(), split="train"
            )
            losses['gen'] = None
            losses['disc'] = discloss.mean() if discloss is not None else None
            self.add_log.update(log_dict_disc)


        return losses
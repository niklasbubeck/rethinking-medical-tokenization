import os
from tqdm import tqdm
import torch
import torch.nn as nn
import time
from abc import abstractmethod, ABC

from omegaconf import DictConfig

from ..utils import MultiMeter, delay2str, get_model_hash, get_torchvision_transforms, get_monai_transforms, init_schedulers
from .utils import DummyProfiler, ProfilerWrapper, load_optimizers, load_schedulers
from src.accelerator import find_executable_batch_size

import random
import numpy as np


class AllTrainer(nn.Module, ABC):
    def __init__(self, 
                 model,
                 train_loader,
                 accelerator,
                 meta_info: dict = None,
                 eval_loader=None,
                 optimizers=None,
                 ema_model=None,
                 schedulers=None,
                 early_stopper=None,
                 evaluator=None,
                 loss_fn=None,
                 scaler=None,
                 diffusion=None,
                 n_epochs=4000, 
                 log_interval=50, 
                 save_interval=10, 
                 output_dir="outputs",
                 experiment_name=None,
                 reduction_strategy="exponential", 
                 normalize=False,
                 grad_clip_norm=5.0,
                 input_key="data",
                 use_profiler=False,
                 gpu_transforms=None,
                 **kwargs):
        
        super().__init__()
        
        # Validate inputs and mixed precision settings
        assert model is not None and train_loader is not None, "model and train_loader cannot be None"
        assert not (accelerator.mixed_precision in ['fp16', 'bf16'] and scaler is not None), \
            "Cannot use custom scaler with Accelerator's mixed precision"
        
        # Store model components as nn.Module attributes
        self.model = model
        self.ema_model = ema_model
        self.loss_fn = loss_fn
        
        # Initialize training components
        self.optimizers = optimizers
        self.schedulers = schedulers
        self.scaler = scaler
        self.early_stopper = early_stopper
        self.evaluator = evaluator
        self.diffusion = diffusion
        # Store data loaders
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        
        # Store training settings
        self.accelerator = accelerator
        self.reduction_strategy = reduction_strategy
        self.normalize = normalize
        self.log_interval = log_interval
        self.save_interval = save_interval
        
        # Initialize metadata
        meta_info = meta_info or {'start_time': time.time(), 'train_days': 1, 'steps': 0, 'epoch': 1}
        self.start_time = meta_info.get('start_time', time.time())
        self.train_days = meta_info.get('train_days', 1)
        self.steps = meta_info.get('steps', 0)
        self.epoch = meta_info.get('epoch', 1)
        self.n_epochs = n_epochs - self.epoch + 1
        self.input_key = input_key
        self.gpu_transforms = get_torchvision_transforms(gpu_transforms) if gpu_transforms is not None else None
        
        # Setup directories and logging
        self.model_name = model.__class__.__name__.lower()
        self.experiment_name = experiment_name or f"run_{time.strftime('%Y%m%d_%H%M%S')}"
        self._setup_dirs(output_dir)
        
        # Initialize logging components
        self.use_profiler = use_profiler
        self.add_log = {}
        self.multi_meter = MultiMeter()
        
        self.grad_clip_norm = grad_clip_norm
        # Store wandb run ID if provided in meta_info
        self.wandb_run_id = meta_info.get('wandb_run_id', None)

        ### before everything is prepared
        self.before_prepare()
        # prepare everything 
        self.model, self.ema_model, self.loss_fn, self.optimizers, self.schedulers, self.train_loader, self.eval_loader = self.accelerator.prepare(self.model, self.ema_model, self.loss_fn, self.optimizers, self.schedulers, self.train_loader, self.eval_loader)
        self.accelerator.wait_for_everyone()

    def __str__(self):
        return (
            f"AllTrainer(\n"
            f"  model={self.model.__class__.__name__},\n"
            f"  ema_model={self.ema_model.__class__.__name__ if self.ema_model else 'None'},\n"
            f"  loss_fn={self.loss_fn.__class__.__name__ if self.loss_fn else 'None'},\n"
            f"  optimizers={', '.join([f'{k}: {self._get_real_class(v).__name__}' for k, v in self.optimizers.items()]) if isinstance(self.optimizers, dict) else self._get_real_class(self.optimizers).__name__ if self.optimizers else 'None'},\n"
            f"  schedulers={', '.join([f'{k}: {self._get_real_class(v).__name__}' for k, v in self.schedulers.items()]) if isinstance(self.schedulers, dict) else self._get_real_class(self.schedulers).__name__ if self.schedulers else 'None'},\n"
            f"  train_loader={self.train_loader.__class__.__name__},\n"
            f"  eval_loader={self.eval_loader.__class__.__name__},\n"
            f"  n_epochs={self.n_epochs},\n"
            f"  steps={self.steps},\n"
            f"  train_days={self.train_days},\n"
            f"  experiment_name={self.experiment_name},\n"
            f"  output_dir={self.output_dir}\n"
            f"  log_interval={self.log_interval},\n"
            f"  save_interval={self.save_interval},\n"
            f"  grad_clip_norm={self.grad_clip_norm},\n"
            f"  normalize={self.normalize}\n"
            f"  wandb_run_id={self.wandb_run_id}\n"
            f"  early_stopper={self.early_stopper.__class__.__name__ if self.early_stopper else 'None'},\n"
            f")"
        )

    def _setup_dirs(self, base_output_dir):
        """Setup output directory structure."""
        self.output_dir = os.path.abspath(base_output_dir)
        self.ckpt_dir = os.path.join(self.output_dir, "train/models")
        self.gen_dir = os.path.join(self.output_dir, "train/media")
        self.profiler_dir = os.path.join(self.output_dir, "train/profiler_logs")
        
        for dir_path in [self.output_dir, self.ckpt_dir, self.gen_dir, self.profiler_dir]:
            os.makedirs(dir_path, exist_ok=True)
            
        if self.is_main:
            self.print(f"Output directories created: {self.output_dir}")

    def before_prepare(self):
        """Entry point for training."""
        if self.optimizers is not None and self.schedulers is not None:
            return
        if self.optimizers is None:
            self.optimizers = self.setup_optimizers()
            if self.optimizers is None:
                raise ValueError("No optimizers provided or setup_optimizers() not implemented")
                
        if self.schedulers is None and hasattr(self, 'setup_schedulers') and self.optimizers is not None:
            self.schedulers = self.setup_schedulers(self.optimizers)
        
        elif self.schedulers is None and self.optimizers is not None:
            self.schedulers = init_schedulers(self.optimizers, epoch_to_steps=len(self.train_loader))
        else: 
            raise ValueError("No schedulers provided or setup_schedulers() not implemented")


    def train(self):
        return self.train_loop()

    def train_loop(self):
        """Main training loop."""
        torch.autograd.set_detect_anomaly(False)
        self.print(f"Training Start... (Epoch {self.epoch} / {self.epoch + self.n_epochs -1})")

        try:
            for i in range(self.n_epochs):
                if i == 0 and self.epoch > 1:
                    self.print("Training is starting from resume, running evaluation first...")
                    self.evaluate()
                    
                self._run_epoch()  # profiler is handled inside _run_epoch
                
                if self.accelerator.check_trigger():
                    self.print(f"Early Stopping triggered at epoch {self.epoch}")
                    break
                    
                self.epoch += 1
                
            return self._finish_training()
            
        except Exception as e:
            self.print(f"Training failed at epoch {self.epoch}: {str(e)}", mode="error")
            raise

    def _run_epoch(self):
        """Run a single training epoch."""
        self.model.train()
        if self.ema_model is not None and hasattr(self.ema_model, 'train'):
            self.ema_model.train()
            
        epoch_start = time.time()
        self.multi_meter.reset()
        
        pbar = self.multi_meter.log_every(self.train_loader, len(self.train_loader), disable=not self.is_main)
        # Only profile the first epoch
        self.use_profiler = self.use_profiler and (self.epoch == 1)
        with ProfilerWrapper(self.use_profiler, self._get_profiler_args()) as profiler:
            for self.iter, batch in enumerate(pbar):
                self.multi_meter.update("steps", self.steps)
                if 't_get' in batch.keys():
                    self.multi_meter.update("t_get", batch['t_get'].mean().item())
                if 't_io' in batch.keys():
                    self.multi_meter.update("t_io", batch['t_io'].mean().item())
                with self.accelerator.accumulate(self.model):
                    if self.gpu_transforms is not None:
                        start_time = time.time()
                        batch[self.input_key] = self.gpu_transforms(batch[self.input_key])
                        self.multi_meter.update("t_gpu_trans", time.time() - start_time)
                    losses = self.get_model_loss(self.model, batch, self.loss_fn)
                    if self.is_main: self._update_metrics(losses)
                    self._optimize_loss(losses)
                    if self.schedulers:
                        self._step_schedulers()
                if self.is_main:
                    self.accelerator.log(self.multi_meter.get_latest(), step=self.steps)
                self.steps += 1
                profiler.step()  # safe for both real and dummy
                # Log the latest values from multi_meter
        # Run evaluation if needed
        if self.eval_loader and (self.epoch % self.log_interval == 0 or self.epoch == self.n_epochs - 1):
            self.evaluate()
            
        # Save checkpoint if needed
        if self.is_main and self.epoch % self.save_interval == 0:
            self.save_ckpt(f'ckpt.{self.train_days}.pt')
        

        self.print(f"Epoch {self.epoch} finished in {delay2str(time.time() - epoch_start)}")

    def _optimize_loss(self, losses):
        """Optimize single or multiple losses."""
        if isinstance(losses, (dict, DictConfig)):
            for loss_name, loss in losses.items():
                if loss is not None:
                    if loss_name not in self.optimizers:
                        self.print(f"[Warning] No optimizer found for {loss_name}. Skipping optimizer step.", mode="error")
                    else:
                        self._step_optimizer(loss, self.optimizers[loss_name])
                # else:
                #     self.print(f"[Warning] No loss found for {loss_name}. Skipping optimizer step. This might be intentional for the discriminator update before disc_start.", mode="warning")
        else:
            self._step_optimizer(losses, self.optimizers, self.scaler)
            
        # Update EMA model if it exists
        if self.ema_model is not None:
            self.ema_model.update()

    def _step_optimizer(self, loss, optimizer, scaler=None):
        """Perform a single optimization step."""
        if torch.isnan(loss) or torch.isinf(loss):
            self.print(f"[Warning] Invalid loss detected at step {self.steps}: {loss.item()}", mode="error")
            raise ValueError("NaN or Inf loss detected.")
            
        optimizer.zero_grad()
        
        if scaler is not None:
            scaler(loss, optimizer, parameters=self.model.parameters(), update_grad=True)
        else:
            self.accelerator.backward(loss)
            self.accelerator.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
            optimizer.step()


    def _step_schedulers(self):
        """Step all schedulers."""
        if isinstance(self.schedulers, (dict, DictConfig)):
            for scheduler in self.schedulers.values():
                if scheduler is not None:
                    scheduler.step()
        elif self.schedulers is not None:
            self.schedulers.step()

    @torch.no_grad()
    def _update_metrics(self, losses):
        """Update training metrics."""
        log_dict = {}
        
        # Log epoch
        self.multi_meter.update("epoch", self.epoch)
        log_dict["Train/epoch"] = self.epoch
        
        # Log losses
        if isinstance(losses, dict):
            for name, loss in losses.items():
                if loss is not None:
                    self.multi_meter.update(f"loss_{name}", loss.detach().item())
                    log_dict[f"Train/loss_{name}"] = loss.detach().item()
                    log_dict[f"Train/lr_{name}"] = self.schedulers[name].get_last_lr()[0]
        elif losses is not None:
            self.multi_meter.update("loss", losses.detach().item())
            log_dict["Train/loss"] = losses.detach().item()
            log_dict["Train/lr"] = self.schedulers.get_last_lr()[0]
            
        # Add any additional logs
        if self.add_log:
            for key, value in self.add_log.items():
                if isinstance(value, torch.Tensor):
                    value = value.float().mean().detach().item()
                log_dict[key] = value
                
        # Log to wandb
        if self.is_main:
            self.accelerator.log(log_dict, step=self.steps)

    def _get_profiler_args(self):
        """Get profiler configuration."""
        return {
            'activities': [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            'schedule': torch.profiler.schedule(wait=3, warmup=5, active=8, repeat=1),
            'on_trace_ready': torch.profiler.tensorboard_trace_handler(self.profiler_dir),
            'record_shapes': False,
            'profile_memory': False,
            'with_stack': False
        }

    @abstractmethod
    def get_model_loss(self, model, batch, loss_fn):
        """Calculate model loss(es). Must be implemented by subclasses."""
        pass

    def evaluate(self):
        """Run evaluation if eval_loader exists."""
        if not self.eval_loader:
            return
            
        self.model.eval()
        metrics = {}
        
        # Store current step to ensure consistent logging
        current_step = self.steps
        
        # Run evaluation loop if implemented
        eval_metrics = self.evaluation_loop(self.model, self.ema_model, self.eval_loader, self.loss_fn)
        if eval_metrics:
            metrics.update(eval_metrics)
            
        # Run evaluator if it exists
        if self.evaluator:
            evaluator_metrics = self._run_evaluator(dataloader=self.eval_loader, mode="eval")
            if evaluator_metrics:
                metrics.update(evaluator_metrics)

            train_evaluator_metrics = self._run_evaluator(dataloader=self.train_loader, mode="eval_train")
            if train_evaluator_metrics:
                metrics.update(train_evaluator_metrics)
                
        # Handle early stopping
        if self.early_stopper and metrics:
            if self.early_stopper.check_early_stopping(metrics):
                self.accelerator.set_trigger()

        if self.is_main and self.early_stopper:
            if self.early_stopper.is_best_overall():
                self.save_ckpt(f'ckpt.best.pt')

        self.model.train()

    def _run_evaluator(self, dataloader, mode="eval"):
        """Run the evaluator on model and EMA model."""
        metrics = {}
        
        # Evaluate main model
        _, model_metrics = self.evaluator(
            model=self.model,
            dataloader=dataloader,
            mode=mode,
            normalize=self.normalize,
            output_dir=self.gen_dir,
            step=self.steps,
            accelerator=self.accelerator
        )
        if model_metrics:
            metrics.update({f"model_{k}": v for k, v in model_metrics.items()})
            
        # Evaluate EMA model if it exists
        if self.ema_model:
            _, ema_metrics = self.evaluator(
                model=self.ema_model.ema_model,
                dataloader=dataloader,
                mode=mode,
                normalize=self.normalize,
                output_dir=self.gen_dir,
                step=self.steps,
                accelerator=self.accelerator,
                is_ema=True
            )
            if ema_metrics:
                metrics.update({f"ema_{k}": v for k, v in ema_metrics.items()})
                
        return metrics

    def save_ckpt(self, name):
        """Save checkpoint with all states using unified state_dict."""
        if not self.is_main:
            return
        
        # Get the complete trainer state
        state = self.state_dict()
        
        torch.save(state, os.path.join(self.ckpt_dir, name))
        self.print(f"Checkpoint saved: {name} ... in {self.ckpt_dir}")


    def _finish_training(self):
        """Finish training and return best metrics."""
        self.print("Training finished.")
        best_metrics = self.early_stopper.get_best_metrics() if self.early_stopper else None
        
        if best_metrics and self.is_main:
            self.print(f"Best metrics: {best_metrics}")
            self.accelerator.log({f"best_{k}": v for k, v in best_metrics.items()})
            
        self.accelerator.wait_for_everyone()
        self.accelerator.end_training()
        return best_metrics

    def print(self, msg, mode="info"):
        """Print message if on main process."""
        if self.is_main:
            print(msg)

    @property
    def is_main(self):
        """Check if this is the main process."""
        return self.accelerator.is_main_process

    # def setup_optimizers(self):
    #     """Optional method to set up optimizers."""
    #     return None

    # def setup_schedulers(self):
    #     """Optional method to set up schedulers."""
    #     return None

    def evaluation_loop(self, model, ema_model, eval_loader, loss_fn):
        """Optional evaluation loop implementation."""
        return {}

    def unwrap_model(self, model):
        """Unwrap model if it's wrapped by Accelerator."""
        return self.accelerator.unwrap_model(model)

    def _get_real_class(self, obj):
        # For AcceleratedOptimizer
        if hasattr(obj, 'optimizer'):
            return type(obj.optimizer)
        # For AcceleratedScheduler (if it exists)
        if hasattr(obj, 'scheduler'):
            return type(obj.scheduler)
        # Fallback
        return type(obj)

    def state_dict(self, *args, **kwargs):
        """Save the complete trainer state including all components and RNG state."""
        state = {}
        
        # Add training metadata
        state.update({
            'epoch': self.epoch,
            'train_days': self.train_days,
            'steps': self.steps,
            'start_time': self.start_time,
            'model_hash': get_model_hash(self.accelerator.unwrap_model(self.model)) if hasattr(self, 'model') else None
        })
        
        # Add model state
        if self.model is not None:
            state['model'] = self.model.state_dict()

        # Add EMA model state if it exists
        if self.ema_model is not None:
            state['ema_model'] = self.ema_model.state_dict()

        # Save wandb run ID if available
        if hasattr(self.accelerator, 'trackers') and self.accelerator.trackers:
            for tracker in self.accelerator.trackers:
                if hasattr(tracker, 'run') and hasattr(tracker.run, 'id'):
                    state['wandb_run_id'] = tracker.run.id
                    break
        
        # Save optimizer states
        if self.optimizers is not None:
            if isinstance(self.optimizers, (dict, DictConfig)):
                state['optimizer'] = {k: v.state_dict() for k, v in self.optimizers.items()}
            else:
                state['optimizer'] = self.optimizers.state_dict()
        
        # Save scheduler states
        if self.schedulers is not None:
            if isinstance(self.schedulers, (dict, DictConfig)):
                state['scheduler'] = {k: v.state_dict() for k, v in self.schedulers.items()}
            else:
                state['scheduler'] = self.schedulers.state_dict()
        
        # Save scaler state
        if self.scaler is not None:
            state['scaler'] = self.scaler.state_dict()
        
        # Save loss function state
        if self.loss_fn is not None:
            state['loss_fn'] = self.loss_fn.state_dict()
        
        if self.early_stopper is not None:
            state['early_stopper'] = self.early_stopper.save_state()

        # Save RNG state
        state['torch_rng_state'] = torch.get_rng_state()
        if torch.cuda.is_available():
            state['cuda_rng_state'] = torch.cuda.get_rng_state_all()
        state['numpy_rng_state'] = np.random.get_state()
        state['random_rng_state'] = random.getstate()
        return state
    
    def load_state_dict(self, state_dict, strict=True):
        """Load the complete trainer state including all components and RNG state."""
        # Do not mutate the caller's dict
        state_dict = dict(state_dict)

        # # recursively remove .module prefix from all dict keys (only for string keys)
        # def remove_module_prefix(obj):
        #     if isinstance(obj, dict):
        #         new_obj = {}
        #         for k, v in obj.items():
        #             if isinstance(k, str):
        #                 new_key = k.replace("module.", "") if k.startswith("module.") else k
        #             else:
        #                 new_key = k
        #             new_obj[new_key] = remove_module_prefix(v)
        #         return new_obj
        #     elif isinstance(obj, list):
        #         return [remove_module_prefix(item) for item in obj]
        #     else:
        #         return obj

        # state_dict = remove_module_prefix(state_dict)
        
        # Extract training metadata
        self.epoch = state_dict.get('epoch', 0)
        self.train_days = state_dict.get('train_days', 1) + 1
        self.steps = state_dict.get('steps', 0)
        self.start_time = state_dict.get('start_time', time.time())
        self.wandb_run_id = state_dict.get('wandb_run_id', None)
        
        # Load model state
        if 'model' in state_dict:
            model_state = state_dict['model']
            self.model.load_state_dict(model_state)
            self.print(f"Model loaded successfully in {self._get_real_class(self.model).__name__}.")

        # Load EMA model state
        if 'ema_model' in state_dict and self.ema_model is not None:
            ema_model_state = state_dict['ema_model']
            self.ema_model.load_state_dict(ema_model_state)
            self.print(f"EMA model loaded successfully in {self._get_real_class(self.ema_model).__name__}.")

        # Load optimizer states
        if 'optimizer' in state_dict:
            optimizer_state = state_dict['optimizer']
            if isinstance(self.optimizers, (dict, DictConfig)) and isinstance(optimizer_state, dict):
                for k, opt_state in optimizer_state.items():
                    if k in self.optimizers:
                        self.optimizers[k].load_state_dict(opt_state)
                        self.print(f"Optimizer '{k}' loaded successfully in {self._get_real_class(self.optimizers[k]).__name__}.")
            elif self.optimizers is not None:
                self.optimizers.load_state_dict(optimizer_state)
                self.print(f"Optimizer loaded successfully in {self._get_real_class(self.optimizers).__name__}.")
        
        # Load scheduler states
        if 'scheduler' in state_dict:
            scheduler_state = state_dict['scheduler']
            if isinstance(self.schedulers, (dict, DictConfig)) and isinstance(scheduler_state, dict):
                for k, sched_state in scheduler_state.items():
                    if k in self.schedulers:
                        self.schedulers[k].load_state_dict(sched_state)
                        self.print(f"Scheduler '{k}' loaded successfully in {self._get_real_class(self.schedulers[k]).__name__}.")
            elif self.schedulers is not None:
                self.schedulers.load_state_dict(scheduler_state)
                self.print(f"Scheduler loaded successfully in {self._get_real_class(self.schedulers).__name__}.")
        
        # Load scaler state
        if 'scaler' in state_dict and self.scaler is not None:
            self.scaler.load_state_dict(state_dict['scaler'])
            self.print(f"Scaler loaded successfully in {self._get_real_class(self.scaler).__name__}.")
        
        # Load loss function state
        if 'loss_fn' in state_dict and self.loss_fn is not None:
            self.loss_fn.load_state_dict(state_dict['loss_fn'])
            self.print(f"Loss function loaded successfully in {self._get_real_class(self.loss_fn).__name__}.")
        
        # Load early stopping state
        if 'early_stopper' in state_dict and self.early_stopper is not None:
            self.early_stopper.load_state(state_dict['early_stopper'])
            self.print(f"Early stopping state loaded successfully.")

        # Evaluate the model
        # print("Evaluating the model... directly after loading the state_dict")
        # self.evaluate()

        return True
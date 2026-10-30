from abc import ABC, abstractmethod
import traceback
import pandas as pd 
from torch.utils.data import DataLoader
from functools import wraps
import time
import itertools
import random 
import traceback
from omegaconf import OmegaConf
import copy
import threading
from functools import partial
import shutil
import os


from ..utils import *
from ..metrics.metrics import CompositeMetrics

# Global logging functions
def log_volume_as_k3d_html(vid_img, caption, accelerator, step=None, normalize=False, **kwargs):
    """
    Visualize a 5D tensor as a 3D volume and log to available trackers.
    
    Args:
    vid_img (torch.Tensor): 5D tensor of shape [B, C, X, Y, Z]
    caption (str): Caption for the visualization
    accelerator: Accelerator instance for distributed training
    step (int, optional): Step or iteration number
    """
    if vid_img.dim() != 5:
        print("WARNING: vid_img must be a 5D tensor to make a k3d volume upload")
        return 

    # Get rid of batch and channel dims
    vid_img = vid_img[0,0,...].clamp(0,1).cpu().detach().numpy()
    x, y, z = vid_img.shape

    # Check if wandb is in the log_with list
    if hasattr(accelerator, 'log_with') and any(logger.value == 'wandb' for logger in accelerator.log_with):
        import wandb
        import k3d  # optional: only needed to log 3D volumes
        plot = k3d.plot()
        volume = k3d.volume(vid_img.astype(np.float32),
                          color_map=k3d.basic_color_maps.Binary,
                          bounds=[0, x, 0, y, 0, z])
        plot += volume
        html_snapshot = plot.get_snapshot()
        wandb.log({f"{caption}/3d_visualization": wandb.Html(html_snapshot)})
    else:
        print(f"WARNING: Wandb is not configured as a tracker. 3D visualization will not be logged.")

def log_videos(videos, name, accelerator, step=None, fps=4, normalize=False):
    """
    Log videos to all available trackers.
    
    Args:
    videos (torch.Tensor): 5D tensor of shape [B, C, T, H, W]
    name (str): Name of the video
    accelerator: Accelerator instance for distributed training
    step (int, optional): Step or iteration number
    fps (int, optional): Frames per second for the video
    normalize (bool): Whether to normalize the video data
    """
    if not isinstance(videos, torch.Tensor):
        raise ValueError("Videos must be a torch.Tensor")

    if normalize: 
        videos = (videos + 1) / 2

    # Extend the vid_img to [B, 3, 1|T, H, W]
    if len(videos.shape) == 3:
        videos = videos.unsqueeze(-3)
    if len(videos.shape) == 4:
        videos = videos.unsqueeze(-3)
    if len(videos.shape) == 5:
        if videos.shape[1] == 1:
            videos = videos.repeat(1, 3, 1, 1, 1)
    else: 
        raise Exception("given shape of data cannot be handled")

    # Convert to numpy and move time axis for compatibility
    videos = rearrange(videos, 'b c t h w -> t c h (b w)').clamp(0,1).cpu().detach().numpy()
    videos = (videos * 255).astype(np.uint8)

    # Check if wandb is in the log_with list
    if any(logger.value == 'wandb' for logger in accelerator.log_with):
        import wandb
        wandb.log({name: wandb.Video(videos, caption=name, fps=fps)})
    else:
        print(f"WARNING: Wandb is not configured as a tracker. Videos will not be logged.")

class EvaluationTask(ABC):
    def __init__(self, name, metrics, input_key='data', save_as=['nii.gz'], delete_samples=False, **kwargs):
        self.kw_args = {k: v for k, v in locals().items() if k not in ['self', '__class__']}
       
        self.input_key = input_key
        self.writer = ToDisk(formats=save_as)
        self.name = name
        self.multi_meter = MultiMeter()
        self.delete_samples = delete_samples
        # Initialize instance variables
        self.output_dir = None
        self.normalize = None
        self.step = None 
        self.wandb_log = None
        self.accelerator = None
        self.reset_wandb_samples()

        self.composite_metrics = CompositeMetrics(metrics)

    def reset_wandb_samples(self):
        """Reset the wandb sample lists"""
        self.sample_wandb = []
        self.ref_wandb = []
        self.sample_ctr = 0

    def save_to_disk(self, path, data):
        """Save data to disk in a separate thread."""
        thread = threading.Thread(target=self.writer.to_disk, args=(path, data))
        thread.daemon = True  # ensures threads exit when main program exits
        thread.start()

    def save_to_wandb(self, vid_img, caption, step=None):
        """Save video and volume visualizations to wandb."""
        if not self.accelerator:
            print("No accelerator available, skipping wandb logging")
            return
        if not self.accelerator.is_main_process:
            return
        ## only log first 8 samples to not spam wandb
        log_videos(vid_img[0:8], caption, self.accelerator)
        log_volume_as_k3d_html(vid_img[0:8], caption, self.accelerator)

    @torch.no_grad()
    def _metrics_iter(self, reference, sample, metric_mask, subject=None):
        # Only collect samples if we're logging to wandb and haven't reached the limit
        if self.wandb_log and len(self.sample_wandb) < 8:
            self.sample_wandb.append(sample)
            self.ref_wandb.append(reference)
        metrics = self.compute_sample_metrics(reference, sample, metric_mask)
        return {key: value.cpu().numpy() for key, value in metrics.items()}
       

    @abstractmethod
    def process_batch(self, model, batch):
        pass

    @torch.no_grad()
    def compute_sample_metrics(self, reference, prediction, metric_mask):
        image_metrics = self.composite_metrics.compute_image_metrics(gt=reference, pred=prediction, mask=metric_mask)
        segmentation_metrics = self.composite_metrics.compute_segmentation_metrics(gt=reference, pred=prediction, mask=metric_mask)
        return {**image_metrics, **segmentation_metrics}

    @torch.no_grad()
    def compute_generation_metrics(self, fake_path, real_path=None):
        if self.output_dir is not None:
            print(f"fake {fake_path} and real {real_path}")
            return self.composite_metrics.compute_generation_metrics(fake_path=fake_path, real_path=real_path)
        return {}

    @torch.no_grad()
    def _normalize_output(self, *samples):
        """Normalizes the output samples based on the normalization setting."""
        if self.normalize:
            # Convert from [-1,1] to [0,1] range
            clamped_samples = [sample.clamp(-1, 1) for sample in samples]
            normalized_samples = [(sample / 2 + 0.5) for sample in clamped_samples]
            return normalized_samples
        else:
            # Just clamp to [0,1] range
            return [sample.clamp(0, 1) for sample in samples]

    @torch.no_grad()
    def _gather_data(self, *inputs):
        """
        Collects data (tensors, numbers, dicts, lists, strings, etc.) from all devices
        using Accelerator.gather_for_metrics, handling non-tensor types safely.
        """
        def safe_gather(v):
            """Recursively gather supported types."""
            if v is None:
                return None

            # Tensor or numeric types
            if isinstance(v, torch.Tensor):
                return self.accelerator.gather_for_metrics(v)
            if isinstance(v, (float, int)):
                return torch.tensor(self.accelerator.gather_for_metrics(v))

            # Strings or list of strings
            if isinstance(v, str):
                return self.accelerator.gather_for_metrics(v)
            if isinstance(v, (list, tuple)) and len(v) > 0 and isinstance(v[0], str):
                return self.accelerator.gather_for_metrics(v)

            # Dicts → recurse
            if isinstance(v, dict):
                return {k: safe_gather(val) for k, val in v.items()}

            # Lists or tuples → recurse
            if isinstance(v, (list, tuple)):
                return [safe_gather(val) for val in v]

            # Fallback for any other type (object, np array, etc.)
            try:
                gathered = self.accelerator.gather_for_metrics(v)
                # try to tensorize if possible
                return torch.tensor(gathered) if not isinstance(gathered, torch.Tensor) else gathered
            except Exception:
                return self.accelerator.gather_for_metrics(v)

        # Process each positional argument
        gathered_data = [safe_gather(x) for x in inputs]
        return gathered_data


    @torch.no_grad()
    def save_evaluation_outputs(self, subject, ref, spl, m_mask, file_name):
        """Method to save outputs to disk. This can be overridden by subclasses."""
        if self.output_dir is not None and self.is_main:
            for file_n, subj, ref, spl, m_mask in zip(file_name, subject, ref, spl, m_mask):
                ref, spl, m_mask = ref[None, ...], spl[None, ...], m_mask[None, ...]
                # Save the GIFs to disk
                self.save_to_disk(os.path.join(self.output_dir, self.name, "sample", f"{subj}", f"{file_n}"), spl)
                self.save_to_disk(os.path.join(self.output_dir, self.name, "reference", f"{subj}", f"{file_n}"), ref)
                

    def delete_evaluation_outputs(self):
        """Method to delete evaluation outputs from disk."""
        if self.output_dir is not None and self.is_main:
            shutil.rmtree(os.path.join(self.output_dir, self.name, "sample"))
            shutil.rmtree(os.path.join(self.output_dir, self.name, "reference"))
            print(f"Deleted evaluation outputs for {self.name}")


    def _compute_generation_metrics(self):
        if self.output_dir is not None:
            return self.compute_generation_metrics(os.path.join(self.output_dir, self.name, "sample"), os.path.join(self.output_dir, self.name, "reference"))
        return {}

    @torch.no_grad()
    def evaluate(self, model, dataloader, mode):
        """Evaluate the model on the given dataloader."""
        # Reset wandb samples at the start of evaluation
        self.reset_wandb_samples()
        # Reset MultiMeter for this evaluation
        self.multi_meter.reset()

        key_prefix = f"{mode}/ema_{self.name}" if self.is_ema else f"{mode}/model_{self.name}"
        
        for i, batch in enumerate(self.multi_meter.log_every(dataloader, len(dataloader), disable=not self.is_main)):
            if hasattr(self, 'n_samples') and self.n_samples is not None and self.sample_ctr >= self.n_samples:
                break
            subject = batch['sbj']
            file_name = batch['file_name']
            start_time = time.time()
            reference, sample, metric_mask = self.process_batch(model, batch)
            self.multi_meter.update("TPS", (time.time() - start_time) / len(batch))
            
            # Normalize outputs based on setting
            reference, sample = self._normalize_output(reference, sample)

            # Call the overridden method to save outputs
            self.save_evaluation_outputs(subject, reference, sample, metric_mask, file_name)
            # Compute metrics and add to MultiMeter
            metrics = self._metrics_iter(reference, sample, metric_mask, subject=subject)
            metrics, *_ = self._gather_data(metrics)
            # Add each metric to the multi_meter
            for metric_name, metric_value in metrics.items():
                if isinstance(metric_value, (np.ndarray, torch.Tensor)):
                    metric_value = metric_value.mean()
                self.multi_meter.update(metric_name, metric_value)

            self.sample_ctr += len(sample)

        # wait a bit so all data can be written to disk   
        time.sleep(5)
        gen_metrics = self._compute_generation_metrics()
        
        if self.delete_samples:
            self.delete_evaluation_outputs()
        
        # Get mean metrics from MultiMeter
        mean_metrics = self.multi_meter.get_averages()
        # Add generation metrics to mean_metrics
        mean_metrics.update(gen_metrics)

        if self.wandb_log and self.is_main and len(self.sample_wandb) > 0:
            # Stack reference and sample images side by side
            vid = torch.cat([
                torch.cat(self.ref_wandb, dim=0),  # Stack all reference images vertically
                torch.cat(self.sample_wandb, dim=0)  # Stack all sample images vertically
            ], dim=-2)  # Concatenate reference and samples horizontally
            self.save_to_wandb(vid, f"{key_prefix}")
            # Log mean metrics to wandb
            if self.accelerator:
                self.accelerator.log({f"{key_prefix}/{k}": v for k, v in mean_metrics.items()})
                
        return None, mean_metrics

    @property
    def is_main(self):
        """Check if this is the main process."""
        return self.accelerator.is_main_process if self.accelerator else True

    def unwrap_model(self, model):
        """Unwrap model if it's wrapped by Accelerator."""
        return self.accelerator.unwrap_model(model) if self.accelerator else model

    def __call__(self, model, dataloader, mode, output_dir=None, wandb_log=True, normalize=True, step=None, accelerator=None, is_ema=False):
        self.output_dir = os.path.join(str(output_dir), str(mode)) if output_dir is not None else None
        self.normalize = normalize
        self.is_ema = is_ema
        self.step = step 
        self.wandb_log = wandb_log
        self.accelerator = accelerator
        result = self.evaluate(model, dataloader, mode)
        return result




class CompositeEvaluator():
    def __init__(self, tasks: list[EvaluationTask]):
        self.tasks = tasks
        self.accelerator = None

    def __call__(self, model, dataloader, mode, output_dir=None, wandb_log=True, normalize=False, step=None, accelerator=None, is_ema=False):
        self.accelerator = accelerator
        if len(self.tasks) == 0:
            print("No tasks to evaluate")
            return None, None

        results_dfs = {}
        results_means = {}

        for task in self.tasks:
            task_name = task.name
            print(f"Running {task_name}: ")
            df, means = task(model, dataloader, mode, output_dir, wandb_log, normalize, step=step, accelerator=accelerator, is_ema=is_ema)
            results_dfs[task_name] = df
            results_means[task_name] = means

        return results_dfs, results_means

    def generate_plots_from_results(self, results_dfs, output_dir, properties=None, metrics=None, advanced=False):
        if not results_dfs:
            print("No result DataFrames to plot.")
            return
        
        all_dfs = []
        for task_name, df in results_dfs.items():
            df = df.copy()
            df["task_name"] = task_name
            all_dfs.append(df)

        combined_df = pd.concat(all_dfs, ignore_index=True)

        # Step 1: Drop irrelevant columns
        drop_cols = ['subject', 'name', 'metrics_cfg', 'save_as']
        combined_df = combined_df.drop(columns=[col for col in drop_cols if col in combined_df.columns], errors='ignore')

        # Step 2: Expand 'kwargs' into separate sweep columns
        if 'kwargs' in combined_df.columns:
            kwargs_expanded = combined_df['kwargs'].apply(pd.Series)
            combined_df = pd.concat([combined_df.drop(columns=['kwargs']), kwargs_expanded], axis=1)

        # Step 3: Auto-detect properties and metrics if not specified
        if properties is None:
            properties = [col for col in kwargs_expanded.columns]
        if metrics is None:
            metrics = [col for col in combined_df.columns if col not in properties + ['task_name']]

        print(f"🧩 Sweep parameters: {properties}")
        print(f"📊 Metrics to plot: {metrics}")

        # Step 4: Plot
        plotter = Plotter(combined_df, output_dir)
        plotter.create_all_plots(properties, metrics, advanced=advanced)


    @staticmethod
    def _generate_sweep_variants(param_dict):
        """Detect sweepable parameters and generate all variant combinations."""
        sweep_keys = []
        sweep_values = []

        def extract_sweep_params(d, prefix=''):
            for k, v in d.items():
                full_key = f"{prefix}.{k}" if prefix else k
                if isinstance(v, dict):
                    if "values" in v:
                        sweep_keys.append(full_key)
                        sweep_values.append(v["values"])
                    elif "min" in v and "max" in v:
                        num_samples = v.get("num_samples", 5)
                        sampled = [random.randint(v["min"], v["max"]) for _ in range(num_samples)]
                        sweep_keys.append(full_key)
                        sweep_values.append(sampled)
                    else:
                        extract_sweep_params(v, full_key)
        
        extract_sweep_params(param_dict)

        # Create parameter combinations
        from itertools import product
        variants = []
        for combination in product(*sweep_values):
            variant = copy.deepcopy(param_dict)
            for k, v in zip(sweep_keys, combination):
                variant[k] = v
            variants.append(variant)
        return variants

    @staticmethod
    def _get_evaluation_tasks(cfg):
        tasks = []
        try:
            for t in cfg:
                base_params = OmegaConf.to_container(t, resolve=True)
                variants = CompositeEvaluator._generate_sweep_variants(base_params)

                # If no variants were generated, keep original
                if not variants:
                    task_cls = instantiate_from_config(t)
                    tasks.append(task_cls)
                else:
                    # Find which parameters are part of the sweep
                    sweep_params = set()
                    def find_sweep_params(d, prefix=''):
                        for k, v in d.items():
                            full_key = f"{prefix}.{k}" if prefix else k
                            if isinstance(v, dict):
                                if "values" in v or ("min" in v and "max" in v):
                                    sweep_params.add(full_key)
                                else:
                                    find_sweep_params(v, full_key)
                    
                    find_sweep_params(base_params)

                    for i, variant in enumerate(variants):
                        # Create a new config with the variant parameters
                        t_variant = copy.deepcopy(t)
                        
                        # Track which sweep parameters were changed for naming
                        changed_params = []
                        
                        # Update the config with variant values
                        for k, v in variant.items():
                            if '.' in k:  # Handle nested parameters
                                parts = k.split('.')
                                current = t_variant
                                for part in parts[:-1]:
                                    if part not in current:
                                        current[part] = {}
                                    current = current[part]
                                current[parts[-1]] = v
                                # Only add sweep parameters to the name
                                if k in sweep_params and isinstance(v, (str, int, float)):
                                    # Remove the prefix for nested parameters
                                    param_name = k.split('.')[-1] if '.' in k else k
                                    changed_params.append(f"{param_name}={v}")
                            else:
                                t_variant[k] = v
                                # Only add sweep parameters to the name
                                if k in sweep_params and isinstance(v, (str, int, float)):
                                    changed_params.append(f"{k}={v}")
                        
                        # Create a descriptive name including only sweep parameters
                        if "name" in t_variant:
                            if len(variants) > 1:
                                param_str = "-".join(changed_params)
                                t_variant["name"] = f"{t_variant['name']}-{param_str}"
                        
                        task_cls = instantiate_from_config(t_variant)
                        tasks.append(task_cls)

        except Exception as e:
            print(f"Error loading tasks: {e}")
            print(traceback.format_exc())

        return tasks
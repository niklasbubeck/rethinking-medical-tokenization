from accelerate import Accelerator, DistributedType, DistributedDataParallelKwargs
import functools
import inspect
import gc
import torch
import numpy as np
from einops import rearrange
from contextlib import contextmanager
from accelerate.utils import MegatronLMPlugin

RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RESET = "\033[0m"  # Reset to default color


# megatron_plugin = MegatronLMPlugin(
#     tp_degree=1,  # Tensor Parallel degree
#     pp_degree=1,  # Pipeline Parallel degree (set to 1 if not using)
#     num_micro_batches=1,
#     sequence_parallelism=True,  # Enable Sequence Parallelism
#     # Add other Megatron-specific arguments as needed
# )

class AccelerateParent():
    _accelerator_instance = None # singleton shared along derived classes
    def __init__(self, accelerator_cfg, **kwargs):
        """
        Initialize an Accelerator singleton instance using the configuration
        provided by the user. If the singleton instance already exists, do
        nothing.

        Args:
            - config: Hydra configuration object
            - **kwargs: Additional keyword arguments to be passed to the
              Accelerator constructor
        """
        if AccelerateParent._accelerator_instance is None:
            AccelerateParent._accelerator_instance = Accelerator(**accelerator_cfg, **{
            'kwargs_handlers': [DistributedDataParallelKwargs(find_unused_parameters = True)],
            # 'megatron_lm_plugin': megatron_plugin,
            })
            print(f"Accelerate Configuration! \n", AccelerateParent._accelerator_instance.state)
    
    @property
    def accelerator(self):
        # Expose the accelerator instance to derived classes
        return AccelerateParent._accelerator_instance

    @property 
    def num_processes(self):
        return AccelerateParent._accelerator_instance.num_processes

    @property
    def device(self):
        """
        Return the device associated with the current process 
        """
        return AccelerateParent._accelerator_instance.device
    
    @property
    def rank(self):
        """
        Return the rank of the current process
        """
        return AccelerateParent._accelerator_instance.process_index

    @property
    def is_distributed(self):
        """
        Property function to determine if the accelerator is distributed (multi gpu).
        """
        return not (AccelerateParent._accelerator_instance.distributed_type == DistributedType.NO and AccelerateParent._accelerator_instance.num_processes == 1)

    @property
    def is_main(self):
        """
        A property that returns whether the current process is the main process.
        """
        return AccelerateParent._accelerator_instance.is_main_process

    @property
    def is_local_main(self):
        """
        A property method that returns whether the accelerator is the local main process of this node.
        """
        return AccelerateParent._accelerator_instance.is_local_main_process

    @property
    def disable_tqdm(self):
        """
        Only enable tqdm if the current process is the main process.
        """
        if self.is_main:
            return False 
        else:
            return True        

    def log_volume_as_k3d_html(self, vid_img, caption, *args, step=None, normalize=False, **kwargs):
        """
        Visualize a 5D tensor as a 3D volume and log to available trackers.
        
        Args:
        vid_img (torch.Tensor): 5D tensor of shape [B, C, X, Y, Z]
        caption (str): Caption for the visualization
        step (int, optional): Step or iteration number
        """
        if vid_img.dim() != 5:
            self.print("vid_img must be a 5D tensor to make a k3d volume upload", mode='warning')
            return 

        # Get rid of batch and channel dims
        vid_img = vid_img[0,0,...].clamp(0,1).cpu().detach().numpy()
        x, y, z = vid_img.shape
        # Check if wandb is in the log_with list
        if any(logger.value == 'wandb' for logger in self.accelerator.log_with):
            import wandb
            import k3d  # optional: only needed to log 3D volumes
            plot = k3d.plot()
            volume = k3d.volume(vid_img.astype(np.float32),
                                color_map=k3d.basic_color_maps.Binary,
                                bounds=[0, x, 0, y, 0, z])
            plot += volume
            html_snapshot = plot.get_snapshot()
            wandb.log({f"{caption}/3d_visualization": wandb.Html(html_snapshot)}, step=step)
        else:
            self.print(f"Wandb is not configured as a tracker. 3D visualization will not be logged. You initialized trackers with {self.accelerator.log_with}", mode='warning')

    def log_videos(self, videos, name, step=None, fps=4, normalize=False):
        """
        Log videos to all available trackers.
        
        Args:
        videos (torch.Tensor): 5D tensor of shape [B, C, T, H, W]
        name (str): Name of the video
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
        if any(logger.value == 'wandb' for logger in self.accelerator.log_with):
            import wandb
            wandb.log({name: wandb.Video(videos, caption=name, fps=fps)}, step=step)
        else:
            self.print(f"Wandb is not configured as a tracker. 3D visualization will not be logged. You initialized trackers with {self.accelerator.log_with}", mode='warning')

    def print(self, msg, *args, mode='default', **kwargs):
        """
        A function that prints a message if it is the main instance.
        
        Parameters:
            msg (str): The message to be printed.
            **kwargs: Additional keyword arguments to be passed to the print function.
        
        Returns:
            None
        """
        if not self.is_main:
            return
        
        if mode == 'warning':
            msg = f"{YELLOW}{msg}{RESET}"
        elif mode == 'error':
            msg = f"{RED}{msg}{RESET}"
        elif mode == 'success':
            msg = f"{GREEN}{msg}{RESET}"

        return self.accelerator.print(msg, *args, **kwargs)
    
    def prepare(self, *args):
        """
        Prepare any objects to work with multiple GPUs, handling dictionaries recursively.
        """
        for arg in args:
            if isinstance(arg, dict):
                for key, value in arg.items():
                    arg[key] = self.accelerator.prepare(value)
        
        return self.accelerator.prepare(*args)

    def unwrap_model(self, model):
        """
        A property that returns the unwrapped UNet model.
        """
        return self.accelerator.unwrap_model(model)

    def wait(self):
        """
        A method that waits for everyone to be ready.
        """
        return self.accelerator.wait_for_everyone()



def find_executable_batch_size(function: callable = None, starting_batch_size: int = 128, reduction_strategy: str = 'half'):
    """
    A decorator that tries to execute `function`. If it fails due to out-of-memory or CUDNN exceptions,
    the batch size is reduced according to the specified reduction strategy and passed to `function` again.

    Args:
        function (`callable`, *optional*): A function to wrap
        starting_batch_size (`int`, *optional*): The initial batch size to try
        reduction_strategy (`str`, *optional*): Strategy to reduce batch size. Options:
            - 'half': Halve the batch size
            - 'linear': Reduce by a fixed amount (default 32)
            - 'exponential': Reduce by a factor (default 0.8)

    Example:
    >>> @find_executable_batch_size(starting_batch_size=128, reduction_strategy='linear')
    ... def train(batch_size, model, optimizer):
    ...     ...
    >>> train(model, optimizer)
    """
    if function is None:
        return functools.partial(find_executable_batch_size, 
                                 starting_batch_size=starting_batch_size, 
                                 reduction_strategy=reduction_strategy)

    batch_size = starting_batch_size

    def reduce_batch_size(current_size):
        if reduction_strategy == 'half':
            return current_size // 2
        elif reduction_strategy == 'linear':
            return max(current_size - 8, 1)  # Reduce by 8, minimum 1
        elif reduction_strategy == 'exponential':
            return max(int(current_size * 0.8), 1)  # Reduce by 10%, minimum 1
        else:
            raise ValueError(f"Unknown reduction strategy: {reduction_strategy}")

    def decorator(*args, **kwargs):
        nonlocal batch_size
        gc.collect()
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
        
        params = list(inspect.signature(function).parameters.keys())
        if len(params) < (len(args) + 1):
            arg_str = ", ".join([f"{arg}={value}" for arg, value in zip(params[1:], args[1:])])
            raise TypeError(
                f"Batch size was passed into `{function.__name__}` as the first argument when called. "
                f"Remove this as the decorator already does so: `{function.__name__}({arg_str})`"
            )
        
        while True:
            if batch_size == 0:
                raise RuntimeError("No executable batch size found, reached zero.")
            try:
                return function(batch_size, *args, **kwargs)
            except Exception as e:
                if should_reduce_batch_size(e):
                    gc.collect()
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                    batch_size = reduce_batch_size(batch_size)
                else:
                    raise

    return decorator

def should_reduce_batch_size(exception: Exception) -> bool:
    """
    Checks if `exception` relates to CUDA out-of-memory, CUDNN not supported, or CPU out-of-memory

    Args:
        exception (`Exception`):
            An exception
    """
    _statements = [
        "CUDA out of memory.",  # CUDA OOM
        "cuDNN error: CUDNN_STATUS_NOT_SUPPORTED.",  # CUDNN SNAFU
        "DefaultCPUAllocator: can't allocate memory",  # CPU OOM
    ]
    if isinstance(exception, RuntimeError) and len(exception.args) == 1:
        return any(err in exception.args[0] for err in _statements)
    return False

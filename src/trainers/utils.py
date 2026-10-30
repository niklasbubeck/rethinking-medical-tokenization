from collections import deque, defaultdict
import time
import datetime
import numpy as np
import torch
from tqdm import tqdm
from omegaconf import DictConfig
from src.utils.general import instantiate_from_config
from torchvision import transforms
from src.utils import DummyProfiler

def get_torchvision_transforms(cfg):
    trfs = []
    for t in cfg:
        transform_cls = instantiate_from_config(t)
        trfs.append(transform_cls)
    trfs = transforms.Compose(trfs)

    return trfs

class ProfilerWrapper:
    def __init__(self, use_profiler, profiler_args):
        self.use_profiler = use_profiler
        if self.use_profiler:
            self.profiler = torch.profiler.profile(**profiler_args)
        else:
            self.profiler = DummyProfiler()
    
    def __enter__(self):
        if self.use_profiler:
            print("Profiler is starting...")
            return self.profiler.__enter__()
        else:
            return self.profiler  # DummyProfiler
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.use_profiler:
            print("Profiler is stopping...")
            return self.profiler.__exit__(exc_type, exc_val, exc_tb)
    
    def step(self):
        self.profiler.step()  # works for both real and dummy
    
    def stop(self):
        if self.use_profiler and self.profiler.running:
            print("Profiler is stopping...")
            self.profiler.stop()

def load_optimizers(optimizers, state):
    """Load optimizer states from checkpoint state dict."""
    if optimizers is None or 'optimizer' not in state:
        return
    
    optimizer_state = state['optimizer']
    if isinstance(optimizers, (dict, DictConfig)) and isinstance(optimizer_state, dict):
        for k, opt_state in optimizer_state.items():
            if k in optimizers:
                optimizers[k].load_state_dict(opt_state)
    elif optimizers is not None:
        optimizers.load_state_dict(optimizer_state)

def load_schedulers(schedulers, state):
    """Load scheduler states from checkpoint state dict."""
    if schedulers is None or 'scheduler' not in state:
        return
    
    scheduler_state = state['scheduler']
    if isinstance(schedulers, (dict, DictConfig)) and isinstance(scheduler_state, dict):
        for k, sched_state in scheduler_state.items():
            if k in schedulers:
                schedulers[k].load_state_dict(sched_state)
    elif schedulers is not None:
        schedulers.load_state_dict(scheduler_state)

def migrate_old_checkpoint(state_dict):
    """Migrate old checkpoint format to new unified format."""
    # This function can be used to convert old checkpoint formats
    # to the new unified trainer state_dict format
    migrated_state = {}
    
    # Copy all existing keys
    for key, value in state_dict.items():
        migrated_state[key] = value
    
    # Add any missing keys that the new format expects
    if 'epoch' not in migrated_state:
        migrated_state['epoch'] = 1
    if 'steps' not in migrated_state:
        migrated_state['steps'] = 0
    if 'train_days' not in migrated_state:
        migrated_state['train_days'] = 1
    if 'start_time' not in migrated_state:
        migrated_state['start_time'] = 0
    
    return migrated_state
    



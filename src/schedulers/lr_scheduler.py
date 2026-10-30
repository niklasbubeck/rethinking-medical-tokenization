import math
import warnings
from typing import List, Optional, Union

import numpy as np
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class WarmupCosineScheduler(_LRScheduler):
    """
    Implements a scheduler with warmup followed by cosine annealing.
    Inherits from PyTorch's _LRScheduler for proper integration.
    """
    def __init__(
        self,
        optimizer: Optimizer,
        warm_up_steps: int,
        max_decay_steps: int,
        min_lr: float,
        max_lr: Optional[float] = None,
        start_lr: float = 0.0,
        last_epoch: int = -1,
        verbose: bool = False
    ):
        self.warm_up_steps = warm_up_steps
        self.max_decay_steps = max_decay_steps
        self.min_lr = min_lr
        self.start_lr = start_lr
        
        # If max_lr is not specified, use the initial learning rate from optimizer
        if max_lr is None:
            self.max_lr = optimizer.param_groups[0]['lr']
        else:
            self.max_lr = max_lr
            
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                        "please use `get_last_lr()`.", UserWarning)

        step = self.last_epoch

        if step < self.warm_up_steps:
            # Linear warmup
            alpha = step / self.warm_up_steps
            factor = self.start_lr + (self.max_lr - self.start_lr) * alpha
        else:
            # Cosine decay
            t = (step - self.warm_up_steps) / (self.max_decay_steps - self.warm_up_steps)
            t = min(t, 1.0)
            factor = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + np.cos(t * np.pi))

        return [factor * base_lr / self.max_lr for base_lr in self.base_lrs]


class MultiCycleLinearScheduler(_LRScheduler):
    """
    Implements a multi-cycle scheduler with warmup and linear decay for each cycle.
    Supports different parameters for each cycle.
    """
    def __init__(
        self,
        optimizer: Optimizer,
        warm_up_steps: List[int],
        cycle_lengths: List[int],
        min_lr: List[float],
        max_lr: List[float],
        start_lr: List[float],
        mode: str = "step",
        num_processes: int = 1,
        epoch_to_steps: Optional[int] = None,
        last_epoch: int = -1, #is loaded from the state, but can be last_epoch or last_step in our case
        verbose: bool = False
    ):
        # Validate inputs
        assert len(warm_up_steps) == len(cycle_lengths) == len(min_lr) == len(max_lr) == len(start_lr), \
            "All parameter lists must have the same length"
        
        if mode == "epoch":
            assert epoch_to_steps is not None, "epoch_to_steps must be provided when mode='epoch'"
            self.warm_up_steps = [steps * epoch_to_steps for steps in warm_up_steps]
            self.cycle_lengths = [cycle * epoch_to_steps for cycle in cycle_lengths]
        else:
            self.warm_up_steps = [steps * num_processes for steps in warm_up_steps]
            self.cycle_lengths = [cycle * num_processes for cycle in cycle_lengths]

        self.min_lr = min_lr
        self.max_lr = max_lr
        self.start_lr = start_lr
        self.cum_cycles = np.cumsum([0] + list(self.cycle_lengths))
        
        super().__init__(optimizer, last_epoch)

    def _find_cycle(self, step: int) -> int:
        """Find which cycle the current step belongs to."""
        for i, end_step in enumerate(self.cum_cycles[1:], 0):
            if step <= end_step:
                return i
        return len(self.cycle_lengths) - 1

    def get_lr(self) -> List[float]:
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                        "please use `get_last_lr()`.", UserWarning)

        step = self.last_epoch
        cycle = self._find_cycle(step)
        cycle_step = step - self.cum_cycles[cycle]

        if cycle_step < self.warm_up_steps[cycle]:
            # Linear warmup within cycle
            alpha = cycle_step / self.warm_up_steps[cycle]
            factor = self.start_lr[cycle] + (self.max_lr[cycle] - self.start_lr[cycle]) * alpha
        else:
            # Linear decay within cycle
            remaining = self.cycle_lengths[cycle] - cycle_step
            total_decay = self.cycle_lengths[cycle] - self.warm_up_steps[cycle]
            alpha = remaining / total_decay
            factor = self.min_lr[cycle] + (self.max_lr[cycle] - self.min_lr[cycle]) * alpha

        return [factor * base_lr / self.max_lr[cycle] for base_lr in self.base_lrs]


class MultiCycleCosineScheduler(MultiCycleLinearScheduler):
    """
    Variant of MultiCycleLinearScheduler that uses cosine annealing instead of linear decay.
    """
    def get_lr(self) -> List[float]:
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                        "please use `get_last_lr()`.", UserWarning)

        step = self.last_epoch
        cycle = self._find_cycle(step)
        cycle_step = step - self.cum_cycles[cycle]

        if cycle_step < self.warm_up_steps[cycle]:
            # Linear warmup within cycle
            alpha = cycle_step / self.warm_up_steps[cycle]
            factor = self.start_lr[cycle] + (self.max_lr[cycle] - self.start_lr[cycle]) * alpha
        else:
            # Cosine decay within cycle
            t = (cycle_step - self.warm_up_steps[cycle]) / (self.cycle_lengths[cycle] - self.warm_up_steps[cycle])
            t = min(t, 1.0)
            factor = self.min_lr[cycle] + 0.5 * (self.max_lr[cycle] - self.min_lr[cycle]) * (1 + np.cos(t * np.pi))

        return [factor * base_lr / self.max_lr[cycle] for base_lr in self.base_lrs]


class LambdaCosineWithMinimumScheduler(_LRScheduler):
    """
    Implements a learning rate scheduler that decreases following the values of the
    cosine function between the initial lr and a minimum rate,
    after a warmup period during which it increases linearly.
    
    This matches the behavior of the cosine_with_minimum scheduler from the provided file.
    """
    def __init__(self, warm_up_steps, f_min, f_max, f_start, cycle_lengths, verbosity_interval=0, mode="step", num_processes=1, epoch_to_steps=None, optimizer=None, last_epoch=-1, verbose=False):
        assert len(warm_up_steps) == len(f_min) == len(f_max) == len(f_start) == len(cycle_lengths)
        if mode == "epoch":
            assert epoch_to_steps is not None, "If you want to use this for epochwise schedule you have to define the epoch_to_steps"
            self.lr_warm_up_steps = [steps * epoch_to_steps for steps in warm_up_steps]
            self.cycle_lengths = [cycle * epoch_to_steps for cycle in cycle_lengths]  
        else:
            self.lr_warm_up_steps = [steps * num_processes for steps in warm_up_steps]
            self.cycle_lengths = [cycle * num_processes for cycle in cycle_lengths]
        
        self.f_start = f_start
        self.f_min = f_min
        self.f_max = f_max
        self.cum_cycles = np.cumsum([0] + list(self.cycle_lengths))
        self.last_f = 0.
        self.verbosity_interval = verbosity_interval

        super().__init__(optimizer, last_epoch)
    def find_in_interval(self, n):
        interval = 0
        for cl in self.cum_cycles[1:]:
            if n <= cl:
                return interval
            interval += 1
        return len(self.cum_cycles) - 2  # Return the last interval if n exceeds all cycles

    def schedule(self, n, *args, **kwargs):
        cycle = self.find_in_interval(n)
        n = n - self.cum_cycles[cycle]
        
        if self.verbosity_interval > 0:
            if n % self.verbosity_interval == 0: 
                print(f"current step: {n}, recent lr-multiplier: {self.last_f}, current cycle {cycle}")
        
        if n < self.lr_warm_up_steps[cycle]:
            # Linear warmup phase
            f = (self.f_max[cycle] - self.f_start[cycle]) / self.lr_warm_up_steps[cycle] * n + self.f_start[cycle]
        else:
            # Cosine decay with minimum value
            progress = (n - self.lr_warm_up_steps[cycle]) / (self.cycle_lengths[cycle] - self.lr_warm_up_steps[cycle])
            progress = min(progress, 1.0)
            cos_term = 0.5 * (1.0 + math.cos(math.pi * progress))
            f = self.f_min[cycle] + (self.f_max[cycle] - self.f_min[cycle]) * cos_term
        
        self.last_f = f
        return f

    def __call__(self, n, **kwargs):
        return self.schedule(n, **kwargs)
    
    def get_lr(self) -> List[float]:
        factor = self.schedule(self.last_epoch)
        return [factor * base_lr for base_lr in self.base_lrs]

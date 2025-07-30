# Copyright (C) 2025 Human Dataware Lab.
# Modified from original work by HDL members
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
#
# Original work Copyright (C) 2025 Plachtaa <https://github.com/Plachtaa>
# Original source: https://github.com/Plachtaa/seed-vc

"""Optimizer module for managing multiple optimizers and learning rate schedulers."""

from functools import reduce
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer


class MultiOptimizer:
    """Multi-optimizer wrapper for training multiple models with different optimizers.

    This class manages multiple optimizers and their corresponding learning rate schedulers,
    allowing different components of a model to be optimized with different settings.

    Attributes:
        optimizers: Dictionary mapping optimizer names to optimizer instances.
        schedulers: Dictionary mapping scheduler names to scheduler instances.
        keys: List of optimizer keys.
        param_groups: Combined parameter groups from all optimizers.
    """

    def __init__(
        self,
        optimizers: Optional[Dict[str, Optimizer]] = None,
        schedulers: Optional[Dict[str, _LRScheduler]] = None,
    ) -> None:
        """Initialize MultiOptimizer.

        Args:
            optimizers: Dictionary of optimizer instances keyed by name.
            schedulers: Dictionary of scheduler instances keyed by name.
        """
        self.optimizers = optimizers if optimizers is not None else {}
        self.schedulers = schedulers if schedulers is not None else {}
        self.keys = list(optimizers.keys() if optimizers else [])
        self.param_groups = reduce(
            lambda x, y: x + y,
            [v.param_groups for v in self.optimizers.values()],
        )

    def state_dict(self) -> List[Tuple[str, Dict[str, Any]]]:
        """Get state dictionaries for all optimizers.

        Returns:
            List of (key, state_dict) tuples for each optimizer.
        """
        state_dicts = [(key, self.optimizers[key].state_dict()) for key in self.keys]
        return state_dicts

    def scheduler_state_dict(self) -> List[Tuple[str, Dict[str, Any]]]:
        """Get state dictionaries for all schedulers.

        Returns:
            List of (key, state_dict) tuples for each scheduler.
        """
        state_dicts = [(key, self.schedulers[key].state_dict()) for key in self.keys]
        return state_dicts

    def load_state_dict(self, state_dict: List[Tuple[str, Dict[str, Any]]]) -> None:
        """Load state dictionaries for all optimizers.

        Args:
            state_dict: List of (key, state_dict) tuples to load.
        """
        for key, val in state_dict:
            try:
                self.optimizers[key].load_state_dict(val)
            except Exception:
                print("Unloaded %s" % key)

    def load_scheduler_state_dict(self, state_dict: List[Tuple[str, Dict[str, Any]]]) -> None:
        """Load state dictionaries for all schedulers.

        Args:
            state_dict: List of (key, state_dict) tuples to load.
        """
        for key, val in state_dict:
            try:
                self.schedulers[key].load_state_dict(val)
            except Exception:
                print("Unloaded %s" % key)

    def step(
        self,
        key: Optional[str] = None,
        scaler: Optional[torch.cuda.amp.GradScaler] = None,
    ) -> None:
        """Perform optimization step for specified or all optimizers.

        Args:
            key: Specific optimizer key. If None, steps all optimizers.
            scaler: Optional gradient scaler for mixed precision training.
        """
        keys = [key] if key is not None else self.keys
        _ = [self._step(key, scaler) for key in keys]

    def _step(self, key: str, scaler: Optional[torch.cuda.amp.GradScaler] = None) -> None:
        """Internal method to perform single optimizer step.

        Args:
            key: Optimizer key.
            scaler: Optional gradient scaler for mixed precision training.
        """
        if scaler is not None:
            scaler.step(self.optimizers[key])
            scaler.update()
        else:
            self.optimizers[key].step()

    def zero_grad(self, key: Optional[str] = None) -> None:
        """Zero gradients for specified or all optimizers.

        Args:
            key: Specific optimizer key. If None, zeros all optimizers.
        """
        if key is not None:
            self.optimizers[key].zero_grad()
        else:
            _ = [self.optimizers[key].zero_grad() for key in self.keys]

    def scheduler(self, *args: Any, key: Optional[str] = None) -> None:
        """Step learning rate scheduler.

        Args:
            *args: Arguments to pass to scheduler.step().
            key: Specific scheduler key. If None, steps all schedulers.
        """
        if key is not None:
            self.schedulers[key].step(*args)
        else:
            _ = [self.schedulers[key].step_batch(*args) for key in self.keys]


def define_scheduler(optimizer: Optimizer, params: Dict[str, float]) -> _LRScheduler:
    """Create an exponential learning rate scheduler.

    Args:
        optimizer: The optimizer to schedule.
        params: Dictionary containing 'gamma' parameter for exponential decay.

    Returns:
        ExponentialLR scheduler instance.
    """
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=params["gamma"])

    return scheduler


def build_optimizer(
    model_dict: Dict[str, torch.nn.Module],
    lr: float,
    type: str = "AdamW",
) -> MultiOptimizer:
    """Build a multi-optimizer for multiple models.

    Creates optimizers and schedulers for each model in the dictionary.

    Args:
        model_dict: Dictionary mapping model names to model instances.
        lr: Learning rate for all optimizers.
        type: Optimizer type. Currently only supports "AdamW".

    Returns:
        MultiOptimizer instance managing all optimizers and schedulers.

    Raises:
        ValueError: If unknown optimizer type is specified.
    """
    optim = {}
    for key, model in model_dict.items():
        model_parameters = model.parameters()
        parameters_names = []
        parameters_names.append(
            [name_param_pair[0] for name_param_pair in model.named_parameters()],
        )
        if type == "AdamW":
            optim[key] = AdamW(
                model_parameters,
                lr=lr,
                betas=(0.9, 0.98),
                eps=1e-6,
                weight_decay=0.01,
            )
        else:
            raise ValueError("Unknown optimizer type: %s" % type)

    schedulers = dict(
        [
            (key, torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.999996))
            for key, opt in optim.items()
        ],
    )

    multi_optim = MultiOptimizer(optim, schedulers)
    return multi_optim


class MinLRExponentialLR(torch.optim.lr_scheduler.ExponentialLR):
    """Exponential learning rate scheduler with minimum learning rate.

    Extends PyTorch's ExponentialLR to enforce a minimum learning rate,
    preventing the learning rate from decaying below a specified threshold.

    Attributes:
        min_lr: Minimum learning rate threshold.
    """

    def __init__(self, optimizer: Optimizer, gamma: float, min_lr: float = 1e-5) -> None:
        """Initialize MinLRExponentialLR.

        Args:
            optimizer: The optimizer to schedule.
            gamma: Multiplicative factor of learning rate decay.
            min_lr: Minimum learning rate. Defaults to 1e-5.
        """
        self.min_lr = min_lr
        super().__init__(optimizer, gamma)

    def get_lr(self) -> List[float]:
        """Get current learning rates with minimum threshold applied.

        Returns:
            List of learning rates for each parameter group.
        """
        lrs = super().get_lr()
        return [max(lr, self.min_lr) for lr in lrs]


def build_single_optimizer(
    model: torch.nn.Module,
    lr: float,
) -> Tuple[Optimizer, _LRScheduler]:
    """Build a single optimizer and scheduler for a model.

    Creates an AdamW optimizer and MinLRExponentialLR scheduler for the model's
    trainable parameters.

    Args:
        model: The model to optimize.
        lr: Initial learning rate.

    Returns:
        Tuple of (optimizer, scheduler).
    """
    model_parameters = model.parameters()
    parameters_require_grad = filter(lambda p: p.requires_grad, model_parameters)
    optim = AdamW(
        parameters_require_grad,
        lr=lr,
        betas=(0.9, 0.98),
        eps=1e-6,
        weight_decay=0.01,
    )

    scheduler = MinLRExponentialLR(optim, gamma=0.999996, min_lr=1e-5)

    return optim, scheduler

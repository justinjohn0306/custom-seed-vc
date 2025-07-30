# Copyright (C) 2025 Human Dataware Lab.
# Modified from original work by HDL members
#
# Original Copyright (c) 2022 junjun3518 <https://github.com/junjun3518>
# Original source: <https://github.com/junjun3518/alias-free-torch>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""PyTorch implementation of alias-free activation functions."""

from typing import Callable

import torch
import torch.nn as nn

from .resample import DownSample1d, UpSample1d


class Activation1d(nn.Module):
    """Alias-free activation layer with upsampling and downsampling.

    This module applies an activation function with alias-free resampling
    to avoid aliasing artifacts in the frequency domain.
    """

    def __init__(
        self,
        activation: Callable[[torch.Tensor], torch.Tensor],
        up_ratio: int = 2,
        down_ratio: int = 2,
        up_kernel_size: int = 12,
        down_kernel_size: int = 12,
    ) -> None:
        """Initialize the Activation1d module.

        Args:
            activation: Activation function to apply (e.g., nn.ReLU()).
            up_ratio: Upsampling ratio. Defaults to 2.
            down_ratio: Downsampling ratio. Defaults to 2.
            up_kernel_size: Kernel size for upsampling. Defaults to 12.
            down_kernel_size: Kernel size for downsampling. Defaults to 12.
        """
        super().__init__()
        self.up_ratio = up_ratio
        self.down_ratio = down_ratio
        self.act = activation
        self.upsample = UpSample1d(up_ratio, up_kernel_size)
        self.downsample = DownSample1d(down_ratio, down_kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the activation layer.

        Args:
            x: Input tensor of shape [B, C, T] where B is batch size,
                C is number of channels, and T is time dimension.

        Returns:
            Output tensor of shape [B, C, T] after alias-free activation.
        """
        x = self.upsample(x)
        x = self.act(x)
        x = self.downsample(x)

        return x

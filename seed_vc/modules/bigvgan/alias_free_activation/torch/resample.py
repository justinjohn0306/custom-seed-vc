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

"""Resampling operations for alias-free signal processing."""

from typing import Optional

import torch
import torch.nn as nn
from torch.nn import functional as F

from .filter import LowPassFilter1d, kaiser_sinc_filter1d


class UpSample1d(nn.Module):
    """1D upsampling layer using Kaiser-windowed sinc filter.

    This module performs alias-free upsampling by using a low-pass
    filter to avoid aliasing artifacts.
    """

    def __init__(self, ratio: int = 2, kernel_size: Optional[int] = None) -> None:
        """Initialize the UpSample1d module.

        Args:
            ratio: Upsampling ratio. Defaults to 2.
            kernel_size: Size of the filter kernel. If None, automatically
                computed as int(6 * ratio // 2) * 2. Defaults to None.
        """
        super().__init__()
        self.ratio = ratio
        self.kernel_size = int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
        self.stride = ratio
        self.pad = self.kernel_size // ratio - 1
        self.pad_left = self.pad * self.stride + (self.kernel_size - self.stride) // 2
        self.pad_right = self.pad * self.stride + (self.kernel_size - self.stride + 1) // 2
        filter = kaiser_sinc_filter1d(
            cutoff=0.5 / ratio,
            half_width=0.6 / ratio,
            kernel_size=self.kernel_size,
        )
        self.register_buffer("filter", filter)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply upsampling to the input.

        Args:
            x: Input tensor of shape [B, C, T] where B is batch size,
                C is number of channels, and T is time dimension.

        Returns:
            Upsampled output tensor of shape [B, C, T * ratio].
        """
        _, C, _ = x.shape

        x = F.pad(x, (self.pad, self.pad), mode="replicate")
        x = self.ratio * F.conv_transpose1d(
            x,
            self.filter.expand(C, -1, -1),
            stride=self.stride,
            groups=C,
        )
        x = x[..., self.pad_left : -self.pad_right]

        return x


class DownSample1d(nn.Module):
    """1D downsampling layer using low-pass filtering.

    This module performs alias-free downsampling by applying a low-pass
    filter before downsampling to avoid aliasing artifacts.
    """

    def __init__(self, ratio: int = 2, kernel_size: Optional[int] = None) -> None:
        """Initialize the DownSample1d module.

        Args:
            ratio: Downsampling ratio. Defaults to 2.
            kernel_size: Size of the filter kernel. If None, automatically
                computed as int(6 * ratio // 2) * 2. Defaults to None.
        """
        super().__init__()
        self.ratio = ratio
        self.kernel_size = int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
        self.lowpass = LowPassFilter1d(
            cutoff=0.5 / ratio,
            half_width=0.6 / ratio,
            stride=ratio,
            kernel_size=self.kernel_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply downsampling to the input.

        Args:
            x: Input tensor of shape [B, C, T] where B is batch size,
                C is number of channels, and T is time dimension.

        Returns:
            Downsampled output tensor of shape [B, C, T // ratio].
        """
        xx = self.lowpass(x)

        return xx

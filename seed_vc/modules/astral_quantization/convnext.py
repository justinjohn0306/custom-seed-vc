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

"""ConvNeXt V2 architecture components for ASTRAL quantization."""

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNextV2LayerNorm(nn.Module):
    """LayerNorm that supports two data formats: channels_last or channels_first.

    The ordering of the dimensions in the inputs. channels_last corresponds to inputs
    with shape (batch_size, height, width, channels) while channels_first corresponds
    to inputs with shape (batch_size, channels, height, width).

    Attributes:
        weight: Learnable scale parameter.
        bias: Learnable shift parameter.
        eps: Small value to avoid division by zero.
        data_format: Either 'channels_last' or 'channels_first'.
        normalized_shape: Shape to normalize over.
    """

    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-6,
        data_format: str = "channels_last",
    ) -> None:
        """Initialize ConvNext LayerNorm.

        Args:
            normalized_shape: Size of normalized dimension.
            eps: Small value for numerical stability.
            data_format: Either 'channels_last' or 'channels_first'.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError(f"Unsupported data format: {self.data_format}")
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply layer normalization.

        Args:
            x: Input tensor.

        Returns:
            Normalized tensor.
        """
        if self.data_format == "channels_last":
            x = torch.nn.functional.layer_norm(
                x,
                self.normalized_shape,
                self.weight,
                self.bias,
                self.eps,
            )
        elif self.data_format == "channels_first":
            input_dtype = x.dtype
            x = x.float()
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = x.to(dtype=input_dtype)
            x = self.weight[None, :, None] * x + self.bias[None, :, None]
        return x


class GRN(nn.Module):
    """Global Response Normalization layer.

    Normalizes features globally and applies learnable scale and shift.

    Attributes:
        gamma: Learnable scale parameter.
        beta: Learnable shift parameter.
    """

    def __init__(self, dim: int) -> None:
        """Initialize GRN layer.

        Args:
            dim: Feature dimension.
        """
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply global response normalization.

        Args:
            x: Input tensor.

        Returns:
            Normalized tensor with residual connection.
        """
        Gx = torch.norm(x, p=2, dim=1, keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


class InterpolationLayer(nn.Module):
    """Simple interpolation layer for changing sequence length."""

    def __init__(self) -> None:
        """Initialize interpolation layer."""
        super().__init__()
        pass

    def forward(self, x: torch.Tensor, target_len: int, *args, **kwargs) -> torch.Tensor:
        """Interpolate tensor to target length.

        Args:
            x: Input tensor [B, C, T].
            target_len: Target sequence length.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Interpolated tensor.
        """
        x = F.interpolate(x, size=target_len, mode="linear")
        return x


class ConvNeXtV2Stage(nn.Module):
    """ConvNeXt V2 stage with multiple blocks and optional resampling.

    Supports downsampling, upsampling, and interpolation at specified layers.

    Attributes:
        blocks: List of ConvNeXt blocks.
        downsample_blocks: Downsampling layers.
        upsample_blocks: Upsampling layers.
        interpolation_blocks: Interpolation layers.
        input_projection: Input dimension projection.
        output_projection: Output dimension projection.
        gin: Optional global conditioning projection.
    """

    def __init__(
        self,
        dim: int = 512,
        intermediate_dim: int = 2048,
        num_blocks: int = 1,
        dilation: int = 1,
        downsample_layer_indices: Optional[List[int]] = None,
        downsample_factors: Optional[List[int]] = None,
        upsample_layer_indices: Optional[List[int]] = None,
        upsample_factors: Optional[List[int]] = None,
        interpolation_layer_indices: Optional[List[int]] = None,
        input_dim: Optional[int] = None,
        output_dim: Optional[int] = None,
        gin_channels: int = 0,
    ) -> None:
        """Initialize ConvNeXt V2 stage.

        Args:
            dim: Hidden dimension.
            intermediate_dim: Intermediate dimension in blocks.
            num_blocks: Number of ConvNeXt blocks.
            dilation: Dilation factor for convolutions.
            downsample_layer_indices: Indices where to downsample.
            downsample_factors: Downsampling factors.
            upsample_layer_indices: Indices where to upsample.
            upsample_factors: Upsampling factors.
            interpolation_layer_indices: Indices where to interpolate.
            input_dim: Input dimension (if different from dim).
            output_dim: Output dimension (if different from dim).
            gin_channels: Global conditioning channels.
        """
        super().__init__()
        # maybe downsample layers
        if downsample_layer_indices is not None:
            assert downsample_factors is not None
            self.downsample_blocks = nn.ModuleList(
                [
                    nn.Sequential(
                        ConvNextV2LayerNorm(dim, data_format="channels_first"),
                        nn.Conv1d(
                            dim,
                            dim,
                            kernel_size=downsample_factor,
                            stride=downsample_factor,
                        ),
                    )
                    for _, downsample_factor in zip(
                        downsample_layer_indices,
                        downsample_factors,
                        strict=False,
                    )
                ],
            )
            self.downsample_layer_indices = downsample_layer_indices
        else:
            self.downsample_blocks = nn.ModuleList()
            self.downsample_layer_indices = []

        # maybe upsample layers
        if upsample_layer_indices is not None:
            assert upsample_factors is not None
            self.upsample_blocks = nn.ModuleList(
                [
                    nn.Sequential(
                        ConvNextV2LayerNorm(dim, data_format="channels_first"),
                        nn.ConvTranspose1d(
                            dim,
                            dim,
                            kernel_size=upsample_factor,
                            stride=upsample_factor,
                        ),
                    )
                    for _, upsample_factor in zip(
                        upsample_layer_indices,
                        upsample_factors,
                        strict=False,
                    )
                ],
            )
            self.upsample_layer_indices = upsample_layer_indices
        else:
            self.upsample_blocks = nn.ModuleList()
            self.upsample_layer_indices = []

        # maybe interpolation layers
        if interpolation_layer_indices is not None:
            self.interpolation_blocks = nn.ModuleList(
                [InterpolationLayer() for _ in interpolation_layer_indices],
            )
            self.interpolation_layer_indices = interpolation_layer_indices
        else:
            self.interpolation_blocks = nn.ModuleList()
            self.interpolation_layer_indices = []

        # main blocks
        self.blocks = nn.ModuleList(
            [
                ConvNeXtV2Block(
                    dim=dim,
                    intermediate_dim=intermediate_dim,
                    dilation=dilation,
                )
                for _ in range(num_blocks)
            ],
        )
        # maybe input and output projections
        if input_dim is not None and input_dim != dim:
            self.input_projection = nn.Conv1d(input_dim, dim, kernel_size=1)
        else:
            self.input_projection = nn.Identity()
        if output_dim is not None and output_dim != dim:
            self.output_projection = nn.Conv1d(dim, output_dim, kernel_size=1)
        else:
            self.output_projection = nn.Identity()

        if gin_channels > 0:
            self.gin = nn.Conv1d(gin_channels, dim, kernel_size=1)

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Forward pass through ConvNeXt stage.

        Args:
            x: Input tensor [B, C, T].
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments including 'g' for conditioning
                     and 'target_len' for interpolation.

        Returns:
            Output tensor.
        """
        x = self.input_projection(x)  # B, D, T
        if hasattr(self, "gin"):
            g = kwargs["g"]
            x = x + self.gin(g)
        # pad to a multiple of cumprod(downsample_factors)
        if len(self.downsample_blocks) > 0:
            downsample_factor = 1
            for factor in self.downsample_blocks:
                downsample_factor *= factor[1].stride[0]
            pad_len = downsample_factor - x.size(-1) % downsample_factor
            if pad_len > 0:
                x = torch.cat([x, torch.zeros_like(x[:, :, :pad_len])], dim=-1)

        # main blocks
        for layer_idx, block in enumerate(self.blocks):
            if layer_idx in self.downsample_layer_indices:
                x = self.downsample_blocks[self.downsample_layer_indices.index(layer_idx)](x)
            if layer_idx in self.upsample_layer_indices:
                x = self.upsample_blocks[self.upsample_layer_indices.index(layer_idx)](x)
            if layer_idx in self.interpolation_layer_indices:
                x = self.interpolation_blocks[self.interpolation_layer_indices.index(layer_idx)](
                    x,
                    target_len=kwargs["target_len"],
                )
            x = block(x)
        x = self.output_projection(x)
        return x

    def setup_caches(self, *args, **kwargs) -> None:
        """Setup caches (placeholder for compatibility).

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.
        """
        pass


class ConvNeXtV2Block(nn.Module):
    """ConvNeXt V2 block with depthwise convolution and GRN.

    Implements: Conv -> Norm -> Linear -> GELU -> GRN -> Linear with residual.

    Attributes:
        dwconv: Depthwise convolution layer.
        norm: Layer normalization.
        pwconv1: First pointwise convolution (linear).
        act: Activation function.
        grn: Global response normalization.
        pwconv2: Second pointwise convolution (linear).
    """

    def __init__(
        self,
        dim: int,
        intermediate_dim: int,
        dilation: int = 1,
    ) -> None:
        """Initialize ConvNeXt V2 block.

        Args:
            dim: Hidden dimension.
            intermediate_dim: Intermediate dimension.
            dilation: Dilation factor for depthwise conv.
        """
        super().__init__()
        padding = (dilation * (7 - 1)) // 2
        self.dwconv = nn.Conv1d(
            dim,
            dim,
            kernel_size=7,
            padding=padding,
            groups=dim,
            dilation=dilation,
        )  # depthwise conv
        self.norm = ConvNextV2LayerNorm(dim, data_format="channels_first")
        self.pwconv1 = nn.Linear(
            dim,
            intermediate_dim,
        )  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.grn = GRN(intermediate_dim)
        self.pwconv2 = nn.Linear(intermediate_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through ConvNeXt block.

        Args:
            x: Input tensor [B, C, T].

        Returns:
            Output tensor with residual connection.
        """
        residual = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = x.transpose(1, 2)  # b d n -> b n d
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.transpose(1, 2)  # b n d -> b d n
        return residual + x

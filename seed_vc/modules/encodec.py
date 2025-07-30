# Copyright (C) 2025 Human Dataware Lab.
# Modified from original work by HDL members
#
# Copyright (C) 2024 Plachtaa <https://github.com/Plachtaa>
# Modified from original work by Plachtaa
#
# Original Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Convolutional layers wrappers and utilities."""

import math
import typing as tp
import warnings

import einops
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import spectral_norm, weight_norm


class ConvLayerNorm(nn.LayerNorm):
    """Convotution-friendly LayerNorm.

    Convolution-friendly LayerNorm that moves channels to last dimensions
    before running the normalization and moves them back to original position right after.
    """

    def __init__(self, normalized_shape: tp.Union[int, tp.List[int], torch.Size], **kwargs):
        """Initialize ConvLayerNorm.

        Args:
            normalized_shape: The normalized shape for layer normalization.
            **kwargs: Additional keyword arguments for LayerNorm.
        """
        super().__init__(normalized_shape, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply layer normalization to convolution output.

        Args:
            x: Input tensor with channels not in last dimension.

        Returns:
            Normalized tensor with channels back in original position.
        """
        x = einops.rearrange(x, "b ... t -> b t ...")
        x = super().forward(x)
        x = einops.rearrange(x, "b t ... -> b ... t")
        return x


CONV_NORMALIZATIONS = frozenset(
    [
        "none",
        "weight_norm",
        "spectral_norm",
        "time_layer_norm",
        "layer_norm",
        "time_group_norm",
    ],
)


def apply_parametrization_norm(module: nn.Module, norm: str = "none") -> nn.Module:
    """Apply parametrization normalization to a module.

    Args:
        module: Module to apply normalization to.
        norm: Type of normalization ('none', 'weight_norm', 'spectral_norm').

    Returns:
        Module with normalization applied.
    """
    assert norm in CONV_NORMALIZATIONS
    if norm == "weight_norm":
        return weight_norm(module)
    elif norm == "spectral_norm":
        return spectral_norm(module)
    else:
        # We already check was in CONV_NORMALIZATION, so any other choice
        # doesn't need reparametrization.
        return module


def get_norm_module(
    module: nn.Module,
    causal: bool = False,
    norm: str = "none",
    **norm_kwargs,
) -> nn.Module:
    """Return the proper normalization module.

    If causal is True, this will ensure the returned module is causal,
    or return an error if the normalization doesn't support causal evaluation.

    Args:
        module: Module to get normalization for.
        causal: Whether to ensure causal normalization.
        norm: Type of normalization.
        **norm_kwargs: Additional normalization arguments.

    Returns:
        Appropriate normalization module.
    """
    assert norm in CONV_NORMALIZATIONS
    if norm == "layer_norm":
        assert isinstance(module, nn.modules.conv._ConvNd)
        return ConvLayerNorm(module.out_channels, **norm_kwargs)
    elif norm == "time_group_norm":
        if causal:
            raise ValueError("GroupNorm doesn't support causal evaluation.")
        assert isinstance(module, nn.modules.conv._ConvNd)
        return nn.GroupNorm(1, module.out_channels, **norm_kwargs)
    else:
        return nn.Identity()


def get_extra_padding_for_conv1d(
    x: torch.Tensor,
    kernel_size: int,
    stride: int,
    padding_total: int = 0,
) -> int:
    """Calculate extra padding needed for conv1d to ensure full windows.

    Args:
        x: Input tensor to calculate padding for.
        kernel_size: Size of the convolution kernel.
        stride: Stride of the convolution.
        padding_total: Total padding already applied.

    Returns:
        Amount of extra padding needed.
    """
    length = x.shape[-1]
    n_frames = (length - kernel_size + padding_total) / stride + 1
    ideal_length = (math.ceil(n_frames) - 1) * stride + (kernel_size - padding_total)
    return ideal_length - length


def pad_for_conv1d(
    x: torch.Tensor,
    kernel_size: int,
    stride: int,
    padding_total: int = 0,
) -> torch.Tensor:
    """Pad for a convolution to make sure that the last window is full.

    Extra padding is added at the end. This is required to ensure that we can rebuild
    an output of the same length, as otherwise, even with padding, some time steps
    might get removed.

    For instance, with total padding = 4, kernel size = 4, stride = 2:
        0 0 1 2 3 4 5 0 0   # (0s are padding)
        1   2   3           # (output frames of a convolution, last 0 is never used)
        0 0 1 2 3 4 5 0     # (output of tr. conv., but pos. 5 is going to get removed as padding)
            1 2 3 4         # once you removed padding, we are missing one time step !

    Args:
        x: Input tensor.
        kernel_size: Convolution kernel size.
        stride: Convolution stride.
        padding_total: Total padding amount.

    Returns:
        Padded tensor.
    """
    extra_padding = get_extra_padding_for_conv1d(x, kernel_size, stride, padding_total)
    return F.pad(x, (0, extra_padding))


def pad1d(
    x: torch.Tensor,
    paddings: tp.Tuple[int, int],
    mode: str = "zero",
    value: float = 0.0,
) -> torch.Tensor:
    """Tiny wrapper around F.pad, just to allow for reflect padding on small input.

    If this is the case, we insert extra 0 padding to the right before the reflection happen.

    Args:
        x: Input tensor to pad.
        paddings: Tuple of (left_pad, right_pad).
        mode: Padding mode.
        value: Padding value for constant padding.

    Returns:
        Padded tensor.
    """
    length = x.shape[-1]
    padding_left, padding_right = paddings
    assert padding_left >= 0, padding_left
    assert padding_right >= 0, padding_right
    if mode == "reflect":
        max_pad = max(padding_left, padding_right)
        extra_pad = 0
        if length <= max_pad:
            extra_pad = max_pad - length + 1
            x = F.pad(x, (0, extra_pad))
        padded = F.pad(x, paddings, mode, value)
        end = padded.shape[-1] - extra_pad
        return padded[..., :end]
    else:
        return F.pad(x, paddings, mode, value)


def unpad1d(x: torch.Tensor, paddings: tp.Tuple[int, int]) -> torch.Tensor:
    """Remove padding from x, handling properly zero padding. Only for 1d!

    Args:
        x: Padded tensor.
        paddings: Tuple of (left_pad, right_pad) to remove.

    Returns:
        Unpadded tensor.
    """
    padding_left, padding_right = paddings
    assert padding_left >= 0, padding_left
    assert padding_right >= 0, padding_right
    assert (padding_left + padding_right) <= x.shape[-1]
    end = x.shape[-1] - padding_right
    return x[..., padding_left:end]


class NormConv1d(nn.Module):
    """Normalized 1D convolution layer.

    Wrapper around Conv1d and normalization applied to this conv to provide a uniform
    interface across normalization approaches.
    """

    def __init__(
        self,
        *args,
        causal: bool = False,
        norm: str = "none",
        norm_kwargs: tp.Dict[str, tp.Any] = None,
        **kwargs,
    ):
        """Initialize NormConv1d.

        Args:
            *args: Positional arguments for Conv1d.
            causal: Whether to use causal convolution.
            norm: Type of normalization to apply.
            norm_kwargs: Additional arguments for normalization layer.
            **kwargs: Additional keyword arguments for Conv1d.
        """
        super().__init__()
        if norm_kwargs is None:
            norm_kwargs = {}
        self.conv = apply_parametrization_norm(nn.Conv1d(*args, **kwargs), norm)
        self.norm = get_norm_module(self.conv, causal, norm, **norm_kwargs)
        self.norm_type = norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through normalized convolution.

        Args:
            x: Input tensor.

        Returns:
            Output tensor after convolution and normalization.
        """
        x = self.conv(x)
        x = self.norm(x)
        return x


class NormConv2d(nn.Module):
    """Normalized 2D convolution layer.

    Wrapper around Conv2d and normalization applied to this conv to provide a uniform
    interface across normalization approaches.
    """

    def __init__(
        self,
        *args,
        norm: str = "none",
        norm_kwargs: tp.Dict[str, tp.Any] = None,
        **kwargs,
    ):
        """Initialize NormConv2d.

        Args:
            *args: Positional arguments for Conv2d.
            norm: Type of normalization to apply.
            norm_kwargs: Additional arguments for normalization layer.
            **kwargs: Additional keyword arguments for Conv2d.
        """
        super().__init__()
        if norm_kwargs is None:
            norm_kwargs = {}
        self.conv = apply_parametrization_norm(nn.Conv2d(*args, **kwargs), norm)
        self.norm = get_norm_module(self.conv, causal=False, norm=norm, **norm_kwargs)
        self.norm_type = norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through normalized 2D convolution.

        Args:
            x: Input tensor.

        Returns:
            Output tensor after convolution and normalization.
        """
        x = self.conv(x)
        x = self.norm(x)
        return x


class NormConvTranspose1d(nn.Module):
    """Normalized 1D transposed convolution layer.

    Wrapper around ConvTranspose1d and normalization applied to this conv to provide a uniform
    interface across normalization approaches.
    """

    def __init__(
        self,
        *args,
        causal: bool = False,
        norm: str = "none",
        norm_kwargs: tp.Dict[str, tp.Any] = None,
        **kwargs,
    ):
        """Initialize NormConvTranspose1d.

        Args:
            *args: Positional arguments for ConvTranspose1d.
            causal: Whether to use causal convolution.
            norm: Type of normalization to apply.
            norm_kwargs: Additional arguments for normalization layer.
            **kwargs: Additional keyword arguments for ConvTranspose1d.
        """
        super().__init__()
        if norm_kwargs is None:
            norm_kwargs = {}
        self.convtr = apply_parametrization_norm(nn.ConvTranspose1d(*args, **kwargs), norm)
        self.norm = get_norm_module(self.convtr, causal, norm, **norm_kwargs)
        self.norm_type = norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through normalized transposed convolution.

        Args:
            x: Input tensor.

        Returns:
            Output tensor after transposed convolution and normalization.
        """
        x = self.convtr(x)
        x = self.norm(x)
        return x


class NormConvTranspose2d(nn.Module):
    """Normalized 2D transposed convolution layer.

    Wrapper around ConvTranspose2d and normalization applied to this conv to provide a uniform
    interface across normalization approaches.
    """

    def __init__(
        self,
        *args,
        norm: str = "none",
        norm_kwargs: tp.Dict[str, tp.Any] = None,
        **kwargs,
    ):
        """Initialize NormConvTranspose2d.

        Args:
            *args: Positional arguments for ConvTranspose2d.
            norm: Type of normalization to apply.
            norm_kwargs: Additional arguments for normalization layer.
            **kwargs: Additional keyword arguments for ConvTranspose2d.
        """
        super().__init__()
        if norm_kwargs is None:
            norm_kwargs = {}
        self.convtr = apply_parametrization_norm(nn.ConvTranspose2d(*args, **kwargs), norm)
        self.norm = get_norm_module(self.convtr, causal=False, norm=norm, **norm_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through normalized 2D transposed convolution.

        Args:
            x: Input tensor.

        Returns:
            Output tensor after transposed convolution and normalization.
        """
        x = self.convtr(x)
        x = self.norm(x)
        return x


class SConv1d(nn.Module):
    """Conv1d with some builtin handling of asymmetric or causal padding and normalization."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        causal: bool = False,
        norm: str = "none",
        norm_kwargs: tp.Dict[str, tp.Any] = None,
        pad_mode: str = "reflect",
        **kwargs,
    ):
        """Initialize SConv1d.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Size of the convolving kernel.
            stride: Stride of the convolution.
            dilation: Spacing between kernel elements.
            groups: Number of blocked connections from input to output.
            bias: Whether to add a learnable bias.
            causal: Whether to use causal convolution.
            norm: Type of normalization to apply.
            norm_kwargs: Additional arguments for normalization layer.
            pad_mode: Type of padding mode to use.
            **kwargs: Additional keyword arguments.
        """
        super().__init__()
        if norm_kwargs is None:
            norm_kwargs = {}
        # warn user on unusual setup between dilation and stride
        if stride > 1 and dilation > 1:
            warnings.warn(
                "SConv1d has been initialized with stride > 1 and dilation > 1"
                f" (kernel_size={kernel_size} stride={stride}, dilation={dilation}).",
                stacklevel=2,
            )
        self.conv = NormConv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            dilation=dilation,
            groups=groups,
            bias=bias,
            causal=causal,
            norm=norm,
            norm_kwargs=norm_kwargs,
        )
        self.causal = causal
        self.pad_mode = pad_mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with smart padding handling.

        Args:
            x: Input tensor of shape (B, C, T).

        Returns:
            Output tensor after convolution with appropriate padding.
        """
        B, C, T = x.shape
        kernel_size = self.conv.conv.kernel_size[0]
        stride = self.conv.conv.stride[0]
        dilation = self.conv.conv.dilation[0]
        kernel_size = (kernel_size - 1) * dilation + 1  # effective kernel size with dilations
        padding_total = kernel_size - stride
        extra_padding = get_extra_padding_for_conv1d(x, kernel_size, stride, padding_total)
        if self.causal:
            # Left padding for causal
            x = pad1d(x, (padding_total, extra_padding), mode=self.pad_mode)
        else:
            # Asymmetric padding required for odd strides
            padding_right = padding_total // 2
            padding_left = padding_total - padding_right
            x = pad1d(x, (padding_left, padding_right + extra_padding), mode=self.pad_mode)
        return self.conv(x)


class SConvTranspose1d(nn.Module):
    """ConvTranspose1d with some builtin handling of asymmetric or causal padding and norm."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        causal: bool = False,
        norm: str = "none",
        trim_right_ratio: float = 1.0,
        norm_kwargs: tp.Dict[str, tp.Any] = None,
        **kwargs,
    ):
        """Initialize SConvTranspose1d.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Size of the convolving kernel.
            stride: Stride of the convolution.
            causal: Whether to use causal convolution.
            norm: Type of normalization to apply.
            trim_right_ratio: Ratio for trimming the right padding.
            norm_kwargs: Additional arguments for normalization layer.
            **kwargs: Additional keyword arguments.
        """
        super().__init__()
        if norm_kwargs is None:
            norm_kwargs = {}
        self.convtr = NormConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            causal=causal,
            norm=norm,
            norm_kwargs=norm_kwargs,
        )
        self.causal = causal
        self.trim_right_ratio = trim_right_ratio
        assert self.causal or self.trim_right_ratio == 1.0, (
            "`trim_right_ratio` != 1.0 only makes sense for causal convolutions"
        )
        assert self.trim_right_ratio >= 0.0
        assert self.trim_right_ratio <= 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with smart padding trimming.

        Args:
            x: Input tensor.

        Returns:
            Output tensor after transposed convolution with appropriate padding removed.
        """
        kernel_size = self.convtr.convtr.kernel_size[0]
        stride = self.convtr.convtr.stride[0]
        padding_total = kernel_size - stride

        y = self.convtr(x)

        # We will only trim fixed padding. Extra padding from `pad_for_conv1d` would be
        # removed at the very end, when keeping only the right length for the output,
        # as removing it here would require also passing the length at the matching layer
        # in the encoder.
        if self.causal:
            # Trim the padding on the right according to the specified ratio
            # if trim_right_ratio = 1.0, trim everything from right
            padding_right = math.ceil(padding_total * self.trim_right_ratio)
            padding_left = padding_total - padding_right
            y = unpad1d(y, (padding_left, padding_right))
        else:
            # Asymmetric padding required for odd strides
            padding_right = padding_total // 2
            padding_left = padding_total - padding_right
            y = unpad1d(y, (padding_left, padding_right))
        return y


class SLSTM(nn.Module):
    """LSTM without worrying about the hidden state, nor the layout of the data.

    Expects input as convolutional layout and handles permutation internally.
    """

    def __init__(self, dimension: int, num_layers: int = 2, skip: bool = True) -> None:
        """Initialize SLSTM.

        Args:
            dimension: Hidden dimension size.
            num_layers: Number of LSTM layers.
            skip: Whether to use skip connections.
        """
        super().__init__()
        self.skip = skip
        self.lstm = nn.LSTM(dimension, dimension, num_layers)
        self.hidden = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through LSTM.

        Args:
            x: Input tensor with convolutional layout (B, C, T).

        Returns:
            Output tensor with same layout as input.
        """
        x = x.permute(2, 0, 1)
        if self.training:
            y, _ = self.lstm(x)
        else:
            y, self.hidden = self.lstm(x, self.hidden)
        if self.skip:
            y = y + x
        y = y.permute(1, 2, 0)
        return y

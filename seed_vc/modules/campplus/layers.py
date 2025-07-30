# Copyright 2025 Human Dataware Lab.
# Modified from original work by HDL members
#
# Copyright 2024 Plachtaa <https://github.com/Plachtaa>
# Modified from original work by Plachtaa
#
# Original Copyright 2023 modelscope <https://github.com/modelscope>
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

"""Neural network layers for CAMPPlus speaker embedding model."""

from typing import Optional

import torch
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from torch import nn


def get_nonlinear(config_str: str, channels: int) -> nn.Sequential:
    """Create nonlinear activation sequence from config string.

    Args:
        config_str: Configuration string with modules separated by '-'.
            Supported: 'relu', 'prelu', 'batchnorm', 'batchnorm_'.
        channels: Number of channels for normalization layers.

    Returns:
        Sequential module with specified nonlinearities.
    """
    nonlinear = nn.Sequential()
    for name in config_str.split("-"):
        if name == "relu":
            nonlinear.add_module("relu", nn.ReLU(inplace=True))
        elif name == "prelu":
            nonlinear.add_module("prelu", nn.PReLU(channels))
        elif name == "batchnorm":
            nonlinear.add_module("batchnorm", nn.BatchNorm1d(channels))
        elif name == "batchnorm_":
            nonlinear.add_module("batchnorm", nn.BatchNorm1d(channels, affine=False))
        else:
            raise ValueError("Unexpected module ({}).".format(name))
    return nonlinear


def statistics_pooling(
    x: torch.Tensor,
    dim: int = -1,
    keepdim: bool = False,
    unbiased: bool = True,
    eps: float = 1e-2,
) -> torch.Tensor:
    """Compute mean and standard deviation pooling.

    Args:
        x: Input tensor.
        dim: Dimension to pool over. Default is -1.
        keepdim: Whether to keep the pooled dimension. Default is False.
        unbiased: Whether to use unbiased std estimation. Default is True.
        eps: Small epsilon for numerical stability. Default is 1e-2.

    Returns:
        Concatenated mean and std statistics.
    """
    mean = x.mean(dim=dim)
    std = x.std(dim=dim, unbiased=unbiased)
    stats = torch.cat([mean, std], dim=-1)
    if keepdim:
        stats = stats.unsqueeze(dim=dim)
    return stats


def masked_statistics_pooling(
    x: torch.Tensor,
    x_lens: torch.Tensor,
    dim: int = -1,
    keepdim: bool = False,
    unbiased: bool = True,
    eps: float = 1e-2,
) -> torch.Tensor:
    """Compute masked mean and standard deviation pooling.

    Args:
        x: Input tensor of shape (batch, channels, time).
        x_lens: Valid lengths for each sequence in batch.
        dim: Dimension to pool over. Default is -1.
        keepdim: Whether to keep the pooled dimension. Default is False.
        unbiased: Whether to use unbiased std estimation. Default is True.
        eps: Small epsilon for numerical stability. Default is 1e-2.

    Returns:
        Concatenated mean and std statistics.
    """
    stats = []
    for i, x_len in enumerate(x_lens):
        x_i = x[i, :, :x_len]
        mean = x_i.mean(dim=dim)
        std = x_i.std(dim=dim, unbiased=unbiased)
        stats.append(torch.cat([mean, std], dim=-1))
    stats = torch.stack(stats, dim=0)
    if keepdim:
        stats = stats.unsqueeze(dim=dim)
    return stats


class StatsPool(nn.Module):
    """Statistics pooling layer for mean and std computation."""

    def __init__(self) -> None:
        """Initialize StatsPool."""
        super().__init__()

    def forward(self, x: torch.Tensor, x_lens: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass for statistics pooling.

        Args:
            x: Input tensor.
            x_lens: Optional sequence lengths for masking.

        Returns:
            Pooled statistics tensor.
        """
        if x_lens is not None:
            return masked_statistics_pooling(x, x_lens)
        return statistics_pooling(x)


class TDNNLayer(nn.Module):
    """Time Delay Neural Network layer."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        bias=False,
        config_str="batchnorm-relu",
    ):
        """Initialize TDNNLayer.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Size of the convolutional kernel.
            stride: Stride for the convolution.
            padding: Padding for the convolution.
            dilation: Dilation rate for the convolution.
            bias: Whether to use bias in convolution.
            config_str: Configuration string for nonlinearity layers.
        """
        super(TDNNLayer, self).__init__()
        if padding < 0:
            assert kernel_size % 2 == 1, (
                "Expect equal paddings, but got even kernel size ({})".format(kernel_size)
            )
            padding = (kernel_size - 1) // 2 * dilation
        self.linear = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )
        self.nonlinear = get_nonlinear(config_str, out_channels)

    def forward(self, x):
        """Forward pass through TDNN layer.

        Args:
            x: Input tensor of shape (batch, channels, time).

        Returns:
            Output tensor after linear transformation and nonlinearity.
        """
        x = self.linear(x)
        x = self.nonlinear(x)
        return x


class CAMLayer(nn.Module):
    """Context-Aware Masking layer."""

    def __init__(
        self,
        bn_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        bias,
        reduction=2,
    ):
        """Initialize CAMLayer.

        Args:
            bn_channels: Number of bottleneck channels.
            out_channels: Number of output channels.
            kernel_size: Size of the convolutional kernel.
            stride: Stride for the convolution.
            padding: Padding for the convolution.
            dilation: Dilation rate for the convolution.
            bias: Whether to use bias in convolution.
            reduction: Reduction factor for context computation.
        """
        super(CAMLayer, self).__init__()
        self.linear_local = nn.Conv1d(
            bn_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )
        self.linear1 = nn.Conv1d(bn_channels, bn_channels // reduction, 1)
        self.relu = nn.ReLU(inplace=True)
        self.linear2 = nn.Conv1d(bn_channels // reduction, out_channels, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """Forward pass through CAM layer.

        Args:
            x: Input tensor of shape (batch, channels, time).

        Returns:
            Output tensor with context-aware masking applied.
        """
        y = self.linear_local(x)
        context = x.mean(-1, keepdim=True) + self.seg_pooling(x)
        context = self.relu(self.linear1(context))
        m = self.sigmoid(self.linear2(context))
        return y * m

    def seg_pooling(self, x, seg_len=100, stype="avg"):
        """Apply segment pooling to input tensor.

        Args:
            x: Input tensor to pool.
            seg_len: Length of segments for pooling.
            stype: Type of pooling ('avg' or 'max').

        Returns:
            Pooled tensor expanded back to original time dimension.
        """
        if stype == "avg":
            seg = F.avg_pool1d(x, kernel_size=seg_len, stride=seg_len, ceil_mode=True)
        elif stype == "max":
            seg = F.max_pool1d(x, kernel_size=seg_len, stride=seg_len, ceil_mode=True)
        else:
            raise ValueError("Wrong segment pooling type.")
        shape = seg.shape
        seg = seg.unsqueeze(-1).expand(*shape, seg_len).reshape(*shape[:-1], -1)
        seg = seg[..., : x.shape[-1]]
        return seg


class CAMDenseTDNNLayer(nn.Module):
    """Context-Aware Masking Dense TDNN layer."""

    def __init__(
        self,
        in_channels,
        out_channels,
        bn_channels,
        kernel_size,
        stride=1,
        dilation=1,
        bias=False,
        config_str="batchnorm-relu",
        memory_efficient=False,
    ):
        """Initialize CAMDenseTDNNLayer.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            bn_channels: Number of bottleneck channels.
            kernel_size: Size of the convolutional kernel.
            stride: Stride for the convolution.
            dilation: Dilation rate for the convolution.
            bias: Whether to use bias in convolution.
            config_str: Configuration string for nonlinearity layers.
            memory_efficient: Whether to use memory-efficient implementation.
        """
        super(CAMDenseTDNNLayer, self).__init__()
        assert kernel_size % 2 == 1, "Expect equal paddings, but got even kernel size ({})".format(
            kernel_size,
        )
        padding = (kernel_size - 1) // 2 * dilation
        self.memory_efficient = memory_efficient
        self.nonlinear1 = get_nonlinear(config_str, in_channels)
        self.linear1 = nn.Conv1d(in_channels, bn_channels, 1, bias=False)
        self.nonlinear2 = get_nonlinear(config_str, bn_channels)
        self.cam_layer = CAMLayer(
            bn_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

    def bn_function(self, x):
        """Apply bottleneck function.

        Args:
            x: Input tensor.

        Returns:
            Output after nonlinearity and linear transformation.
        """
        return self.linear1(self.nonlinear1(x))

    def forward(self, x):
        """Forward pass through CAMDenseTDNN layer.

        Args:
            x: Input tensor.

        Returns:
            Output tensor after bottleneck and CAM processing.
        """
        if self.training and self.memory_efficient:
            x = cp.checkpoint(self.bn_function, x)
        else:
            x = self.bn_function(x)
        x = self.cam_layer(self.nonlinear2(x))
        return x


class CAMDenseTDNNBlock(nn.ModuleList):
    """Block of CAMDenseTDNN layers."""

    def __init__(
        self,
        num_layers,
        in_channels,
        out_channels,
        bn_channels,
        kernel_size,
        stride=1,
        dilation=1,
        bias=False,
        config_str="batchnorm-relu",
        memory_efficient=False,
    ):
        """Initialize CAMDenseTDNNBlock.

        Args:
            num_layers: Number of layers in the block.
            in_channels: Number of input channels.
            out_channels: Number of output channels per layer.
            bn_channels: Number of bottleneck channels.
            kernel_size: Size of the convolutional kernel.
            stride: Stride for the convolution.
            dilation: Dilation rate for the convolution.
            bias: Whether to use bias in convolution.
            config_str: Configuration string for nonlinearity layers.
            memory_efficient: Whether to use memory-efficient implementation.
        """
        super(CAMDenseTDNNBlock, self).__init__()
        for i in range(num_layers):
            layer = CAMDenseTDNNLayer(
                in_channels=in_channels + i * out_channels,
                out_channels=out_channels,
                bn_channels=bn_channels,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                bias=bias,
                config_str=config_str,
                memory_efficient=memory_efficient,
            )
            self.add_module("tdnnd%d" % (i + 1), layer)

    def forward(self, x):
        """Forward pass through CAMDenseTDNN block.

        Args:
            x: Input tensor.

        Returns:
            Output tensor with concatenated features from all layers.
        """
        for layer in self:
            x = torch.cat([x, layer(x)], dim=1)
        return x


class TransitLayer(nn.Module):
    """Transition layer for dimension reduction."""

    def __init__(self, in_channels, out_channels, bias=True, config_str="batchnorm-relu"):
        """Initialize TransitLayer.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            bias: Whether to use bias in linear layer.
            config_str: Configuration string for nonlinearity layers.
        """
        super(TransitLayer, self).__init__()
        self.nonlinear = get_nonlinear(config_str, in_channels)
        self.linear = nn.Conv1d(in_channels, out_channels, 1, bias=bias)

    def forward(self, x):
        """Forward pass through transit layer.

        Args:
            x: Input tensor.

        Returns:
            Output tensor after nonlinearity and dimension reduction.
        """
        x = self.nonlinear(x)
        x = self.linear(x)
        return x


class DenseLayer(nn.Module):
    """Dense layer with optional nonlinearity."""

    def __init__(self, in_channels, out_channels, bias=False, config_str="batchnorm-relu"):
        """Initialize DenseLayer.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            bias: Whether to use bias in linear layer.
            config_str: Configuration string for nonlinearity layers.
        """
        super(DenseLayer, self).__init__()
        self.linear = nn.Conv1d(in_channels, out_channels, 1, bias=bias)
        self.nonlinear = get_nonlinear(config_str, out_channels)

    def forward(self, x):
        """Forward pass through dense layer.

        Args:
            x: Input tensor (2D or 3D).

        Returns:
            Output tensor after linear transformation and nonlinearity.
        """
        if len(x.shape) == 2:
            x = self.linear(x.unsqueeze(dim=-1)).squeeze(dim=-1)
        else:
            x = self.linear(x)
        x = self.nonlinear(x)
        return x


class BasicResBlock(nn.Module):
    """Basic residual block for 2D convolutions."""

    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        """Initialize BasicResBlock.

        Args:
            in_planes: Number of input channels.
            planes: Number of output channels.
            stride: Stride for the first convolution.
        """
        super(BasicResBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes,
            planes,
            kernel_size=3,
            stride=(stride, 1),
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=(stride, 1),
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        """Forward pass through basic residual block.

        Args:
            x: Input tensor.

        Returns:
            Output tensor with residual connection.
        """
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

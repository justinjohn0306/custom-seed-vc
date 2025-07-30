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

"""Neural network modules and layers for OpenVoice."""

import math
from typing import Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import Conv1d
from torch.nn import functional as F
from torch.nn.utils import remove_weight_norm, weight_norm

from . import commons
from .attentions import Encoder
from .commons import get_padding, init_weights
from .transforms import piecewise_rational_quadratic_transform

LRELU_SLOPE = 0.1


class LayerNorm(nn.Module):
    """Layer normalization module for channel-wise normalization."""

    def __init__(self, channels: int, eps: float = 1e-5) -> None:
        """Initialize LayerNorm.

        Args:
            channels: Number of channels to normalize.
            eps: Small value to avoid division by zero.
        """
        super().__init__()
        self.channels = channels
        self.eps = eps

        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply layer normalization.

        Args:
            x: Input tensor of shape [batch, channels, time].

        Returns:
            Normalized tensor of same shape.
        """
        x = x.transpose(1, -1)
        x = F.layer_norm(x, (self.channels,), self.gamma, self.beta, self.eps)
        return x.transpose(1, -1)


class ConvReluNorm(nn.Module):
    """Convolutional block with ReLU activation and layer normalization."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        kernel_size: int,
        n_layers: int,
        p_dropout: float,
    ) -> None:
        """Initialize ConvReluNorm.

        Args:
            in_channels: Number of input channels.
            hidden_channels: Number of hidden channels.
            out_channels: Number of output channels.
            kernel_size: Kernel size for convolutions.
            n_layers: Number of convolutional layers.
            p_dropout: Dropout probability.
        """
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.p_dropout = p_dropout
        assert n_layers > 1, "Number of layers should be larger than 0."

        self.conv_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        self.conv_layers.append(
            nn.Conv1d(in_channels, hidden_channels, kernel_size, padding=kernel_size // 2),
        )
        self.norm_layers.append(LayerNorm(hidden_channels))
        self.relu_drop = nn.Sequential(nn.ReLU(), nn.Dropout(p_dropout))
        for _ in range(n_layers - 1):
            self.conv_layers.append(
                nn.Conv1d(
                    hidden_channels,
                    hidden_channels,
                    kernel_size,
                    padding=kernel_size // 2,
                ),
            )
            self.norm_layers.append(LayerNorm(hidden_channels))
        self.proj = nn.Conv1d(hidden_channels, out_channels, 1)
        self.proj.weight.data.zero_()
        self.proj.bias.data.zero_()

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor) -> torch.Tensor:
        """Forward pass through ConvReluNorm.

        Args:
            x: Input tensor of shape [batch, channels, time].
            x_mask: Mask tensor.

        Returns:
            Output tensor with residual connection.
        """
        x_org = x
        for i in range(self.n_layers):
            x = self.conv_layers[i](x * x_mask)
            x = self.norm_layers[i](x)
            x = self.relu_drop(x)
        x = x_org + self.proj(x)
        return x * x_mask


class DDSConv(nn.Module):
    """Dilated and Depth-Separable Convolution.

    Uses depth-separable convolutions with increasing dilation rates.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        n_layers: int,
        p_dropout: float = 0.0,
    ) -> None:
        """Initialize DDSConv.

        Args:
            channels: Number of channels.
            kernel_size: Base kernel size.
            n_layers: Number of layers.
            p_dropout: Dropout probability.
        """
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.p_dropout = p_dropout

        self.drop = nn.Dropout(p_dropout)
        self.convs_sep = nn.ModuleList()
        self.convs_1x1 = nn.ModuleList()
        self.norms_1 = nn.ModuleList()
        self.norms_2 = nn.ModuleList()
        for i in range(n_layers):
            dilation = kernel_size**i
            padding = (kernel_size * dilation - dilation) // 2
            self.convs_sep.append(
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    groups=channels,
                    dilation=dilation,
                    padding=padding,
                ),
            )
            self.convs_1x1.append(nn.Conv1d(channels, channels, 1))
            self.norms_1.append(LayerNorm(channels))
            self.norms_2.append(LayerNorm(channels))

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        g: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through DDSConv.

        Args:
            x: Input tensor.
            x_mask: Mask tensor.
            g: Optional global conditioning.

        Returns:
            Output tensor.
        """
        if g is not None:
            x = x + g
        for i in range(self.n_layers):
            y = self.convs_sep[i](x * x_mask)
            y = self.norms_1[i](y)
            y = F.gelu(y)
            y = self.convs_1x1[i](y)
            y = self.norms_2[i](y)
            y = F.gelu(y)
            y = self.drop(y)
            x = x + y
        return x * x_mask


class WN(torch.nn.Module):
    """WaveNet module with gated convolutions.

    Implements a stack of dilated causal convolutions with gating mechanism.
    """

    def __init__(
        self,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        n_layers: int,
        gin_channels: int = 0,
        p_dropout: float = 0,
    ) -> None:
        """Initialize WN.

        Args:
            hidden_channels: Number of hidden channels.
            kernel_size: Kernel size for convolutions.
            dilation_rate: Base dilation rate.
            n_layers: Number of WaveNet layers.
            gin_channels: Number of global conditioning channels.
            p_dropout: Dropout probability.
        """
        super(WN, self).__init__()
        assert kernel_size % 2 == 1
        self.hidden_channels = hidden_channels
        self.kernel_size = (kernel_size,)
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels
        self.p_dropout = p_dropout

        self.in_layers = torch.nn.ModuleList()
        self.res_skip_layers = torch.nn.ModuleList()
        self.drop = nn.Dropout(p_dropout)

        if gin_channels != 0:
            cond_layer = torch.nn.Conv1d(gin_channels, 2 * hidden_channels * n_layers, 1)
            self.cond_layer = torch.nn.utils.weight_norm(cond_layer, name="weight")

        for i in range(n_layers):
            dilation = dilation_rate**i
            padding = int((kernel_size * dilation - dilation) / 2)
            in_layer = torch.nn.Conv1d(
                hidden_channels,
                2 * hidden_channels,
                kernel_size,
                dilation=dilation,
                padding=padding,
            )
            in_layer = torch.nn.utils.weight_norm(in_layer, name="weight")
            self.in_layers.append(in_layer)

            # last one is not necessary
            if i < n_layers - 1:
                res_skip_channels = 2 * hidden_channels
            else:
                res_skip_channels = hidden_channels

            res_skip_layer = torch.nn.Conv1d(hidden_channels, res_skip_channels, 1)
            res_skip_layer = torch.nn.utils.weight_norm(res_skip_layer, name="weight")
            self.res_skip_layers.append(res_skip_layer)

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        g: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass through WaveNet.

        Args:
            x: Input tensor.
            x_mask: Mask tensor.
            g: Optional global conditioning.
            **kwargs: Additional arguments.

        Returns:
            Output tensor.
        """
        output = torch.zeros_like(x)
        n_channels_tensor = torch.IntTensor([self.hidden_channels])

        if g is not None:
            g = self.cond_layer(g)

        for i in range(self.n_layers):
            x_in = self.in_layers[i](x)
            if g is not None:
                cond_offset = i * 2 * self.hidden_channels
                g_l = g[:, cond_offset : cond_offset + 2 * self.hidden_channels, :]
            else:
                g_l = torch.zeros_like(x_in)

            acts = commons.fused_add_tanh_sigmoid_multiply(x_in, g_l, n_channels_tensor)
            acts = self.drop(acts)

            res_skip_acts = self.res_skip_layers[i](acts)
            if i < self.n_layers - 1:
                res_acts = res_skip_acts[:, : self.hidden_channels, :]
                x = (x + res_acts) * x_mask
                output = output + res_skip_acts[:, self.hidden_channels :, :]
            else:
                output = output + res_skip_acts
        return output * x_mask

    def remove_weight_norm(self) -> None:
        """Remove weight normalization from all layers."""
        if self.gin_channels != 0:
            torch.nn.utils.remove_weight_norm(self.cond_layer)
        for l in self.in_layers:
            torch.nn.utils.remove_weight_norm(l)
        for l in self.res_skip_layers:
            torch.nn.utils.remove_weight_norm(l)


class ResBlock1(torch.nn.Module):
    """Residual block type 1 with multiple dilations."""

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: Tuple[int, int, int] = (1, 3, 5),
    ) -> None:
        """Initialize ResBlock1.

        Args:
            channels: Number of channels.
            kernel_size: Kernel size for convolutions.
            dilation: Tuple of dilation rates.
        """
        super(ResBlock1, self).__init__()
        self.convs1 = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[0],
                        padding=get_padding(kernel_size, dilation[0]),
                    ),
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[1],
                        padding=get_padding(kernel_size, dilation[1]),
                    ),
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[2],
                        padding=get_padding(kernel_size, dilation[2]),
                    ),
                ),
            ],
        )
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    ),
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    ),
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    ),
                ),
            ],
        )
        self.convs2.apply(init_weights)

    def forward(self, x: torch.Tensor, x_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through ResBlock1.

        Args:
            x: Input tensor.
            x_mask: Optional mask tensor.

        Returns:
            Output tensor with residual connections.
        """
        for c1, c2 in zip(self.convs1, self.convs2, strict=False):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            if x_mask is not None:
                xt = xt * x_mask
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            if x_mask is not None:
                xt = xt * x_mask
            xt = c2(xt)
            x = xt + x
        if x_mask is not None:
            x = x * x_mask
        return x

    def remove_weight_norm(self) -> None:
        """Remove weight normalization from convolution layers."""
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)


class ResBlock2(torch.nn.Module):
    """Residual block type 2 with two dilations."""

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: Tuple[int, int] = (1, 3),
    ) -> None:
        """Initialize ResBlock2.

        Args:
            channels: Number of channels.
            kernel_size: Kernel size for convolutions.
            dilation: Tuple of dilation rates.
        """
        super(ResBlock2, self).__init__()
        self.convs = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[0],
                        padding=get_padding(kernel_size, dilation[0]),
                    ),
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[1],
                        padding=get_padding(kernel_size, dilation[1]),
                    ),
                ),
            ],
        )
        self.convs.apply(init_weights)

    def forward(self, x: torch.Tensor, x_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through ResBlock2.

        Args:
            x: Input tensor.
            x_mask: Optional mask tensor.

        Returns:
            Output tensor with residual connections.
        """
        for c in self.convs:
            xt = F.leaky_relu(x, LRELU_SLOPE)
            if x_mask is not None:
                xt = xt * x_mask
            xt = c(xt)
            x = xt + x
        if x_mask is not None:
            x = x * x_mask
        return x

    def remove_weight_norm(self) -> None:
        """Remove weight normalization from convolution layers."""
        for l in self.convs:
            remove_weight_norm(l)


class Log(nn.Module):
    """Logarithm flow layer for normalizing flows."""

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        reverse: bool = False,
        **kwargs,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Apply log transform.

        Args:
            x: Input tensor.
            x_mask: Mask tensor.
            reverse: Whether to apply inverse transform.
            **kwargs: Additional arguments.

        Returns:
            Transformed tensor and log determinant (forward) or just tensor (reverse).
        """
        if not reverse:
            y = torch.log(torch.clamp_min(x, 1e-5)) * x_mask
            logdet = torch.sum(-y, [1, 2])
            return y, logdet
        else:
            x = torch.exp(x) * x_mask
            return x


class Flip(nn.Module):
    """Channel flip layer for normalizing flows."""

    def forward(
        self,
        x: torch.Tensor,
        *args,
        reverse: bool = False,
        **kwargs,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Flip channels.

        Args:
            x: Input tensor.
            *args: Additional positional arguments.
            reverse: Whether to apply inverse transform.
            **kwargs: Additional keyword arguments.

        Returns:
            Flipped tensor and log determinant (forward) or just tensor (reverse).
        """
        x = torch.flip(x, [1])
        if not reverse:
            logdet = torch.zeros(x.size(0)).to(dtype=x.dtype, device=x.device)
            return x, logdet
        else:
            return x


class ElementwiseAffine(nn.Module):
    """Elementwise affine transformation layer."""

    def __init__(self, channels: int) -> None:
        """Initialize ElementwiseAffine.

        Args:
            channels: Number of channels.
        """
        super().__init__()
        self.channels = channels
        self.m = nn.Parameter(torch.zeros(channels, 1))
        self.logs = nn.Parameter(torch.zeros(channels, 1))

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        reverse: bool = False,
        **kwargs,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Apply elementwise affine transform.

        Args:
            x: Input tensor.
            x_mask: Mask tensor.
            reverse: Whether to apply inverse transform.
            **kwargs: Additional arguments.

        Returns:
            Transformed tensor and log determinant (forward) or just tensor (reverse).
        """
        if not reverse:
            y = self.m + torch.exp(self.logs) * x
            y = y * x_mask
            logdet = torch.sum(self.logs * x_mask, [1, 2])
            return y, logdet
        else:
            x = (x - self.m) * torch.exp(-self.logs) * x_mask
            return x


class ResidualCouplingLayer(nn.Module):
    """Residual coupling layer for normalizing flows.

    Implements affine coupling with WaveNet-based transformation network
    for flow-based generative models.
    """

    def __init__(
        self,
        channels,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_layers,
        p_dropout=0,
        gin_channels=0,
        mean_only=False,
    ):
        """Initialize ResidualCouplingLayer.

        Args:
            channels: Number of input channels (must be divisible by 2)
            hidden_channels: Number of hidden channels in WaveNet
            kernel_size: Kernel size for WaveNet layers
            dilation_rate: Dilation rate for WaveNet layers
            n_layers: Number of layers in WaveNet
            p_dropout: Dropout probability. Defaults to 0.
            gin_channels: Number of global conditioning channels. Defaults to 0.
            mean_only: If True, only predict mean (no log variance). Defaults to False.
        """
        assert channels % 2 == 0, "channels should be divisible by 2"
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.half_channels = channels // 2
        self.mean_only = mean_only

        self.pre = nn.Conv1d(self.half_channels, hidden_channels, 1)
        self.enc = WN(
            hidden_channels,
            kernel_size,
            dilation_rate,
            n_layers,
            p_dropout=p_dropout,
            gin_channels=gin_channels,
        )
        self.post = nn.Conv1d(hidden_channels, self.half_channels * (2 - mean_only), 1)
        self.post.weight.data.zero_()
        self.post.bias.data.zero_()

    def forward(self, x, x_mask, g=None, reverse=False):
        """Forward pass through residual coupling layer.

        Args:
            x: Input tensor.
            x_mask: Mask tensor.
            g: Global conditioning tensor.
            reverse: Whether to reverse the flow.

        Returns:
            Output tensor and log-determinant.
        """
        x0, x1 = torch.split(x, [self.half_channels] * 2, 1)
        h = self.pre(x0) * x_mask
        h = self.enc(h, x_mask, g=g)
        stats = self.post(h) * x_mask
        if not self.mean_only:
            m, logs = torch.split(stats, [self.half_channels] * 2, 1)
        else:
            m = stats
            logs = torch.zeros_like(m)

        if not reverse:
            x1 = m + x1 * torch.exp(logs) * x_mask
            x = torch.cat([x0, x1], 1)
            logdet = torch.sum(logs, [1, 2])
            return x, logdet
        else:
            x1 = (x1 - m) * torch.exp(-logs) * x_mask
            x = torch.cat([x0, x1], 1)
            return x


class ConvFlow(nn.Module):
    """Convolutional flow layer using rational quadratic splines.

    Implements a normalizing flow transformation using piecewise rational
    quadratic transforms with convolutional conditioning networks.
    """

    def __init__(
        self,
        in_channels,
        filter_channels,
        kernel_size,
        n_layers,
        num_bins=10,
        tail_bound=5.0,
    ):
        """Initialize ConvFlow.

        Args:
            in_channels: Number of input channels
            filter_channels: Number of filter channels in convolution layers
            kernel_size: Kernel size for convolution layers
            n_layers: Number of layers in DDSConv
            num_bins: Number of bins for piecewise rational quadratic transform. Defaults to 10.
            tail_bound: Bound for transform tails. Defaults to 5.0.
        """
        super().__init__()
        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.num_bins = num_bins
        self.tail_bound = tail_bound
        self.half_channels = in_channels // 2

        self.pre = nn.Conv1d(self.half_channels, filter_channels, 1)
        self.convs = DDSConv(filter_channels, kernel_size, n_layers, p_dropout=0.0)
        self.proj = nn.Conv1d(filter_channels, self.half_channels * (num_bins * 3 - 1), 1)
        self.proj.weight.data.zero_()
        self.proj.bias.data.zero_()

    def forward(self, x, x_mask, g=None, reverse=False):
        """Forward pass through ConvFlow.

        Args:
            x: Input tensor.
            x_mask: Mask tensor.
            g: Global conditioning tensor.
            reverse: Whether to reverse the flow.

        Returns:
            Output tensor and log-determinant.
        """
        x0, x1 = torch.split(x, [self.half_channels] * 2, 1)
        h = self.pre(x0)
        h = self.convs(h, x_mask, g=g)
        h = self.proj(h) * x_mask

        b, c, t = x0.shape
        h = h.reshape(b, c, -1, t).permute(0, 1, 3, 2)  # [b, cx?, t] -> [b, c, t, ?]

        unnormalized_widths = h[..., : self.num_bins] / math.sqrt(self.filter_channels)
        unnormalized_heights = h[..., self.num_bins : 2 * self.num_bins] / math.sqrt(
            self.filter_channels,
        )
        unnormalized_derivatives = h[..., 2 * self.num_bins :]

        x1, logabsdet = piecewise_rational_quadratic_transform(
            x1,
            unnormalized_widths,
            unnormalized_heights,
            unnormalized_derivatives,
            inverse=reverse,
            tails="linear",
            tail_bound=self.tail_bound,
        )

        x = torch.cat([x0, x1], 1) * x_mask
        logdet = torch.sum(logabsdet * x_mask, [1, 2])
        if not reverse:
            return x, logdet
        else:
            return x


class TransformerCouplingLayer(nn.Module):
    """Transformer-based coupling layer for normalizing flows.

    Uses transformer encoder instead of WaveNet for the transformation
    network in affine coupling layers.
    """

    def __init__(
        self,
        channels,
        hidden_channels,
        kernel_size,
        n_layers,
        n_heads,
        p_dropout=0,
        filter_channels=0,
        mean_only=False,
        wn_sharing_parameter=None,
        gin_channels=0,
    ):
        """Initialize TransformerCouplingLayer.

        Args:
            channels: Number of input channels (must be divisible by 2)
            hidden_channels: Number of hidden channels
            kernel_size: Kernel size for convolution layers
            n_layers: Number of transformer layers (must be 3)
            n_heads: Number of attention heads
            p_dropout: Dropout probability. Defaults to 0.
            filter_channels: Number of filter channels. Defaults to 0.
            mean_only: If True, only predict mean (no log variance). Defaults to False.
            wn_sharing_parameter: Shared parameters for WaveNet. Defaults to None.
            gin_channels: Number of global conditioning channels. Defaults to 0.
        """
        assert n_layers == 3, n_layers
        assert channels % 2 == 0, "channels should be divisible by 2"
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.half_channels = channels // 2
        self.mean_only = mean_only

        self.pre = nn.Conv1d(self.half_channels, hidden_channels, 1)
        self.enc = (
            Encoder(
                hidden_channels,
                filter_channels,
                n_heads,
                n_layers,
                kernel_size,
                p_dropout,
                isflow=True,
                gin_channels=gin_channels,
            )
            if wn_sharing_parameter is None
            else wn_sharing_parameter
        )
        self.post = nn.Conv1d(hidden_channels, self.half_channels * (2 - mean_only), 1)
        self.post.weight.data.zero_()
        self.post.bias.data.zero_()

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        g: Optional[torch.Tensor] = None,
        reverse: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Apply transformer coupling transform.

        Args:
            x: Input tensor.
            x_mask: Mask tensor.
            g: Optional global conditioning.
            reverse: Whether to apply inverse transform.

        Returns:
            Transformed tensor and log determinant (forward) or just tensor (reverse).
        """
        x0, x1 = torch.split(x, [self.half_channels] * 2, 1)
        h = self.pre(x0) * x_mask
        h = self.enc(h, x_mask, g=g)
        stats = self.post(h) * x_mask
        if not self.mean_only:
            m, logs = torch.split(stats, [self.half_channels] * 2, 1)
        else:
            m = stats
            logs = torch.zeros_like(m)

        if not reverse:
            x1 = m + x1 * torch.exp(logs) * x_mask
            x = torch.cat([x0, x1], 1)
            logdet = torch.sum(logs, [1, 2])
            return x, logdet
        else:
            x1 = (x1 - m) * torch.exp(-logs) * x_mask
            x = torch.cat([x0, x1], 1)
            return x

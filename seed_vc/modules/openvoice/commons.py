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

"""Common utilities for OpenVoice models."""

import math
from typing import List, Optional, Tuple, Union

import torch
from torch.nn import functional as F


def init_weights(m: torch.nn.Module, mean: float = 0.0, std: float = 0.01) -> None:
    """Initialize weights of convolutional layers.

    Args:
        m: Module to initialize.
        mean: Mean of the normal distribution.
        std: Standard deviation of the normal distribution.
    """
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def get_padding(kernel_size: int, dilation: int = 1) -> int:
    """Calculate padding size for 'same' convolution.

    Args:
        kernel_size: Size of the convolutional kernel.
        dilation: Dilation rate.

    Returns:
        Padding size.
    """
    return int((kernel_size * dilation - dilation) / 2)


def convert_pad_shape(pad_shape: List[List[int]]) -> List[int]:
    """Convert padding shape to PyTorch format.

    Args:
        pad_shape: Padding shape as nested list.

    Returns:
        Flattened padding shape for F.pad.
    """
    layer = pad_shape[::-1]
    pad_shape = [item for sublist in layer for item in sublist]
    return pad_shape


def intersperse(lst: List, item) -> List:
    """Intersperse an item between list elements.

    Args:
        lst: Input list.
        item: Item to intersperse.

    Returns:
        List with item interspersed between elements.
    """
    result = [item] * (len(lst) * 2 + 1)
    result[1::2] = lst
    return result


def kl_divergence(
    m_p: torch.Tensor,
    logs_p: torch.Tensor,
    m_q: torch.Tensor,
    logs_q: torch.Tensor,
) -> torch.Tensor:
    """Calculate KL divergence KL(P||Q).

    Args:
        m_p: Mean of distribution P.
        logs_p: Log standard deviation of distribution P.
        m_q: Mean of distribution Q.
        logs_q: Log standard deviation of distribution Q.

    Returns:
        KL divergence between distributions.
    """
    kl = (logs_q - logs_p) - 0.5
    kl += 0.5 * (torch.exp(2.0 * logs_p) + ((m_p - m_q) ** 2)) * torch.exp(-2.0 * logs_q)
    return kl


def rand_gumbel(shape: Union[torch.Size, Tuple[int, ...]]) -> torch.Tensor:
    """Sample from the Gumbel distribution, protect from overflows.

    Args:
        shape: Shape of the output tensor.

    Returns:
        Samples from Gumbel distribution.
    """
    uniform_samples = torch.rand(shape) * 0.99998 + 0.00001
    return -torch.log(-torch.log(uniform_samples))


def rand_gumbel_like(x: torch.Tensor) -> torch.Tensor:
    """Sample from Gumbel distribution with same shape as input.

    Args:
        x: Input tensor to match shape and device.

    Returns:
        Gumbel samples with same shape as input.
    """
    g = rand_gumbel(x.size()).to(dtype=x.dtype, device=x.device)
    return g


def slice_segments(x: torch.Tensor, ids_str: torch.Tensor, segment_size: int = 4) -> torch.Tensor:
    """Slice segments from input tensor.

    Args:
        x: Input tensor of shape [batch, channels, time].
        ids_str: Starting indices for each batch.
        segment_size: Size of segments to extract.

    Returns:
        Sliced segments of shape [batch, channels, segment_size].
    """
    ret = torch.zeros_like(x[:, :, :segment_size])
    for i in range(x.size(0)):
        idx_str = ids_str[i]
        idx_end = idx_str + segment_size
        ret[i] = x[i, :, idx_str:idx_end]
    return ret


def rand_slice_segments(
    x: torch.Tensor,
    x_lengths: Optional[torch.Tensor] = None,
    segment_size: int = 4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Randomly slice segments from input tensor.

    Args:
        x: Input tensor of shape [batch, channels, time].
        x_lengths: Lengths of sequences in batch.
        segment_size: Size of segments to extract.

    Returns:
        Tuple of (sliced segments, starting indices).
    """
    b, d, t = x.size()
    if x_lengths is None:
        x_lengths = t
    ids_str_max = x_lengths - segment_size + 1
    ids_str = (torch.rand([b]).to(device=x.device) * ids_str_max).to(dtype=torch.long)
    ret = slice_segments(x, ids_str, segment_size)
    return ret, ids_str


def get_timing_signal_1d(
    length: int,
    channels: int,
    min_timescale: float = 1.0,
    max_timescale: float = 1.0e4,
) -> torch.Tensor:
    """Generate sinusoidal timing signal.

    Args:
        length: Sequence length.
        channels: Number of channels.
        min_timescale: Minimum timescale.
        max_timescale: Maximum timescale.

    Returns:
        Timing signal of shape [1, channels, length].
    """
    position = torch.arange(length, dtype=torch.float)
    num_timescales = channels // 2
    log_timescale_increment = math.log(float(max_timescale) / float(min_timescale)) / (
        num_timescales - 1
    )
    inv_timescales = min_timescale * torch.exp(
        torch.arange(num_timescales, dtype=torch.float) * -log_timescale_increment,
    )
    scaled_time = position.unsqueeze(0) * inv_timescales.unsqueeze(1)
    signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], 0)
    signal = F.pad(signal, [0, 0, 0, channels % 2])
    signal = signal.view(1, channels, length)
    return signal


def add_timing_signal_1d(
    x: torch.Tensor,
    min_timescale: float = 1.0,
    max_timescale: float = 1.0e4,
) -> torch.Tensor:
    """Add timing signal to input tensor.

    Args:
        x: Input tensor of shape [batch, channels, length].
        min_timescale: Minimum timescale.
        max_timescale: Maximum timescale.

    Returns:
        Tensor with timing signal added.
    """
    b, channels, length = x.size()
    signal = get_timing_signal_1d(length, channels, min_timescale, max_timescale)
    return x + signal.to(dtype=x.dtype, device=x.device)


def cat_timing_signal_1d(
    x: torch.Tensor,
    min_timescale: float = 1.0,
    max_timescale: float = 1.0e4,
    axis: int = 1,
) -> torch.Tensor:
    """Concatenate timing signal to input tensor.

    Args:
        x: Input tensor of shape [batch, channels, length].
        min_timescale: Minimum timescale.
        max_timescale: Maximum timescale.
        axis: Axis to concatenate along.

    Returns:
        Tensor with timing signal concatenated.
    """
    b, channels, length = x.size()
    signal = get_timing_signal_1d(length, channels, min_timescale, max_timescale)
    return torch.cat([x, signal.to(dtype=x.dtype, device=x.device)], axis)


def subsequent_mask(length: int) -> torch.Tensor:
    """Generate subsequent mask for masked self-attention.

    Args:
        length: Sequence length.

    Returns:
        Lower triangular mask of shape [1, 1, length, length].
    """
    mask = torch.tril(torch.ones(length, length)).unsqueeze(0).unsqueeze(0)
    return mask


@torch.jit.script
def fused_add_tanh_sigmoid_multiply(
    input_a: torch.Tensor,
    input_b: torch.Tensor,
    n_channels: torch.Tensor,
) -> torch.Tensor:
    """Fused operation for add, tanh, sigmoid, and multiply.

    Args:
        input_a: First input tensor.
        input_b: Second input tensor.
        n_channels: Number of channels for tanh activation.

    Returns:
        Result of fused operations.
    """
    n_channels_int = n_channels[0]
    in_act = input_a + input_b
    t_act = torch.tanh(in_act[:, :n_channels_int, :])
    s_act = torch.sigmoid(in_act[:, n_channels_int:, :])
    acts = t_act * s_act
    return acts


def shift_1d(x: torch.Tensor) -> torch.Tensor:
    """Shift tensor by one position along time dimension.

    Args:
        x: Input tensor.

    Returns:
        Shifted tensor.
    """
    x = F.pad(x, convert_pad_shape([[0, 0], [0, 0], [1, 0]]))[:, :, :-1]
    return x


def sequence_mask(length: torch.Tensor, max_length: Optional[int] = None) -> torch.Tensor:
    """Generate sequence mask from lengths.

    Args:
        length: Tensor of sequence lengths.
        max_length: Maximum sequence length.

    Returns:
        Boolean mask tensor.
    """
    if max_length is None:
        max_length = length.max()
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)


def generate_path(duration: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Generate alignment path from duration.

    Args:
        duration: Duration tensor of shape [b, 1, t_x].
        mask: Mask tensor of shape [b, 1, t_y, t_x].

    Returns:
        Path tensor of shape [b, 1, t_y, t_x].
    """
    b, _, t_y, t_x = mask.shape
    cum_duration = torch.cumsum(duration, -1)

    cum_duration_flat = cum_duration.view(b * t_x)
    path = sequence_mask(cum_duration_flat, t_y).to(mask.dtype)
    path = path.view(b, t_x, t_y)
    path = path - F.pad(path, convert_pad_shape([[0, 0], [1, 0], [0, 0]]))[:, :-1]
    path = path.unsqueeze(1).transpose(2, 3) * mask
    return path


def clip_grad_value_(
    parameters: Union[torch.Tensor, List[torch.Tensor]],
    clip_value: Optional[float],
    norm_type: float = 2,
) -> float:
    """Clip gradients by value.

    Args:
        parameters: Model parameters or list of parameters.
        clip_value: Maximum gradient value.
        norm_type: Type of norm to compute.

    Returns:
        Total norm of gradients.
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    if clip_value is not None:
        clip_value = float(clip_value)

    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
        if clip_value is not None:
            p.grad.data.clamp_(min=-clip_value, max=clip_value)
    total_norm = total_norm ** (1.0 / norm_type)
    return total_norm

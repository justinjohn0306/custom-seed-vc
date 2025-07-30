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

"""Attention mechanisms for OpenVoice models."""

import logging
import math
from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from . import commons

logger = logging.getLogger(__name__)


class LayerNorm(nn.Module):
    """Layer normalization module.

    Applies layer normalization over the channel dimension.
    """

    def __init__(self, channels: int, eps: float = 1e-5) -> None:
        """Initialize LayerNorm.

        Args:
            channels: Number of channels.
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
            Normalized tensor of same shape as input.
        """
        x = x.transpose(1, -1)
        x = F.layer_norm(x, (self.channels,), self.gamma, self.beta, self.eps)
        return x.transpose(1, -1)


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


class Encoder(nn.Module):
    """Transformer encoder with multi-head attention.

    Implements a stack of transformer encoder layers with speaker embedding conditioning.
    """

    def __init__(
        self,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        n_layers: int,
        kernel_size: int = 1,
        p_dropout: float = 0.0,
        window_size: int = 4,
        isflow: bool = True,
        **kwargs,
    ) -> None:
        """Initialize Encoder.

        Args:
            hidden_channels: Number of hidden channels.
            filter_channels: Number of filter channels in FFN.
            n_heads: Number of attention heads.
            n_layers: Number of encoder layers.
            kernel_size: Kernel size for FFN.
            p_dropout: Dropout probability.
            window_size: Window size for relative positional encoding.
            isflow: Whether this is a flow model (unused).
            **kwargs: Additional arguments including gin_channels and cond_layer_idx.
        """
        super().__init__()
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.window_size = window_size
        # if isflow:
        #  cond_layer = torch.nn.Conv1d(256, 2*hidden_channels*n_layers, 1)
        #  self.cond_pre = torch.nn.Conv1d(hidden_channels, 2*hidden_channels, 1)
        #  self.cond_layer = weight_norm(cond_layer, name='weight')
        #  self.gin_channels = 256
        self.cond_layer_idx = self.n_layers
        if "gin_channels" in kwargs:
            self.gin_channels = kwargs["gin_channels"]
            if self.gin_channels != 0:
                self.spk_emb_linear = nn.Linear(self.gin_channels, self.hidden_channels)
                # vits2 says 3rd block, so idx is 2 by default
                self.cond_layer_idx = kwargs["cond_layer_idx"] if "cond_layer_idx" in kwargs else 2
                # logging.debug(self.gin_channels, self.cond_layer_idx)
                assert self.cond_layer_idx < self.n_layers, (
                    "cond_layer_idx should be less than n_layers"
                )
        self.drop = nn.Dropout(p_dropout)
        self.attn_layers = nn.ModuleList()
        self.norm_layers_1 = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        self.norm_layers_2 = nn.ModuleList()

        for _i in range(self.n_layers):
            self.attn_layers.append(
                MultiHeadAttention(
                    hidden_channels,
                    hidden_channels,
                    n_heads,
                    p_dropout=p_dropout,
                    window_size=window_size,
                ),
            )
            self.norm_layers_1.append(LayerNorm(hidden_channels))
            self.ffn_layers.append(
                FFN(
                    hidden_channels,
                    hidden_channels,
                    filter_channels,
                    kernel_size,
                    p_dropout=p_dropout,
                ),
            )
            self.norm_layers_2.append(LayerNorm(hidden_channels))

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        g: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through encoder.

        Args:
            x: Input tensor of shape [batch, hidden_channels, time].
            x_mask: Mask tensor of shape [batch, 1, time].
            g: Optional speaker embedding of shape [batch, gin_channels, 1].

        Returns:
            Encoded representation of shape [batch, hidden_channels, time].
        """
        attn_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(-1)
        x = x * x_mask
        for i in range(self.n_layers):
            if i == self.cond_layer_idx and g is not None:
                g = self.spk_emb_linear(g.transpose(1, 2))
                g = g.transpose(1, 2)
                x = x + g
                x = x * x_mask
            y = self.attn_layers[i](x, x, attn_mask)
            y = self.drop(y)
            x = self.norm_layers_1[i](x + y)

            y = self.ffn_layers[i](x, x_mask)
            y = self.drop(y)
            x = self.norm_layers_2[i](x + y)
        x = x * x_mask
        return x


class Decoder(nn.Module):
    """Transformer decoder with multi-head attention.

    Implements a stack of transformer decoder layers with self-attention and cross-attention.
    """

    def __init__(
        self,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        n_layers: int,
        kernel_size: int = 1,
        p_dropout: float = 0.0,
        proximal_bias: bool = False,
        proximal_init: bool = True,
        **kwargs,
    ) -> None:
        """Initialize Decoder.

        Args:
            hidden_channels: Number of hidden channels.
            filter_channels: Number of filter channels in FFN.
            n_heads: Number of attention heads.
            n_layers: Number of decoder layers.
            kernel_size: Kernel size for FFN.
            p_dropout: Dropout probability.
            proximal_bias: Whether to use proximal bias in self-attention.
            proximal_init: Whether to initialize keys with queries.
            **kwargs: Additional arguments.
        """
        super().__init__()
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.proximal_bias = proximal_bias
        self.proximal_init = proximal_init

        self.drop = nn.Dropout(p_dropout)
        self.self_attn_layers = nn.ModuleList()
        self.norm_layers_0 = nn.ModuleList()
        self.encdec_attn_layers = nn.ModuleList()
        self.norm_layers_1 = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        self.norm_layers_2 = nn.ModuleList()
        for _i in range(self.n_layers):
            self.self_attn_layers.append(
                MultiHeadAttention(
                    hidden_channels,
                    hidden_channels,
                    n_heads,
                    p_dropout=p_dropout,
                    proximal_bias=proximal_bias,
                    proximal_init=proximal_init,
                ),
            )
            self.norm_layers_0.append(LayerNorm(hidden_channels))
            self.encdec_attn_layers.append(
                MultiHeadAttention(hidden_channels, hidden_channels, n_heads, p_dropout=p_dropout),
            )
            self.norm_layers_1.append(LayerNorm(hidden_channels))
            self.ffn_layers.append(
                FFN(
                    hidden_channels,
                    hidden_channels,
                    filter_channels,
                    kernel_size,
                    p_dropout=p_dropout,
                    causal=True,
                ),
            )
            self.norm_layers_2.append(LayerNorm(hidden_channels))

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        h: torch.Tensor,
        h_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass through decoder.

        Args:
            x: Decoder input of shape [batch, hidden_channels, time].
            x_mask: Decoder mask of shape [batch, 1, time].
            h: Encoder output of shape [batch, hidden_channels, time].
            h_mask: Encoder mask of shape [batch, 1, time].

        Returns:
            Decoded representation of shape [batch, hidden_channels, time].
        """
        self_attn_mask = commons.subsequent_mask(x_mask.size(2)).to(device=x.device, dtype=x.dtype)
        encdec_attn_mask = h_mask.unsqueeze(2) * x_mask.unsqueeze(-1)
        x = x * x_mask
        for i in range(self.n_layers):
            y = self.self_attn_layers[i](x, x, self_attn_mask)
            y = self.drop(y)
            x = self.norm_layers_0[i](x + y)

            y = self.encdec_attn_layers[i](x, h, encdec_attn_mask)
            y = self.drop(y)
            x = self.norm_layers_1[i](x + y)

            y = self.ffn_layers[i](x, x_mask)
            y = self.drop(y)
            x = self.norm_layers_2[i](x + y)
        x = x * x_mask
        return x


class MultiHeadAttention(nn.Module):
    """Multi-head attention module.

    Implements multi-head attention with optional relative positional encoding
    and proximal bias.
    """

    def __init__(
        self,
        channels: int,
        out_channels: int,
        n_heads: int,
        p_dropout: float = 0.0,
        window_size: Optional[int] = None,
        heads_share: bool = True,
        block_length: Optional[int] = None,
        proximal_bias: bool = False,
        proximal_init: bool = False,
    ) -> None:
        """Initialize MultiHeadAttention.

        Args:
            channels: Number of input channels.
            out_channels: Number of output channels.
            n_heads: Number of attention heads.
            p_dropout: Dropout probability.
            window_size: Window size for relative positional encoding.
            heads_share: Whether heads share relative embeddings.
            block_length: Block length for local attention.
            proximal_bias: Whether to use proximal bias.
            proximal_init: Whether to initialize keys with queries.
        """
        super().__init__()
        assert channels % n_heads == 0

        self.channels = channels
        self.out_channels = out_channels
        self.n_heads = n_heads
        self.p_dropout = p_dropout
        self.window_size = window_size
        self.heads_share = heads_share
        self.block_length = block_length
        self.proximal_bias = proximal_bias
        self.proximal_init = proximal_init
        self.attn = None

        self.k_channels = channels // n_heads
        self.conv_q = nn.Conv1d(channels, channels, 1)
        self.conv_k = nn.Conv1d(channels, channels, 1)
        self.conv_v = nn.Conv1d(channels, channels, 1)
        self.conv_o = nn.Conv1d(channels, out_channels, 1)
        self.drop = nn.Dropout(p_dropout)

        if window_size is not None:
            n_heads_rel = 1 if heads_share else n_heads
            rel_stddev = self.k_channels**-0.5
            self.emb_rel_k = nn.Parameter(
                torch.randn(n_heads_rel, window_size * 2 + 1, self.k_channels) * rel_stddev,
            )
            self.emb_rel_v = nn.Parameter(
                torch.randn(n_heads_rel, window_size * 2 + 1, self.k_channels) * rel_stddev,
            )

        nn.init.xavier_uniform_(self.conv_q.weight)
        nn.init.xavier_uniform_(self.conv_k.weight)
        nn.init.xavier_uniform_(self.conv_v.weight)
        if proximal_init:
            with torch.no_grad():
                self.conv_k.weight.copy_(self.conv_q.weight)
                self.conv_k.bias.copy_(self.conv_q.bias)

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through attention.

        Args:
            x: Query tensor of shape [batch, channels, time].
            c: Key/value tensor of shape [batch, channels, time].
            attn_mask: Optional attention mask.

        Returns:
            Attention output of shape [batch, out_channels, time].
        """
        q = self.conv_q(x)
        k = self.conv_k(c)
        v = self.conv_v(c)

        x, self.attn = self.attention(q, k, v, mask=attn_mask)

        x = self.conv_o(x)
        return x

    def attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute multi-head attention.

        Args:
            query: Query tensor of shape [batch, channels, query_time].
            key: Key tensor of shape [batch, channels, key_time].
            value: Value tensor of shape [batch, channels, key_time].
            mask: Optional attention mask.

        Returns:
            Tuple of (attention output, attention weights).
        """
        # reshape [b, d, t] -> [b, n_h, t, d_k]
        b, d, t_s, t_t = (*key.size(), query.size(2))
        query = query.view(b, self.n_heads, self.k_channels, t_t).transpose(2, 3)
        key = key.view(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)
        value = value.view(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)

        scores = torch.matmul(query / math.sqrt(self.k_channels), key.transpose(-2, -1))
        if self.window_size is not None:
            assert t_s == t_t, "Relative attention is only available for self-attention."
            key_relative_embeddings = self._get_relative_embeddings(self.emb_rel_k, t_s)
            rel_logits = self._matmul_with_relative_keys(
                query / math.sqrt(self.k_channels),
                key_relative_embeddings,
            )
            scores_local = self._relative_position_to_absolute_position(rel_logits)
            scores = scores + scores_local
        if self.proximal_bias:
            assert t_s == t_t, "Proximal bias is only available for self-attention."
            scores = scores + self._attention_bias_proximal(t_s).to(
                device=scores.device,
                dtype=scores.dtype,
            )
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e4)
            if self.block_length is not None:
                assert t_s == t_t, "Local attention is only available for self-attention."
                block_mask = (
                    torch.ones_like(scores).triu(-self.block_length).tril(self.block_length)
                )
                scores = scores.masked_fill(block_mask == 0, -1e4)
        p_attn = F.softmax(scores, dim=-1)  # [b, n_h, t_t, t_s]
        p_attn = self.drop(p_attn)
        output = torch.matmul(p_attn, value)
        if self.window_size is not None:
            relative_weights = self._absolute_position_to_relative_position(p_attn)
            value_relative_embeddings = self._get_relative_embeddings(self.emb_rel_v, t_s)
            output = output + self._matmul_with_relative_values(
                relative_weights,
                value_relative_embeddings,
            )
        output = (
            output.transpose(2, 3).contiguous().view(b, d, t_t)
        )  # [b, n_h, t_t, d_k] -> [b, d, t_t]
        return output, p_attn

    def _matmul_with_relative_values(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Matrix multiplication with relative values.

        Args:
            x: Input tensor of shape [b, h, l, m].
            y: Relative embeddings of shape [h or 1, m, d].

        Returns:
            Result tensor of shape [b, h, l, d].
        """
        ret = torch.matmul(x, y.unsqueeze(0))
        return ret

    def _matmul_with_relative_keys(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Matrix multiplication with relative keys.

        Args:
            x: Input tensor of shape [b, h, l, d].
            y: Relative embeddings of shape [h or 1, m, d].

        Returns:
            Result tensor of shape [b, h, l, m].
        """
        ret = torch.matmul(x, y.unsqueeze(0).transpose(-2, -1))
        return ret

    def _get_relative_embeddings(
        self,
        relative_embeddings: torch.Tensor,
        length: int,
    ) -> torch.Tensor:
        """Get relative embeddings for given sequence length.

        Args:
            relative_embeddings: Relative embedding parameters.
            length: Sequence length.

        Returns:
            Sliced relative embeddings for the sequence.
        """
        2 * self.window_size + 1
        # Pad first before slice to avoid using cond ops.
        pad_length = max(length - (self.window_size + 1), 0)
        slice_start_position = max((self.window_size + 1) - length, 0)
        slice_end_position = slice_start_position + 2 * length - 1
        if pad_length > 0:
            padded_relative_embeddings = F.pad(
                relative_embeddings,
                commons.convert_pad_shape([[0, 0], [pad_length, pad_length], [0, 0]]),
            )
        else:
            padded_relative_embeddings = relative_embeddings
        used_relative_embeddings = padded_relative_embeddings[
            :,
            slice_start_position:slice_end_position,
        ]
        return used_relative_embeddings

    def _relative_position_to_absolute_position(self, x: torch.Tensor) -> torch.Tensor:
        """Convert relative position to absolute position.

        Args:
            x: Relative position tensor of shape [b, h, l, 2*l-1].

        Returns:
            Absolute position tensor of shape [b, h, l, l].
        """
        batch, heads, length, _ = x.size()
        # Concat columns of pad to shift from relative to absolute indexing.
        x = F.pad(x, commons.convert_pad_shape([[0, 0], [0, 0], [0, 0], [0, 1]]))

        # Concat extra elements so to add up to shape (len+1, 2*len-1).
        x_flat = x.view([batch, heads, length * 2 * length])
        x_flat = F.pad(x_flat, commons.convert_pad_shape([[0, 0], [0, 0], [0, length - 1]]))

        # Reshape and slice out the padded elements.
        x_final = x_flat.view([batch, heads, length + 1, 2 * length - 1])[
            :,
            :,
            :length,
            length - 1 :,
        ]
        return x_final

    def _absolute_position_to_relative_position(self, x: torch.Tensor) -> torch.Tensor:
        """Convert absolute position to relative position.

        Args:
            x: Absolute position tensor of shape [b, h, l, l].

        Returns:
            Relative position tensor of shape [b, h, l, 2*l-1].
        """
        batch, heads, length, _ = x.size()
        # pad along column
        x = F.pad(x, commons.convert_pad_shape([[0, 0], [0, 0], [0, 0], [0, length - 1]]))
        x_flat = x.view([batch, heads, length**2 + length * (length - 1)])
        # add 0's in the beginning that will skew the elements after reshape
        x_flat = F.pad(x_flat, commons.convert_pad_shape([[0, 0], [0, 0], [length, 0]]))
        x_final = x_flat.view([batch, heads, length, 2 * length])[:, :, :, 1:]
        return x_final

    def _attention_bias_proximal(self, length: int) -> torch.Tensor:
        """Bias for self-attention to encourage attention to close positions.

        Args:
            length: Sequence length.

        Returns:
            Bias tensor with shape [1, 1, length, length].
        """
        r = torch.arange(length, dtype=torch.float32)
        diff = torch.unsqueeze(r, 0) - torch.unsqueeze(r, 1)
        return torch.unsqueeze(torch.unsqueeze(-torch.log1p(torch.abs(diff)), 0), 0)


class FFN(nn.Module):
    """Feed-forward network module.

    Implements a 2-layer convolutional feed-forward network with dropout.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        filter_channels: int,
        kernel_size: int,
        p_dropout: float = 0.0,
        activation: Optional[str] = None,
        causal: bool = False,
    ) -> None:
        """Initialize FFN.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            filter_channels: Number of channels in hidden layer.
            kernel_size: Kernel size for convolutions.
            p_dropout: Dropout probability.
            activation: Activation function ('gelu' or None for ReLU).
            causal: Whether to use causal padding.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.activation = activation
        self.causal = causal

        if causal:
            self.padding = self._causal_padding
        else:
            self.padding = self._same_padding

        self.conv_1 = nn.Conv1d(in_channels, filter_channels, kernel_size)
        self.conv_2 = nn.Conv1d(filter_channels, out_channels, kernel_size)
        self.drop = nn.Dropout(p_dropout)

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor) -> torch.Tensor:
        """Forward pass through FFN.

        Args:
            x: Input tensor of shape [batch, channels, time].
            x_mask: Mask tensor of shape [batch, 1, time].

        Returns:
            Output tensor of shape [batch, out_channels, time].
        """
        x = self.conv_1(self.padding(x * x_mask))
        if self.activation == "gelu":
            x = x * torch.sigmoid(1.702 * x)
        else:
            x = torch.relu(x)
        x = self.drop(x)
        x = self.conv_2(self.padding(x * x_mask))
        return x * x_mask

    def _causal_padding(self, x: torch.Tensor) -> torch.Tensor:
        """Apply causal padding to input.

        Args:
            x: Input tensor.

        Returns:
            Padded tensor.
        """
        if self.kernel_size == 1:
            return x
        pad_l = self.kernel_size - 1
        pad_r = 0
        padding = [[0, 0], [0, 0], [pad_l, pad_r]]
        x = F.pad(x, commons.convert_pad_shape(padding))
        return x

    def _same_padding(self, x: torch.Tensor) -> torch.Tensor:
        """Apply same padding to input.

        Args:
            x: Input tensor.

        Returns:
            Padded tensor.
        """
        if self.kernel_size == 1:
            return x
        pad_l = (self.kernel_size - 1) // 2
        pad_r = self.kernel_size // 2
        padding = [[0, 0], [0, 0], [pad_l, pad_r]]
        x = F.pad(x, commons.convert_pad_shape(padding))
        return x

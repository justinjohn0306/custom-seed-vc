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

"""Transformer architecture components for ASTRAL quantization."""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F


def find_multiple(n: int, k: int) -> int:
    """Find the smallest multiple of k that is >= n.

    Args:
        n: Input number.
        k: Multiple to find.

    Returns:
        Smallest multiple of k that is >= n.
    """
    if n % k == 0:
        return n
    return n + k - (n % k)


class AdaptiveLayerNorm(nn.Module):
    """Adaptive Layer Normalization.

    Applies layer normalization with learnable scale and shift
    parameters based on an embedding.

    Attributes:
        project_layer: Linear layer to project embedding to scale and shift.
        norm: Base normalization layer.
        d_model: Model dimension.
        eps: Epsilon value from norm layer.
    """

    def __init__(self, d_model: int, norm: nn.Module) -> None:
        """Initialize adaptive layer norm.

        Args:
            d_model: Model dimension.
            norm: Base normalization module.
        """
        super(AdaptiveLayerNorm, self).__init__()
        self.project_layer = nn.Linear(d_model, 2 * d_model)
        self.norm = norm
        self.d_model = d_model
        self.eps = self.norm.eps

    def forward(self, input: Tensor, embedding: Optional[Tensor] = None) -> Tensor:
        """Apply adaptive layer normalization.

        Args:
            input: Input tensor to normalize.
            embedding: Optional embedding for adaptive scaling and shifting.

        Returns:
            Normalized tensor.
        """
        if embedding is None:
            return self.norm(input)
        weight, bias = torch.split(
            self.project_layer(embedding),
            split_size_or_sections=self.d_model,
            dim=-1,
        )
        return weight * self.norm(input) + bias


@dataclass
class ModelArgs:
    """Model configuration arguments."""

    block_size: int = 2048
    vocab_size: int = 32000
    n_layer: int = 32
    n_head: int = 32
    dim: int = 4096
    intermediate_size: int = None
    n_local_heads: int = -1
    head_dim: int = 64
    rope_base: float = 10000
    norm_eps: float = 1e-5
    has_cross_attention: bool = False
    context_dim: int = 0
    is_causal: bool = False
    dropout_rate: float = 0.1
    attn_dropout_rate: float = 0.1

    def __post_init__(self):
        if self.n_local_heads == -1:
            self.n_local_heads = self.n_head
        if self.intermediate_size is None:
            hidden_dim = 4 * self.dim
            n_hidden = int(2 * hidden_dim / 3)
            self.intermediate_size = find_multiple(n_hidden, 256)
        # self.head_dim = self.dim // self.n_head


class Transformer(nn.Module):
    """Transformer model with adaptive normalization.

    Implements a transformer with optional cross-attention and RoPE.

    Attributes:
        config: Model configuration.
        layers: List of transformer blocks.
        norm: Final adaptive layer normalization.
        max_batch_size: Maximum batch size for caching.
        max_seq_length: Maximum sequence length.
        freqs_cis: Precomputed frequency tensor for RoPE.
        causal_mask: Causal attention mask.
    """

    def __init__(self, config: ModelArgs) -> None:
        """Initialize transformer.

        Args:
            config: Model configuration.
        """
        super().__init__()
        self.config = config

        self.layers = nn.ModuleList(TransformerBlock(config) for _ in range(config.n_layer))
        self.norm = AdaptiveLayerNorm(config.dim, RMSNorm(config.dim, eps=config.norm_eps))

        self.max_batch_size = -1
        self.max_seq_length = config.block_size
        freqs_cis = precompute_freqs_cis(
            self.config.block_size,
            self.config.head_dim,
            self.config.rope_base,
        )
        self.register_buffer("freqs_cis", freqs_cis)

        causal_mask = torch.tril(
            torch.ones(self.max_seq_length, self.max_seq_length, dtype=torch.bool),
        )
        self.register_buffer("causal_mask", causal_mask)

    def forward(
        self,
        x: Tensor,
        c: Tensor,
        input_pos: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        context: Optional[Tensor] = None,
        context_input_pos: Optional[Tensor] = None,
        cross_attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass through the transformer.

        Args:
            x: Input tensor.
            c: Conditioning tensor.
            input_pos: Position indices for input.
            mask: Attention mask.
            context: Context tensor for cross-attention.
            context_input_pos: Position indices for context.
            cross_attention_mask: Mask for cross-attention.

        Returns:
            Output tensor after transformer processing.
        """
        if mask is None:
            mask = self.causal_mask[: x.size(1), : x.size(1)]
        else:
            mask = mask[..., input_pos]
        freqs_cis = self.freqs_cis[input_pos]
        if context is not None:
            context_freqs_cis = self.freqs_cis[context_input_pos]
        else:
            context_freqs_cis = None
        for _i, layer in enumerate(self.layers):
            x = layer(x, c, freqs_cis, mask, context, context_freqs_cis, cross_attention_mask)
        x = self.norm(x, c)
        return x


class TransformerBlock(nn.Module):
    """Single transformer block with self-attention and feed-forward.

    Attributes:
        attention: Self-attention module.
        feed_forward: Feed-forward network.
        ffn_norm: Adaptive normalization for FFN.
        attention_norm: Adaptive normalization for attention.
        has_cross_attention: Whether block has cross-attention.
        cross_attention: Optional cross-attention module.
        cross_attention_norm: Optional cross-attention normalization.
    """

    def __init__(self, config: ModelArgs) -> None:
        """Initialize transformer block.

        Args:
            config: Model configuration.
        """
        super().__init__()
        self.attention = Attention(config)
        self.feed_forward = FeedForward(config)
        self.ffn_norm = AdaptiveLayerNorm(config.dim, RMSNorm(config.dim, eps=config.norm_eps))
        self.attention_norm = AdaptiveLayerNorm(
            config.dim,
            RMSNorm(config.dim, eps=config.norm_eps),
        )

        if config.has_cross_attention:
            self.has_cross_attention = True
            self.cross_attention = Attention(config, is_cross_attention=True)
            self.cross_attention_norm = AdaptiveLayerNorm(
                config.dim,
                RMSNorm(config.dim, eps=config.norm_eps),
            )
        else:
            self.has_cross_attention = False

    def forward(
        self,
        x: Tensor,
        c: Tensor,
        freqs_cis: Tensor,
        mask: Tensor,
        context: Optional[Tensor] = None,
        context_freqs_cis: Optional[Tensor] = None,
        cross_attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass through the transformer block.

        Args:
            x: Input tensor.
            c: Conditioning tensor.
            freqs_cis: Frequency embeddings for positional encoding.
            mask: Attention mask.
            context: Context tensor for cross-attention.
            context_freqs_cis: Frequency embeddings for context.
            cross_attention_mask: Mask for cross-attention.

        Returns:
            Output tensor after processing.
        """
        # time_attn_start = time.time()
        h = x + self.attention(self.attention_norm(x, c), freqs_cis, mask)
        # print(
        #     "time take for attention of sequence length "
        #     f"{x.shape[1]} is {time.time() - time_attn_start}"
        # )
        if self.has_cross_attention:
            h = h + self.cross_attention(
                self.cross_attention_norm(h, c),
                freqs_cis,
                cross_attention_mask,
                context,
                context_freqs_cis,
            )
        out = h + self.feed_forward(self.ffn_norm(h, c))
        return out


class Attention(nn.Module):
    """Multi-head attention with RoPE and optional cross-attention.

    Attributes:
        wqkv: Combined QKV projection (for self-attention).
        wq: Query projection (for cross-attention).
        wkv: Key-value projection (for cross-attention).
        wo: Output projection.
        kv_cache: Optional KV cache.
        n_head: Number of attention heads.
        head_dim: Dimension per head.
        n_local_heads: Number of local heads for KV.
        dim: Model dimension.
        attn_dropout_rate: Attention dropout rate.
    """

    def __init__(self, config: ModelArgs, is_cross_attention: bool = False) -> None:
        """Initialize attention module.

        Args:
            config: Model configuration.
            is_cross_attention: Whether this is cross-attention.
        """
        super().__init__()
        assert config.dim % config.n_head == 0

        total_head_dim = (config.n_head + 2 * config.n_local_heads) * config.head_dim
        # key, query, value projections for all heads, but in a batch
        if is_cross_attention:
            self.wq = nn.Linear(config.dim, config.n_head * config.head_dim, bias=False)
            self.wkv = nn.Linear(
                config.context_dim,
                2 * config.n_local_heads * config.head_dim,
                bias=False,
            )
        else:
            self.wqkv = nn.Linear(config.dim, total_head_dim, bias=False)
        self.wo = nn.Linear(config.head_dim * config.n_head, config.dim, bias=False)
        self.kv_cache = None

        self.n_head = config.n_head
        self.head_dim = config.head_dim
        self.n_local_heads = config.n_local_heads
        self.dim = config.dim
        self.attn_dropout_rate = config.attn_dropout_rate

    def forward(
        self,
        x: Tensor,
        freqs_cis: Tensor,
        mask: Tensor,
        context: Optional[Tensor] = None,
        context_freqs_cis: Optional[Tensor] = None,
    ) -> Tensor:
        """Apply multi-head attention.

        Args:
            x: Input tensor.
            freqs_cis: Frequency embeddings for positional encoding.
            mask: Attention mask.
            context: Context tensor for cross-attention.
            context_freqs_cis: Frequency embeddings for context.

        Returns:
            Output tensor after attention.
        """
        bsz, seqlen, _ = x.shape

        kv_size = self.n_local_heads * self.head_dim
        if context is None:
            q, k, v = self.wqkv(x).split([kv_size, kv_size, kv_size], dim=-1)
            context_seqlen = seqlen
        else:
            q = self.wq(x)
            k, v = self.wkv(context).split([kv_size, kv_size], dim=-1)
            context_seqlen = context.shape[1]

        q = q.view(bsz, seqlen, self.n_head, self.head_dim)
        k = k.view(bsz, context_seqlen, self.n_local_heads, self.head_dim)
        v = v.view(bsz, context_seqlen, self.n_local_heads, self.head_dim)

        q = apply_rotary_emb(q, freqs_cis)
        k = apply_rotary_emb(k, context_freqs_cis if context_freqs_cis is not None else freqs_cis)

        q, k, v = map(lambda x: x.transpose(1, 2), (q, k, v))

        k = k.repeat_interleave(self.n_head // self.n_local_heads, dim=1)
        v = v.repeat_interleave(self.n_head // self.n_local_heads, dim=1)
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            dropout_p=self.attn_dropout_rate if self.training else 0.0,
        )

        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, self.head_dim * self.n_head)

        y = self.wo(y)
        return y


class FeedForward(nn.Module):
    """Feed-forward network with SwiGLU activation.

    Attributes:
        w1: First linear layer.
        w3: Gate linear layer.
        w2: Output linear layer.
        dropout: Dropout layer.
    """

    def __init__(self, config: ModelArgs) -> None:
        """Initialize feed-forward network.

        Args:
            config: Model configuration.
        """
        super().__init__()
        self.w1 = nn.Linear(config.dim, config.intermediate_size, bias=False)
        self.w3 = nn.Linear(config.dim, config.intermediate_size, bias=False)
        self.w2 = nn.Linear(config.intermediate_size, config.dim, bias=False)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through feed-forward network.

        Args:
            x: Input tensor.

        Returns:
            Output tensor.
        """
        return self.w2(self.dropout(F.silu(self.w1(x)) * self.w3(x)))


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Attributes:
        eps: Small value to avoid division by zero.
        weight: Learnable scale parameter.
    """

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        """Initialize RMSNorm.

        Args:
            dim: Dimension to normalize.
            eps: Epsilon value.
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: Tensor) -> Tensor:
        """Apply RMS normalization.

        Args:
            x: Input tensor.

        Returns:
            Normalized tensor.
        """
        return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through RMSNorm.

        Args:
            x: Input tensor.

        Returns:
            Normalized and scaled tensor.
        """
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(
    seq_len: int,
    n_elem: int,
    base: int = 10000,
    dtype: torch.dtype = torch.bfloat16,
) -> Tensor:
    """Precompute frequency tensor for rotary positional embeddings.

    Args:
        seq_len: Maximum sequence length.
        n_elem: Number of elements (head dimension).
        base: Base for exponential decay.
        dtype: Output tensor dtype.

    Returns:
        Cached frequency tensor for RoPE.
    """
    freqs = 1.0 / (base ** (torch.arange(0, n_elem, 2)[: (n_elem // 2)].float() / n_elem))
    t = torch.arange(seq_len, device=freqs.device)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    cache = torch.stack([freqs_cis.real, freqs_cis.imag], dim=-1)
    return cache.to(dtype=dtype)


def apply_rotary_emb(x: Tensor, freqs_cis: Tensor) -> Tensor:
    """Apply rotary positional embeddings to input tensor.

    Args:
        x: Input tensor of shape (..., seq_len, n_heads, head_dim).
        freqs_cis: Precomputed frequency tensor.

    Returns:
        Tensor with rotary embeddings applied.
    """
    xshaped = x.float().reshape(*x.shape[:-1], -1, 2)
    freqs_cis = freqs_cis.view(1, xshaped.size(1), 1, xshaped.size(3), 2)
    x_out2 = torch.stack(
        [
            xshaped[..., 0] * freqs_cis[..., 0] - xshaped[..., 1] * freqs_cis[..., 1],
            xshaped[..., 1] * freqs_cis[..., 0] + xshaped[..., 0] * freqs_cis[..., 1],
        ],
        -1,
    )

    x_out2 = x_out2.flatten(3)
    return x_out2.type_as(x)

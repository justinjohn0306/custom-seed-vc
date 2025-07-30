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

"""Diffusion Transformer (DiT) model for Seed-VC v2."""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F


def find_multiple(n: int, k: int) -> int:
    """Find the nearest multiple of k that is >= n.

    Args:
        n: Input number.
        k: Multiple to find.

    Returns:
        Nearest multiple of k that is >= n.
    """
    if n % k == 0:
        return n
    return n + k - (n % k)


class AdaptiveLayerNorm(nn.Module):
    """Adaptive Layer Normalization for DiT models.

    Modulates layer normalization with learnable scale and shift parameters
    based on a conditioning embedding.
    """

    def __init__(self, d_model: int, norm: nn.Module) -> None:
        """Initialize AdaptiveLayerNorm.

        Args:
            d_model: Model dimension.
            norm: Base normalization layer.
        """
        super(AdaptiveLayerNorm, self).__init__()
        self.linear = nn.Linear(d_model, 6 * d_model)
        self.act = nn.SiLU()
        self.norm = norm
        self.d_model = d_model
        self.eps = self.norm.eps

    def forward(self, x: Tensor, emb: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Apply adaptive layer normalization.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model).
            emb: Conditioning embedding of shape (batch_size, d_model).

        Returns:
            Tuple of:
                - Normalized x with scale and shift for MSA.
                - Gate for MSA.
                - Shift for MLP.
                - Scale for MLP.
                - Gate for MLP.
        """
        emb = self.linear(self.act(emb))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = torch.chunk(emb, 6, dim=-1)

        x = self.norm(x) * (1 + scale_msa) + shift_msa
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp


class AdaptiveLayerNormFinal(nn.Module):
    """Final adaptive layer normalization for DiT models.

    Similar to AdaptiveLayerNorm but only applies scale and shift
    without gating.
    """

    def __init__(self, d_model: int, norm: nn.Module) -> None:
        """Initialize AdaptiveLayerNormFinal.

        Args:
            d_model: Model dimension.
            norm: Base normalization layer.
        """
        super(AdaptiveLayerNormFinal, self).__init__()
        self.linear = nn.Linear(d_model, 2 * d_model)
        self.act = nn.SiLU()
        self.norm = norm
        self.d_model = d_model
        self.eps = self.norm.eps

    def forward(self, x: Tensor, emb: Tensor) -> Tensor:
        """Apply final adaptive layer normalization.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model).
            emb: Conditioning embedding of shape (batch_size, d_model).

        Returns:
            Normalized tensor with adaptive scale and shift.
        """
        emb = self.linear(self.act(emb))
        scale, shift = torch.chunk(emb, 2, dim=-1)

        x = self.norm(x) * (1 + scale) + shift
        return x


@dataclass
class ModelArgs:
    """Configuration for Diffusion Transformer (DiT) models.

    Defines hyperparameters for DiT architecture including transformer
    settings, U-ViT options, and training configurations.
    """

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
    uvit_skip_connection: bool = False
    time_as_token: bool = False
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
    """Transformer model for Diffusion Transformers (DiT).

    Implements a transformer with adaptive layer normalization
    and optional U-ViT skip connections.
    """

    def __init__(self, config: ModelArgs) -> None:
        """Initialize Transformer.

        Args:
            config: Model configuration.
        """
        super().__init__()
        self.config = config

        self.layers = nn.ModuleList(TransformerBlock(config) for _ in range(config.n_layer))
        self.norm = AdaptiveLayerNormFinal(config.dim, RMSNorm(config.dim, eps=config.norm_eps))

        self.max_batch_size = -1
        self.max_seq_length = config.block_size

        self.uvit_skip_connection = self.config.uvit_skip_connection
        if self.uvit_skip_connection:
            self.layers_emit_skip = [
                i for i in range(self.config.n_layer) if i < self.config.n_layer // 2
            ]
            self.layers_receive_skip = [
                i for i in range(self.config.n_layer) if i > self.config.n_layer // 2
            ]
        else:
            self.layers_emit_skip = []
            self.layers_receive_skip = []
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
    ) -> Tensor:
        """Forward pass through the transformer.

        Args:
            x: Input tensor.
            c: Conditioning tensor.
            input_pos: Input position indices.
            mask: Attention mask.

        Returns:
            Output tensor.
        """
        mask = mask[..., input_pos]
        freqs_cis = self.freqs_cis[input_pos]
        for _i, layer in enumerate(self.layers):
            x = layer(x, c, freqs_cis, mask)
        x = self.norm(x, c)
        return x


class TransformerBlock(nn.Module):
    """Transformer block with adaptive layer normalization for DiT.

    Implements a standard transformer block with self-attention and FFN,
    enhanced with adaptive normalization for conditioning.
    """

    def __init__(self, config: ModelArgs) -> None:
        """Initialize TransformerBlock.

        Args:
            config: Model configuration.
        """
        super().__init__()
        self.attention = Attention(config)
        self.feed_forward = FeedForward(config)
        self.ffn_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.attention_norm = AdaptiveLayerNorm(
            config.dim,
            RMSNorm(config.dim, eps=config.norm_eps),
        )

    def forward(
        self,
        x: Tensor,
        c: Tensor,
        freqs_cis: Tensor,
        mask: Tensor,
    ) -> Tensor:
        """Forward pass through the transformer block.

        Args:
            x: Input tensor of shape (batch_size, seq_len, dim).
            c: Conditioning tensor of shape (batch_size, dim).
            freqs_cis: Precomputed frequency tensor for rotary embeddings.
            mask: Attention mask.

        Returns:
            Output tensor of shape (batch_size, seq_len, dim).
        """
        normed_x, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.attention_norm(x, emb=c)
        # attention
        attn_output = self.attention(normed_x, freqs_cis, mask)
        x = x + gate_msa * attn_output
        normed_x = self.ffn_norm(x) * (1 + scale_mlp) + shift_mlp
        ff_output = self.feed_forward(normed_x)
        x = x + gate_mlp * ff_output
        return x


class Attention(nn.Module):
    """Multi-head attention module for DiT models.

    Implements scaled dot-product attention with rotary position embeddings
    and optional cross-attention capabilities.
    """

    def __init__(self, config: ModelArgs, is_cross_attention: bool = False):
        """Initialize Attention.

        Args:
            config: Model configuration.
            is_cross_attention: Whether this is a cross-attention layer.
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
        """Forward pass through the attention layer.

        Args:
            x: Input tensor of shape (batch_size, seq_len, dim).
            freqs_cis: Precomputed frequency tensor for rotary embeddings.
            mask: Attention mask.
            context: Optional context tensor for cross-attention.
            context_freqs_cis: Optional frequency tensor for context.

        Returns:
            Output tensor of shape (batch_size, seq_len, dim).
        """
        bsz, seqlen, _ = x.shape

        kv_size = self.n_local_heads * self.head_dim
        q, k, v = self.wqkv(x).split([kv_size, kv_size, kv_size], dim=-1)
        context_seqlen = seqlen

        q = q.view(bsz, seqlen, self.n_head, self.head_dim)
        k = k.view(bsz, context_seqlen, self.n_local_heads, self.head_dim)
        v = v.view(bsz, context_seqlen, self.n_local_heads, self.head_dim)

        q = apply_rotary_emb(q, freqs_cis)
        k = apply_rotary_emb(k, freqs_cis)

        q, k, v = map(lambda x: x.transpose(1, 2), (q, k, v))

        k = k.repeat_interleave(self.n_head // self.n_local_heads, dim=1)
        v = v.repeat_interleave(self.n_head // self.n_local_heads, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0)

        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, self.head_dim * self.n_head)

        y = self.wo(y)
        return y


class FeedForward(nn.Module):
    """Feed-forward network with SwiGLU activation for DiT.

    Implements a two-layer MLP with SwiGLU activation function and
    dropout for regularization.
    """

    def __init__(self, config: ModelArgs) -> None:
        """Initialize FeedForward.

        Args:
            config: Model configuration.
        """
        super().__init__()
        self.w1 = nn.Linear(config.dim, config.intermediate_size, bias=False)
        self.w3 = nn.Linear(config.dim, config.intermediate_size, bias=False)
        self.w2 = nn.Linear(config.intermediate_size, config.dim, bias=False)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the feed-forward network.

        Args:
            x: Input tensor of shape (batch_size, seq_len, dim).

        Returns:
            Output tensor of shape (batch_size, seq_len, dim).
        """
        return self.w2(self.dropout(F.silu(self.w1(x)) * self.w3(x)))


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Normalizes activations by their RMS value and applies a learned
    scale parameter for each dimension.
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        """Initialize RMSNorm.

        Args:
            dim: The dimension to normalize.
            eps: Small value to prevent division by zero.
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
        """Apply RMS normalization to the input.

        Args:
            x: Input tensor to normalize.

        Returns:
            Normalized tensor with the same shape as input.
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

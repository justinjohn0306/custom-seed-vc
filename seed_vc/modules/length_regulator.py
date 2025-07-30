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

"""Length regulation and prosody modeling for Seed-VC."""

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from dac.nn.quantize import VectorQuantize
from torch.nn import functional as F

from seed_vc.modules.commons import sequence_mask

# f0_bin = 256
f0_max = 1100.0
f0_min = 50.0
f0_mel_min = 1127 * np.log(1 + f0_min / 700)
f0_mel_max = 1127 * np.log(1 + f0_max / 700)


def f0_to_coarse(f0: torch.Tensor, f0_bin: int) -> torch.Tensor:
    """Convert F0 values to coarse discrete bins.

    Args:
        f0: Fundamental frequency tensor.
        f0_bin: Number of discrete bins.

    Returns:
        Coarse F0 indices as long tensor.
    """
    f0_mel = 1127 * (1 + f0 / 700).log()
    a = (f0_bin - 2) / (f0_mel_max - f0_mel_min)
    b = f0_mel_min * a - 1.0
    f0_mel = torch.where(f0_mel > 0, f0_mel * a - b, f0_mel)
    # torch.clip_(f0_mel, min=1., max=float(f0_bin - 1))
    f0_coarse = torch.round(f0_mel).long()
    f0_coarse = f0_coarse * (f0_coarse > 0)
    f0_coarse = f0_coarse + ((f0_coarse < 1) * 1)
    f0_coarse = f0_coarse * (f0_coarse < f0_bin)
    f0_coarse = f0_coarse + ((f0_coarse >= f0_bin) * (f0_bin - 1))
    return f0_coarse


class InterpolateRegulator(nn.Module):
    """Length regulator with interpolation and optional vector quantization.

    This module handles length regulation for sequences, with support for
    discrete or continuous inputs, vector quantization, and F0 conditioning.
    """

    def __init__(
        self,
        channels: int,
        sampling_ratios: Tuple,
        is_discrete: bool = False,
        in_channels: Optional[int] = None,
        vector_quantize: bool = False,
        codebook_size: int = 1024,
        out_channels: Optional[int] = None,
        groups: int = 1,
        n_codebooks: int = 1,
        quantizer_dropout: float = 0.0,
        f0_condition: bool = False,
        n_f0_bins: int = 512,
    ) -> None:
        """Initialize InterpolateRegulator.

        Args:
            channels: Number of channels in the hidden representation.
            sampling_ratios: Tuple of sampling ratios for interpolation.
            is_discrete: Whether input is discrete tokens.
            in_channels: Number of input channels (for continuous input).
            vector_quantize: Whether to use vector quantization.
            codebook_size: Size of the codebook for discrete inputs.
            out_channels: Number of output channels.
            groups: Number of groups for group normalization.
            n_codebooks: Number of codebooks for multi-codebook quantization.
            quantizer_dropout: Dropout rate for quantizer.
            f0_condition: Whether to condition on F0.
            n_f0_bins: Number of F0 bins for conditioning.
        """
        super().__init__()
        self.sampling_ratios = sampling_ratios
        out_channels = out_channels or channels
        model = nn.ModuleList([])
        if len(sampling_ratios) > 0:
            self.interpolate = True
            for _ in sampling_ratios:
                module = nn.Conv1d(channels, channels, 3, 1, 1)
                norm = nn.GroupNorm(groups, channels)
                act = nn.Mish()
                model.extend([module, norm, act])
        else:
            self.interpolate = False
        model.append(nn.Conv1d(channels, out_channels, 1, 1))
        self.model = nn.Sequential(*model)
        self.embedding = nn.Embedding(codebook_size, channels)
        self.is_discrete = is_discrete

        self.mask_token = nn.Parameter(torch.zeros(1, channels))

        self.n_codebooks = n_codebooks
        if n_codebooks > 1:
            self.extra_codebooks = nn.ModuleList(
                [nn.Embedding(codebook_size, channels) for _ in range(n_codebooks - 1)],
            )
            self.extra_codebook_mask_tokens = nn.ParameterList(
                [nn.Parameter(torch.zeros(1, channels)) for _ in range(n_codebooks - 1)],
            )
        self.quantizer_dropout = quantizer_dropout

        if f0_condition:
            self.f0_embedding = nn.Embedding(n_f0_bins, channels)
            self.f0_condition = f0_condition
            self.n_f0_bins = n_f0_bins
            self.f0_bins = torch.arange(2, 1024, 1024 // n_f0_bins)
            self.f0_mask = nn.Parameter(torch.zeros(1, channels))
        else:
            self.f0_condition = False

        if not is_discrete:
            self.content_in_proj = nn.Linear(in_channels, channels)
            if vector_quantize:
                self.vq = VectorQuantize(channels, codebook_size, 8)

    def forward(
        self,
        x: torch.Tensor,
        ylens: Optional[torch.Tensor] = None,
        n_quantizers: Optional[int] = None,
        f0: Optional[torch.Tensor] = None,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        """Forward pass through the length regulator.

        Args:
            x: Input tensor, either discrete tokens or continuous features.
            ylens: Target sequence lengths.
            n_quantizers: Number of quantizers to use.
            f0: Optional F0 values for conditioning.

        Returns:
            Tuple of:
                - Regulated output tensor.
                - Output lengths.
                - Quantization codes (if applicable).
                - Commitment loss (if applicable).
                - Codebook loss (if applicable).
        """
        # apply token drop
        if self.training:
            n_quantizers = torch.ones((x.shape[0],)) * self.n_codebooks
            dropout = torch.randint(1, self.n_codebooks + 1, (x.shape[0],))
            n_dropout = int(x.shape[0] * self.quantizer_dropout)
            n_quantizers[:n_dropout] = dropout[:n_dropout]
            n_quantizers = n_quantizers.to(x.device)
            # decide whether to drop for each sample in batch
        else:
            n_quantizers = torch.ones((x.shape[0],), device=x.device) * (
                self.n_codebooks if n_quantizers is None else n_quantizers
            )
        if self.is_discrete:
            if self.n_codebooks > 1:
                assert len(x.size()) == 3
                x_emb = self.embedding(x[:, 0])
                for i, emb in enumerate(self.extra_codebooks):
                    x_emb = x_emb + (n_quantizers > i + 1)[..., None, None] * emb(x[:, i + 1])
                    # add mask token if not using this codebook
                    # x_emb = (
                    #     x_emb
                    #     + (n_quantizers <= i + 1)[..., None, None]
                    #     * self.extra_codebook_mask_tokens[i]
                    # )
                x = x_emb
            elif self.n_codebooks == 1:
                if len(x.size()) == 2:
                    x = self.embedding(x)
                else:
                    x = self.embedding(x[:, 0])
        else:
            x = self.content_in_proj(x)
        # x in (B, T, D)
        mask = sequence_mask(ylens).unsqueeze(-1)
        if self.interpolate:
            x = F.interpolate(x.transpose(1, 2).contiguous(), size=ylens.max(), mode="nearest")
        else:
            x = x.transpose(1, 2).contiguous()
            mask = mask[:, : x.size(2), :]
            ylens = ylens.clamp(max=x.size(2)).long()
        if self.f0_condition:
            if f0 is None:
                x = x + self.f0_mask.unsqueeze(-1)
            else:
                # quantized_f0 = torch.bucketize(f0, self.f0_bins.to(f0.device))  # (N, T)
                quantized_f0 = f0_to_coarse(f0, self.n_f0_bins)
                quantized_f0 = quantized_f0.clamp(0, self.n_f0_bins - 1).long()
                f0_emb = self.f0_embedding(quantized_f0)
                f0_emb = F.interpolate(
                    f0_emb.transpose(1, 2).contiguous(),
                    size=ylens.max(),
                    mode="nearest",
                )
                x = x + f0_emb
        out = self.model(x).transpose(1, 2).contiguous()
        if hasattr(self, "vq"):
            (
                out_q,
                commitment_loss,
                codebook_loss,
                codes,
                out,
            ) = self.vq(out.transpose(1, 2))
            out_q = out_q.transpose(1, 2)
            return out_q * mask, ylens, codes, commitment_loss, codebook_loss
        olens = ylens
        return out * mask, olens, None, None, None

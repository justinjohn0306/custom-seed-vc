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

"""Length regulation module for Seed-VC v2."""

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

from seed_vc.modules.commons import sequence_mask

# f0_bin = 256
f0_max = 1100.0
f0_min = 50.0
f0_mel_min = 1127 * np.log(1 + f0_min / 700)
f0_mel_max = 1127 * np.log(1 + f0_max / 700)


def f0_to_coarse(f0: torch.Tensor, f0_bin: int) -> torch.Tensor:
    """Convert F0 values to coarse F0 bins.

    Args:
        f0: F0 values in Hz.
        f0_bin: Number of F0 bins.

    Returns:
        Coarse F0 bin indices.
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
    """Length regulator using interpolation for sequence length adjustment.

    Handles both discrete (codebook) and continuous inputs, with optional F0 conditioning.

    Attributes:
        sampling_ratios: Ratios for upsampling.
        interpolate: Whether to perform interpolation.
        model: Sequential model for processing.
        embedding: Embedding layer for discrete inputs.
        is_discrete: Whether input is discrete tokens.
        mask_token: Learnable mask token.
        f0_condition: Whether to use F0 conditioning.
        f0_embedding: F0 embedding layer.
        n_f0_bins: Number of F0 bins.
        f0_bins: F0 bin boundaries.
        f0_mask: F0 mask token.
        content_in_proj: Input projection for continuous inputs.
    """

    def __init__(
        self,
        channels: int,
        sampling_ratios: Tuple,
        is_discrete: bool = False,
        in_channels: Optional[int] = None,  # only applies to continuous input
        codebook_size: int = 1024,  # for discrete only
        out_channels: Optional[int] = None,
        groups: int = 1,
        f0_condition: bool = False,
        n_f0_bins: int = 512,
    ) -> None:
        """Initialize interpolate regulator.

        Args:
            channels: Number of channels.
            sampling_ratios: Upsampling ratios.
            is_discrete: Whether input is discrete tokens.
            in_channels: Input channels for continuous input.
            codebook_size: Size of codebook for discrete input.
            out_channels: Output channels.
            groups: Number of groups for group normalization.
            f0_condition: Whether to use F0 conditioning.
            n_f0_bins: Number of F0 bins.
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
        model.append(
            nn.Conv1d(channels, out_channels, 1, 1) if channels != out_channels else nn.Identity(),
        )
        self.model = nn.Sequential(*model)
        self.embedding = nn.Embedding(codebook_size, channels)
        self.is_discrete = is_discrete

        self.mask_token = nn.Parameter(torch.zeros(1, channels))

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

    def forward(
        self,
        x: torch.Tensor,
        ylens: Optional[torch.Tensor] = None,
        f0: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of length regulator.

        Args:
            x: Input tensor - discrete tokens [B, T] or continuous [B, T, D].
            ylens: Target lengths [B].
            f0: F0 values for conditioning [B, T].

        Returns:
            Tuple of:
                - Regulated output [B, T', D].
                - Output lengths [B].
        """
        if self.is_discrete:
            if len(x.size()) == 2:
                x = self.embedding(x)
            else:
                x = self.embedding(x[:, 0])
        else:
            x = self.content_in_proj(x)
        # x in (B, T, D)

        if self.interpolate:
            mask = sequence_mask(ylens).unsqueeze(-1)
            x = F.interpolate(x.transpose(1, 2).contiguous(), size=ylens.max(), mode="nearest")
        else:
            x = x.transpose(1, 2).contiguous()
            mask = None
            # mask = mask[:, :x.size(2), :]
            # ylens = ylens.clamp(max=x.size(2)).long()
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
        out = out * mask if mask is not None else out
        olens = ylens
        return out, olens

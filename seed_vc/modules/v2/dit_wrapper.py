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

"""DiT wrapper module for Seed-VC v2."""

import math

import torch
from torch import nn
from torch.nn.utils import weight_norm

from seed_vc.modules.commons import sequence_mask
from seed_vc.modules.v2.dit_model import ModelArgs, Transformer


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Apply modulation to input tensor using shift and scale.

    Args:
        x: Input tensor to modulate.
        shift: Shift values.
        scale: Scale values.

    Returns:
        Modulated tensor.
    """
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################


class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into vector representations.

    Attributes:
        mlp: MLP for processing timestep embeddings.
        frequency_embedding_size: Size of frequency embeddings.
    """

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256) -> None:
        """Initialize timestep embedder.

        Args:
            hidden_size: Hidden dimension size.
            frequency_embedding_size: Size of frequency embeddings.
        """
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(
        t: torch.Tensor,
        dim: int,
        max_period: int = 10000,
        scale: int = 1000,
    ) -> torch.Tensor:
        """Create sinusoidal timestep embeddings.

        Args:
            t: 1-D Tensor of N indices, one per batch element. These may be fractional.
            dim: The dimension of the output.
            max_period: Controls the minimum frequency of the embeddings.
            scale: Scaling factor for embeddings.

        Returns:
            An (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half,
        ).to(device=t.device)
        args = scale * t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Forward pass of timestep embedder.

        Args:
            t: Timestep values.

        Returns:
            Timestep embeddings.
        """
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class DiT(torch.nn.Module):
    """Diffusion Transformer (DiT) model for voice conversion.

    Implements a transformer-based diffusion model with style and content conditioning.

    Attributes:
        time_as_token: Whether to treat time as a token.
        style_as_token: Whether to treat style as a token.
        uvit_skip_connection: Whether to use U-ViT skip connections.
        transformer: The main transformer model.
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        num_heads: Number of attention heads.
        x_embedder: Linear layer for embedding input.
        content_dim: Dimension of content features.
        cond_projection: Projection layer for content conditioning.
        t_embedder: Timestep embedder.
        final_mlp: Final MLP for output projection.
        class_dropout_prob: Probability of class dropout.
        cond_x_merge_linear: Linear layer for merging conditions.
        style_in: Linear layer for style input.
    """

    def __init__(
        self,
        time_as_token: bool,
        style_as_token: bool,
        uvit_skip_connection: bool,
        block_size: int,
        depth: int,
        num_heads: int,
        hidden_dim: int,
        in_channels: int,
        content_dim: int,
        style_encoder_dim: int,
        class_dropout_prob: float,
        dropout_rate: float,
        attn_dropout_rate: float,
    ) -> None:
        """Initialize DiT model.

        Args:
            time_as_token: Whether to treat time as a token.
            style_as_token: Whether to treat style as a token.
            uvit_skip_connection: Whether to use U-ViT skip connections.
            block_size: Maximum sequence length.
            depth: Number of transformer layers.
            num_heads: Number of attention heads.
            hidden_dim: Hidden dimension size.
            in_channels: Number of input channels.
            content_dim: Dimension of content features.
            style_encoder_dim: Dimension of style encoder output.
            class_dropout_prob: Probability of applying class dropout.
            dropout_rate: Dropout rate for MLP.
            attn_dropout_rate: Dropout rate for attention.
        """
        super(DiT, self).__init__()
        self.time_as_token = time_as_token
        self.style_as_token = style_as_token
        self.uvit_skip_connection = uvit_skip_connection
        model_args = ModelArgs(
            block_size=block_size,
            n_layer=depth,
            n_head=num_heads,
            dim=hidden_dim,
            head_dim=hidden_dim // num_heads,
            vocab_size=1,  # we don't use this
            uvit_skip_connection=self.uvit_skip_connection,
            time_as_token=self.time_as_token,
            dropout_rate=dropout_rate,
            attn_dropout_rate=attn_dropout_rate,
        )
        self.transformer = Transformer(model_args)
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.num_heads = num_heads

        self.x_embedder = weight_norm(nn.Linear(in_channels, hidden_dim, bias=True))

        self.content_dim = content_dim  # for continuous content
        self.cond_projection = nn.Linear(content_dim, hidden_dim, bias=True)  # continuous content

        self.t_embedder = TimestepEmbedder(hidden_dim)

        self.final_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, in_channels),
        )

        self.class_dropout_prob = class_dropout_prob

        self.cond_x_merge_linear = nn.Linear(hidden_dim + in_channels + in_channels, hidden_dim)
        self.style_in = nn.Linear(style_encoder_dim, hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        prompt_x: torch.Tensor,
        x_lens: torch.Tensor,
        t: torch.Tensor,
        style: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass of DiT model.

        Args:
            x: Input mel-spectrogram [B, C, T].
            prompt_x: Prompt mel-spectrogram [B, C, T].
            x_lens: Lengths of each sequence [B].
            t: Timestep values [B].
            style: Style embeddings [B, style_dim].
            cond: Content conditioning [B, T, content_dim].

        Returns:
            Predicted noise or velocity [B, C, T].
        """
        class_dropout = False
        content_dropout = False
        if self.training and torch.rand(1) < self.class_dropout_prob:
            class_dropout = True
            if self.training and torch.rand(1) < 0.5:
                content_dropout = True
        cond_in_module = self.cond_projection

        B, _, T = x.size()

        t1 = self.t_embedder(t)  # (N, D)
        cond = cond_in_module(cond)

        x = x.transpose(1, 2)
        prompt_x = prompt_x.transpose(1, 2)

        x_in = torch.cat([x, prompt_x, cond], dim=-1)
        if class_dropout:
            x_in[..., self.in_channels : self.in_channels * 2] = 0
            if content_dropout:
                x_in[..., self.in_channels * 2 :] = 0
        x_in = self.cond_x_merge_linear(x_in)  # (N, T, D)

        style = self.style_in(style)
        style = torch.zeros_like(style) if class_dropout else style
        if self.style_as_token:
            x_in = torch.cat([style.unsqueeze(1), x_in], dim=1)
        if self.time_as_token:
            x_in = torch.cat([t1.unsqueeze(1), x_in], dim=1)
        x_mask = (
            sequence_mask(
                x_lens + self.style_as_token + self.time_as_token,
                max_length=x_in.size(1),
            )
            .to(x.device)
            .unsqueeze(1)
        )
        input_pos = torch.arange(x_in.size(1)).to(x.device)
        x_mask_expanded = x_mask[:, None, :].repeat(1, 1, x_in.size(1), 1)
        x_res = self.transformer(x_in, t1.unsqueeze(1), input_pos, x_mask_expanded)
        x_res = x_res[:, 1:] if self.time_as_token else x_res
        x_res = x_res[:, 1:] if self.style_as_token else x_res
        x = self.final_mlp(x_res)
        x = x.transpose(1, 2)
        return x

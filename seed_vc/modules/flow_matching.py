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

"""Flow Matching models for Seed-VC."""

from abc import ABC
from typing import List, Optional, Tuple

import torch
from tqdm import tqdm

from seed_vc.modules.diffusion_transformer import DiT


class BASECFM(torch.nn.Module, ABC):
    """Base class for Conditional Flow Matching models."""

    def __init__(
        self,
        args,
    ) -> None:
        """Initialize the BASECFM model.

        Args:
            args: Configuration object containing model hyperparameters.
        """
        super().__init__()
        self.sigma_min = 1e-6

        self.estimator: Optional[torch.nn.Module] = None

        self.in_channels = args.DiT.in_channels

        self.criterion = torch.nn.MSELoss() if args.reg_loss_type == "l2" else torch.nn.L1Loss()

        if hasattr(args.DiT, "zero_prompt_speech_token"):
            self.zero_prompt_speech_token = args.DiT.zero_prompt_speech_token
        else:
            self.zero_prompt_speech_token = False

    @torch.inference_mode()
    def inference(
        self,
        mu: torch.Tensor,
        x_lens: torch.Tensor,
        prompt: torch.Tensor,
        style: torch.Tensor,
        f0: torch.Tensor,
        n_timesteps: int,
        temperature: float = 1.0,
        inference_cfg_rate: float = 0.5,
    ) -> torch.Tensor:
        """Forward diffusion process for generation.

        Args:
            mu: Output of encoder, shape (batch_size, n_feats, mel_timesteps).
            x_lens: Sequence lengths for masking.
            prompt: Prompt tensor for conditioning.
            style: Style embedding tensor.
            f0: Fundamental frequency tensor.
            n_timesteps: Number of diffusion steps.
            temperature: Temperature for scaling noise. Defaults to 1.0.
            inference_cfg_rate: Rate for classifier-free guidance. Defaults to 0.5.

        Returns:
            Generated mel-spectrogram of shape (batch_size, n_feats, mel_timesteps).
        """
        B, T = mu.size(0), mu.size(1)
        z = torch.randn([B, self.in_channels, T], device=mu.device) * temperature
        t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device)
        # t_span = t_span + (-1) * (torch.cos(torch.pi / 2 * t_span) - 1 + t_span)
        return self.solve_euler(z, x_lens, prompt, mu, style, f0, t_span, inference_cfg_rate)

    def solve_euler(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        prompt: torch.Tensor,
        mu: torch.Tensor,
        style: torch.Tensor,
        f0: torch.Tensor,
        t_span: torch.Tensor,
        inference_cfg_rate: float = 0.5,
    ) -> torch.Tensor:
        """Fixed euler solver for ODEs.

        Args:
            x: Random noise tensor.
            x_lens: Sequence lengths.
            prompt: Prompt tensor for conditioning.
            mu: Output of encoder, shape (batch_size, n_feats, mel_timesteps).
            style: Style embedding tensor.
            f0: Fundamental frequency tensor.
            t_span: Time steps interpolated, shape (n_timesteps + 1,).
            inference_cfg_rate: Rate for classifier-free guidance.

        Returns:
            Final generated sample after solving ODE.
        """
        t, _, _ = t_span[0], t_span[-1], t_span[1] - t_span[0]

        # I am storing this because I can later plot it by putting a debugger here
        # and saving it to a file. Or in future might add like a return_all_steps flag
        sol: List[torch.Tensor] = []
        # apply prompt
        prompt_len = prompt.size(-1)
        prompt_x = torch.zeros_like(x)
        prompt_x[..., :prompt_len] = prompt[..., :prompt_len]
        x[..., :prompt_len] = 0
        if self.zero_prompt_speech_token:
            mu[..., :prompt_len] = 0
        for step in tqdm(range(1, len(t_span))):
            dt = t_span[step] - t_span[step - 1]
            if inference_cfg_rate > 0:
                # Stack original and CFG (null) inputs for batched processing
                stacked_prompt_x = torch.cat([prompt_x, torch.zeros_like(prompt_x)], dim=0)
                stacked_style = torch.cat([style, torch.zeros_like(style)], dim=0)
                stacked_mu = torch.cat([mu, torch.zeros_like(mu)], dim=0)
                stacked_x = torch.cat([x, x], dim=0)
                stacked_t = torch.cat([t.unsqueeze(0), t.unsqueeze(0)], dim=0)

                # Perform a single forward pass for both original and CFG inputs
                stacked_dphi_dt = self.estimator(
                    stacked_x,
                    stacked_prompt_x,
                    x_lens,
                    stacked_t,
                    stacked_style,
                    stacked_mu,
                )

                # Split the output back into the original and CFG components
                dphi_dt, cfg_dphi_dt = stacked_dphi_dt.chunk(2, dim=0)

                # Apply CFG formula
                dphi_dt = (1.0 + inference_cfg_rate) * dphi_dt - inference_cfg_rate * cfg_dphi_dt
            else:
                dphi_dt = self.estimator(x, prompt_x, x_lens, t.unsqueeze(0), style, mu)

            x = x + dt * dphi_dt
            t = t + dt
            sol.append(x)
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t
            x[:, :, :prompt_len] = 0

        return sol[-1]

    def forward(
        self,
        x1: torch.Tensor,
        x_lens: torch.Tensor,
        prompt_lens: torch.Tensor,
        mu: torch.Tensor,
        style: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes conditional flow matching loss.

        Args:
            x1: Target tensor, shape (batch_size, n_feats, mel_timesteps).
            x_lens: Sequence lengths for masking.
            prompt_lens: Lengths of prompts for each batch item.
            mu: Output of encoder, shape (batch_size, n_feats, mel_timesteps).
            style: Style embedding tensor.

        Returns:
            Tuple of:
                - loss: Conditional flow matching loss.
                - y: Conditional flow, shape (batch_size, n_feats, mel_timesteps).
        """
        b, _, t = x1.shape

        # random timestep
        t = torch.rand([b, 1, 1], device=mu.device, dtype=x1.dtype)
        # sample noise p(x_0)
        z = torch.randn_like(x1)

        y = (1 - (1 - self.sigma_min) * t) * z + t * x1
        u = x1 - (1 - self.sigma_min) * z

        prompt = torch.zeros_like(x1)
        for bib in range(b):
            prompt[bib, :, : prompt_lens[bib]] = x1[bib, :, : prompt_lens[bib]]
            # range covered by prompt are set to 0
            y[bib, :, : prompt_lens[bib]] = 0
            if self.zero_prompt_speech_token:
                mu[bib, :, : prompt_lens[bib]] = 0

        estimator_out = self.estimator(
            y,
            prompt,
            x_lens,
            t.squeeze(1).squeeze(1),
            style,
            mu,
            prompt_lens,
        )
        loss = 0
        for bib in range(b):
            loss += self.criterion(
                estimator_out[bib, :, prompt_lens[bib] : x_lens[bib]],
                u[bib, :, prompt_lens[bib] : x_lens[bib]],
            )
        loss /= b

        return loss, estimator_out + (1 - self.sigma_min) * z


class CFM(BASECFM):
    """Conditional Flow Matching model implementation."""

    def __init__(self, args) -> None:
        """Initialize the CFM model.

        Args:
            args: Configuration object containing model hyperparameters.
        """
        super().__init__(args)
        if args.dit_type == "DiT":
            self.estimator = DiT(args)
        else:
            raise NotImplementedError(f"Unknown diffusion type {args.dit_type}")

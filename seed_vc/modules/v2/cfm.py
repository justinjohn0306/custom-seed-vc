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

"""Conditional Flow Matching implementation for Seed-VC v2."""

from typing import Tuple

import torch
from tqdm import tqdm


class CFM(torch.nn.Module):
    """Conditional Flow Matching model for voice conversion.

    Implements conditional flow matching for mel-spectrogram generation
    with classifier-free guidance support.
    """

    def __init__(
        self,
        estimator: torch.nn.Module,
    ) -> None:
        """Initialize the CFM model.

        Args:
            estimator (torch.nn.Module): Neural network estimator for flow matching.

        """
        super().__init__()
        self.sigma_min = 1e-6
        self.estimator = estimator
        self.in_channels = estimator.in_channels
        self.criterion = torch.nn.L1Loss()

    @torch.inference_mode()
    def inference(
        self,
        mu: torch.Tensor,
        x_lens: torch.Tensor,
        prompt: torch.Tensor,
        style: torch.Tensor,
        n_timesteps: int = 10,
        temperature: float = 1.0,
        inference_cfg_rate: Tuple[float, float] = (0.5, 0.5),
        random_voice: bool = False,
    ) -> torch.Tensor:
        """Forward diffusion.

        Args:
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            x_lens (torch.Tensor): length of each mel-spectrogram
                shape: (batch_size,)
            prompt (torch.Tensor): prompt
                shape: (batch_size, n_feats, prompt_len)
            style (torch.Tensor): style
                shape: (batch_size, style_dim)
            n_timesteps (int): number of diffusion steps
            temperature (float, optional): temperature for scaling noise. Defaults to 1.0.
            inference_cfg_rate (float, optional): Classifier-Free Guidance inference introduced in
                VoiceBox. Defaults to 0.5.
            random_voice (bool, optional): Whether to use random voice for CFG. Defaults to False.

        Returns:
            sample: generated mel-spectrogram
                shape: (batch_size, n_feats, mel_timesteps)
        """
        B, T = mu.size(0), mu.size(1)
        z = torch.randn([B, self.in_channels, T], device=mu.device) * temperature
        t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device)
        t_span = t_span + (-1) * (torch.cos(torch.pi / 2 * t_span) - 1 + t_span)
        return self.solve_euler(
            z,
            x_lens,
            prompt,
            mu,
            style,
            t_span,
            inference_cfg_rate,
            random_voice,
        )

    def solve_euler(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        prompt: torch.Tensor,
        mu: torch.Tensor,
        style: torch.Tensor,
        t_span: torch.Tensor,
        inference_cfg_rate: Tuple[float, float] = (0.5, 0.5),
        random_voice: bool = False,
    ) -> torch.Tensor:
        """Fixed euler solver for ODEs.

        Args:
            x (torch.Tensor): random noise
            t_span (torch.Tensor): n_timesteps interpolated
                shape: (n_timesteps + 1,)
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            x_lens (torch.Tensor): length of each mel-spectrogram
                shape: (batch_size,)
            prompt (torch.Tensor): prompt
                shape: (batch_size, n_feats, prompt_len)
            style (torch.Tensor): style
                shape: (batch_size, style_dim)
            inference_cfg_rate (float, optional): Classifier-Free Guidance inference introduced in
                VoiceBox. Defaults to 0.5.
            random_voice (bool, optional): Whether to use random voice for CFG. Defaults to False.

        Returns:
            x (torch.Tensor): generated tensor w/ shape: (batch_size, n_feats, mel_timesteps)
        """
        t, _, dt = t_span[0], t_span[-1], t_span[1] - t_span[0]

        # apply prompt
        prompt_len = prompt.size(-1)
        prompt_x = torch.zeros_like(x)
        prompt_x[..., :prompt_len] = prompt[..., :prompt_len]
        x[..., :prompt_len] = 0
        for step in tqdm(range(1, len(t_span))):
            if random_voice:
                cfg_dphi_dt = self.estimator(
                    torch.cat([x, x], dim=0),
                    torch.cat([torch.zeros_like(prompt_x), torch.zeros_like(prompt_x)], dim=0),
                    torch.cat([x_lens, x_lens], dim=0),
                    torch.cat([t.unsqueeze(0), t.unsqueeze(0)], dim=0),
                    torch.cat([torch.zeros_like(style), torch.zeros_like(style)], dim=0),
                    torch.cat([mu, torch.zeros_like(mu)], dim=0),
                )
                cond_txt, uncond = cfg_dphi_dt[0:1], cfg_dphi_dt[1:2]
                dphi_dt = (1.0 + inference_cfg_rate[0]) * cond_txt - inference_cfg_rate[0] * uncond
            elif all(i == 0 for i in inference_cfg_rate):
                dphi_dt = self.estimator(x, prompt_x, x_lens, t.unsqueeze(0), style, mu)
            elif inference_cfg_rate[0] == 0:
                # Classifier-Free Guidance inference introduced in VoiceBox
                cfg_dphi_dt = self.estimator(
                    torch.cat([x, x], dim=0),
                    torch.cat([prompt_x, torch.zeros_like(prompt_x)], dim=0),
                    torch.cat([x_lens, x_lens], dim=0),
                    torch.cat([t.unsqueeze(0), t.unsqueeze(0)], dim=0),
                    torch.cat([style, torch.zeros_like(style)], dim=0),
                    torch.cat([mu, mu], dim=0),
                )
                cond_txt_spk, cond_txt = cfg_dphi_dt[0:1], cfg_dphi_dt[1:2]
                dphi_dt = (1.0 + inference_cfg_rate[1]) * cond_txt_spk - inference_cfg_rate[
                    1
                ] * cond_txt
            elif inference_cfg_rate[1] == 0:
                cfg_dphi_dt = self.estimator(
                    torch.cat([x, x], dim=0),
                    torch.cat([prompt_x, torch.zeros_like(prompt_x)], dim=0),
                    torch.cat([x_lens, x_lens], dim=0),
                    torch.cat([t.unsqueeze(0), t.unsqueeze(0)], dim=0),
                    torch.cat([style, torch.zeros_like(style)], dim=0),
                    torch.cat([mu, torch.zeros_like(mu)], dim=0),
                )
                cond_txt_spk, uncond = cfg_dphi_dt[0:1], cfg_dphi_dt[1:2]
                dphi_dt = (1.0 + inference_cfg_rate[0]) * cond_txt_spk - inference_cfg_rate[
                    0
                ] * uncond
            else:
                # Multi-condition Classifier-Free Guidance inference introduced in MegaTTS3
                cfg_dphi_dt = self.estimator(
                    torch.cat([x, x, x], dim=0),
                    torch.cat(
                        [prompt_x, torch.zeros_like(prompt_x), torch.zeros_like(prompt_x)],
                        dim=0,
                    ),
                    torch.cat([x_lens, x_lens, x_lens], dim=0),
                    torch.cat([t.unsqueeze(0), t.unsqueeze(0), t.unsqueeze(0)], dim=0),
                    torch.cat([style, torch.zeros_like(style), torch.zeros_like(style)], dim=0),
                    torch.cat([mu, mu, torch.zeros_like(mu)], dim=0),
                )
                cond_txt_spk, cond_txt, uncond = (
                    cfg_dphi_dt[0:1],
                    cfg_dphi_dt[1:2],
                    cfg_dphi_dt[2:3],
                )
                dphi_dt = (
                    (1.0 + inference_cfg_rate[0] + inference_cfg_rate[1]) * cond_txt_spk
                    - inference_cfg_rate[0] * uncond
                    - inference_cfg_rate[1] * cond_txt
                )
            x = x + dt * dphi_dt
            t = t + dt
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t
            x[:, :, :prompt_len] = 0

        return x

    def forward(
        self,
        x1: torch.Tensor,
        x_lens: torch.Tensor,
        prompt_lens: torch.Tensor,
        mu: torch.Tensor,
        style: torch.Tensor,
    ) -> torch.Tensor:
        """Computes diffusion loss.

        Args:
            x1: Target mel-spectrogram of shape (batch_size, n_feats, mel_timesteps).
            x_lens: Length of each mel-spectrogram of shape (batch_size,).
            prompt_lens: Length of prompt for each sample of shape (batch_size,).
            mu: Output of encoder of shape (batch_size, n_feats, mel_timesteps).
            style: Style embedding of shape (batch_size, style_dim).

        Returns:
            Conditional flow matching loss.
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

        estimator_out = self.estimator(y, prompt, x_lens, t.squeeze(), style, mu)
        loss = 0
        for bib in range(b):
            loss += self.criterion(
                estimator_out[bib, :, prompt_lens[bib] : x_lens[bib]],
                u[bib, :, prompt_lens[bib] : x_lens[bib]],
            )
        loss /= b

        return loss

# Copyright 2025 Human Dataware Lab.
# Modified from original work by HDL members
#
# Copyright 2024 Plachtaa <https://github.com/Plachtaa>
# Modified from original work by Plachtaa
#
# Copyright (c) 2024 NVIDIA CORPORATION.
# Modified from original work by NVIDIA
#
# Original Copyright 2020 EdwardDixon <https://github.com/EdwardDixon>
# Original source: https://github.com/EdwardDixon/snake
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

"""Activation functions for BigVGAN vocoder."""

import torch
from torch import nn
from torch.nn import Parameter


class Snake(nn.Module):
    """Implementation of a sine-based periodic activation function.

    Shape:
        - Input: (B, C, T)
        - Output: (B, C, T), same shape as the input
    Parameters:
        - alpha - trainable parameter
    References:
        - This activation function is from this paper by Liu Ziyin, Tilman Hartwig, Masahito Ueda:
        https://arxiv.org/abs/2006.08195
    Examples:
        >>> a1 = snake(256)
        >>> x = torch.randn(256)
        >>> x = a1(x).
    """

    def __init__(
        self,
        in_features: int,
        alpha: float = 1.0,
        alpha_trainable: bool = True,
        alpha_logscale: bool = False,
    ) -> None:
        """Initialize Snake activation function.

        Args:
            in_features: Number of input features/channels.
            alpha: Trainable parameter controlling frequency.
                Higher values = higher-frequency. Default is 1.0.
            alpha_trainable: Whether alpha is trainable. Default is True.
            alpha_logscale: Whether to use log scale for alpha. Default is False.
        """
        super(Snake, self).__init__()
        self.in_features = in_features

        # initialize alpha
        self.alpha_logscale = alpha_logscale
        if self.alpha_logscale:  # log scale alphas initialized to zeros
            self.alpha = Parameter(torch.zeros(in_features) * alpha)
        else:  # linear scale alphas initialized to ones
            self.alpha = Parameter(torch.ones(in_features) * alpha)

        self.alpha.requires_grad = alpha_trainable

        self.no_div_by_zero = 0.000000001

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of Snake activation.

        Applies the function to the input elementwise.
        Snake ∶= x + 1/a * sin^2 (xa)

        Args:
            x: Input tensor of shape (B, C, T).

        Returns:
            Output tensor of same shape as input.
        """
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1)  # line up with x to [B, C, T]
        if self.alpha_logscale:
            alpha = torch.exp(alpha)
        x = x + (1.0 / (alpha + self.no_div_by_zero)) * torch.pow(torch.sin(x * alpha), 2)

        return x


class SnakeBeta(nn.Module):
    """SnakeBeta.

    A modified Snake function which uses separate parameters
    for the magnitude of the periodic components.

    Shape:
        - Input: (B, C, T)
        - Output: (B, C, T), same shape as the input
    Parameters:
        - alpha - trainable parameter that controls frequency
        - beta - trainable parameter that controls magnitude
    References:
        - This activation function is a modified version based on the following paper:
          https://arxiv.org/abs/2006.08195
    Examples:
        >>> a1 = snakebeta(256)
        >>> x = torch.randn(256)
        >>> x = a1(x).
    """

    def __init__(
        self,
        in_features: int,
        alpha: float = 1.0,
        alpha_trainable: bool = True,
        alpha_logscale: bool = False,
    ) -> None:
        """Initialize SnakeBeta activation function.

        Args:
            in_features: Number of input features/channels.
            alpha: Initial value for both alpha (frequency) and beta (magnitude).
                Default is 1.0.
            alpha_trainable: Whether alpha and beta are trainable. Default is True.
            alpha_logscale: Whether to use log scale for parameters. Default is False.
        """
        super(SnakeBeta, self).__init__()
        self.in_features = in_features

        # initialize alpha
        self.alpha_logscale = alpha_logscale
        if self.alpha_logscale:  # log scale alphas initialized to zeros
            self.alpha = Parameter(torch.zeros(in_features) * alpha)
            self.beta = Parameter(torch.zeros(in_features) * alpha)
        else:  # linear scale alphas initialized to ones
            self.alpha = Parameter(torch.ones(in_features) * alpha)
            self.beta = Parameter(torch.ones(in_features) * alpha)

        self.alpha.requires_grad = alpha_trainable
        self.beta.requires_grad = alpha_trainable

        self.no_div_by_zero = 0.000000001

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of SnakeBeta activation.

        Applies the function to the input elementwise.
        SnakeBeta ∶= x + 1/b * sin^2 (xa)

        Args:
            x: Input tensor of shape (B, C, T).

        Returns:
            Output tensor of same shape as input.
        """
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1)  # line up with x to [B, C, T]
        beta = self.beta.unsqueeze(0).unsqueeze(-1)
        if self.alpha_logscale:
            alpha = torch.exp(alpha)
            beta = torch.exp(beta)
        x = x + (1.0 / (beta + self.no_div_by_zero)) * torch.pow(torch.sin(x * alpha), 2)

        return x

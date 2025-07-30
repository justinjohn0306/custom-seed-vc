# Copyright (C) 2025 Human Dataware Lab.
# Modified from original work by HDL members
#
# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu, Kai Hu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""F0 predictor module for HiFi-GAN."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


class ConvRNNF0Predictor(nn.Module):
    """Convolutional RNN-based F0 predictor for pitch estimation.

    This module uses a series of convolutional layers followed by a linear classifier
    to predict F0 (fundamental frequency) values from input features.

    Attributes:
        num_class: Number of output classes (typically 1 for F0 prediction).
        condnet: Sequential convolutional network for feature extraction.
        classifier: Linear layer for final F0 prediction.
    """

    def __init__(self, num_class: int = 1, in_channels: int = 80, cond_channels: int = 512) -> None:
        """Initialize the ConvRNNF0Predictor.

        Args:
            num_class: Number of output classes for prediction. Defaults to 1.
            in_channels: Number of input channels (e.g., mel-spectrogram bins). Defaults to 80.
            cond_channels: Number of channels in the conditioning network. Defaults to 512.
        """
        super().__init__()

        self.num_class = num_class
        self.condnet = nn.Sequential(
            weight_norm(nn.Conv1d(in_channels, cond_channels, kernel_size=3, padding=1)),
            nn.ELU(),
            weight_norm(nn.Conv1d(cond_channels, cond_channels, kernel_size=3, padding=1)),
            nn.ELU(),
            weight_norm(nn.Conv1d(cond_channels, cond_channels, kernel_size=3, padding=1)),
            nn.ELU(),
            weight_norm(nn.Conv1d(cond_channels, cond_channels, kernel_size=3, padding=1)),
            nn.ELU(),
            weight_norm(nn.Conv1d(cond_channels, cond_channels, kernel_size=3, padding=1)),
            nn.ELU(),
        )
        self.classifier = nn.Linear(in_features=cond_channels, out_features=self.num_class)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the F0 predictor.

        Args:
            x: Input tensor of shape (batch_size, in_channels, time_steps).

        Returns:
            Predicted F0 values of shape (batch_size, time_steps).
            The output is absolute values to ensure non-negative F0 predictions.
        """
        x = self.condnet(x)
        x = x.transpose(1, 2)
        return torch.abs(self.classifier(x).squeeze(-1))

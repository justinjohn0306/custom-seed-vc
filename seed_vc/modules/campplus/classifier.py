# Copyright 2025 Human Dataware Lab.
# Modified from original work by HDL members
#
# Copyright 2024 Plachtaa <https://github.com/Plachtaa>
# Modified from original work by Plachtaa
#
# Original Copyright 2023 modelscope <https://github.com/modelscope>
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

"""Classifier modules for CAMPPlus speaker embedding model."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from seed_vc.modules.campplus.layers import DenseLayer


class CosineClassifier(nn.Module):
    """Cosine similarity based classifier.

    This classifier uses cosine similarity between normalized feature vectors
    and weight vectors for classification.

    Args:
        input_dim: Input feature dimension.
        num_blocks: Number of dense blocks before classification layer.
        inter_dim: Intermediate dimension for dense blocks.
        out_neurons: Number of output classes/neurons.
    """

    def __init__(
        self,
        input_dim: int,
        num_blocks: int = 0,
        inter_dim: int = 512,
        out_neurons: int = 1000,
    ) -> None:
        """Initialize CosineClassifier.

        Args:
            input_dim: Input feature dimension.
            num_blocks: Number of dense blocks before classification layer.
            inter_dim: Intermediate dimension for dense blocks.
            out_neurons: Number of output classes/neurons.
        """
        super().__init__()
        self.blocks: nn.ModuleList = nn.ModuleList()

        for _index in range(num_blocks):
            self.blocks.append(DenseLayer(input_dim, inter_dim, config_str="batchnorm"))
            input_dim = inter_dim

        self.weight: nn.Parameter = nn.Parameter(torch.FloatTensor(out_neurons, input_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the cosine classifier.

        Args:
            x: Input tensor of shape [B, dim] where B is batch size.

        Returns:
            Output tensor of shape [B, out_neurons] with cosine similarities.
        """
        # x: [B, dim]
        for layer in self.blocks:
            x = layer(x)

        # normalized
        x = F.linear(F.normalize(x), F.normalize(self.weight))
        return x


class LinearClassifier(nn.Module):
    """Standard linear classifier with optional dense blocks.

    This classifier uses linear transformation with ReLU activation
    for classification.

    Args:
        input_dim: Input feature dimension.
        num_blocks: Number of dense blocks before classification layer.
        inter_dim: Intermediate dimension for dense blocks.
        out_neurons: Number of output classes/neurons.
    """

    def __init__(
        self,
        input_dim: int,
        num_blocks: int = 0,
        inter_dim: int = 512,
        out_neurons: int = 1000,
    ) -> None:
        """Initialize LinearClassifier.

        Args:
            input_dim: Input feature dimension.
            num_blocks: Number of dense blocks before classification layer.
            inter_dim: Intermediate dimension for dense blocks.
            out_neurons: Number of output classes/neurons.
        """
        super().__init__()
        self.blocks: nn.ModuleList = nn.ModuleList()

        self.nonlinear: nn.ReLU = nn.ReLU(inplace=True)
        for _index in range(num_blocks):
            self.blocks.append(DenseLayer(input_dim, inter_dim, bias=True))
            input_dim = inter_dim

        self.linear: nn.Linear = nn.Linear(input_dim, out_neurons, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the linear classifier.

        Args:
            x: Input tensor of shape [B, dim] where B is batch size.

        Returns:
            Output tensor of shape [B, out_neurons] with logits.
        """
        # x: [B, dim]
        x = self.nonlinear(x)
        for layer in self.blocks:
            x = layer(x)
        x = self.linear(x)
        return x

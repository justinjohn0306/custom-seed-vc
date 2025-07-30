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
# limitations under the License.

"""Deep Time-Delay Dense Neural Network (DTDNN) implementation for CAMPPlus."""

from collections import OrderedDict
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from seed_vc.modules.campplus.layers import (
    BasicResBlock,
    CAMDenseTDNNBlock,
    DenseLayer,
    StatsPool,
    TDNNLayer,
    TransitLayer,
    get_nonlinear,
)


class FCM(nn.Module):
    """Feature Context Module for CAMPPlus.

    Processes input features through convolutional layers with residual blocks.

    Args:
        block: Basic building block class for the residual network.
        num_blocks: Tuple specifying number of blocks in each layer.
        m_channels: Number of channels in the convolutional layers.
        feat_dim: Input feature dimension.
    """

    def __init__(
        self,
        block: type = BasicResBlock,
        num_blocks: Tuple[int, int] = (2, 2),
        m_channels: int = 32,
        feat_dim: int = 80,
    ) -> None:
        """Initialize Feature Context Module.

        Args:
            block: Basic building block class for the residual network.
            num_blocks: Tuple specifying number of blocks in each layer.
            m_channels: Number of channels in the convolutional layers.
            feat_dim: Input feature dimension.
        """
        super(FCM, self).__init__()
        self.in_planes = m_channels
        self.conv1 = nn.Conv2d(1, m_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(m_channels)

        self.layer1 = self._make_layer(block, m_channels, num_blocks[0], stride=2)
        self.layer2 = self._make_layer(block, m_channels, num_blocks[1], stride=2)

        self.conv2 = nn.Conv2d(
            m_channels,
            m_channels,
            kernel_size=3,
            stride=(2, 1),
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(m_channels)
        self.out_channels = m_channels * (feat_dim // 8)

    def _make_layer(self, block: type, planes: int, num_blocks: int, stride: int) -> nn.Sequential:
        """Create a sequential layer with the specified number of blocks.

        Args:
            block: Block class to use.
            planes: Number of output channels.
            num_blocks: Number of blocks in the layer.
            stride: Stride for the first block.

        Returns:
            Sequential container of blocks.
        """
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through FCM.

        Args:
            x: Input tensor of shape (batch_size, feat_dim, time_steps).

        Returns:
            Output tensor of shape (batch_size, out_channels, time_steps).
        """
        x = x.unsqueeze(1)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = F.relu(self.bn2(self.conv2(out)))

        shape = out.shape
        out = out.reshape(shape[0], shape[1] * shape[2], shape[3])
        return out


class CAMPPlus(nn.Module):
    """CAMPPlus speaker encoder model.

    Context-Aware Masking and Prototype-based speaker encoder for voice conversion.

    Args:
        feat_dim: Input feature dimension.
        embedding_size: Output embedding dimension.
        growth_rate: Growth rate for DenseNet blocks.
        bn_size: Bottleneck size factor.
        init_channels: Initial number of channels after FCM.
        config_str: Configuration string for layers (e.g., "batchnorm-relu").
        memory_efficient: Whether to use memory-efficient implementation.
    """

    def __init__(
        self,
        feat_dim: int = 80,
        embedding_size: int = 512,
        growth_rate: int = 32,
        bn_size: int = 4,
        init_channels: int = 128,
        config_str: str = "batchnorm-relu",
        memory_efficient: bool = True,
    ) -> None:
        """Initialize CAMPPlus speaker encoder.

        Args:
            feat_dim: Input feature dimension.
            embedding_size: Output embedding dimension.
            growth_rate: Growth rate for DenseNet blocks.
            bn_size: Bottleneck size factor.
            init_channels: Initial number of channels after FCM.
            config_str: Configuration string for layers (e.g., "batchnorm-relu").
            memory_efficient: Whether to use memory-efficient implementation.
        """
        super(CAMPPlus, self).__init__()

        self.head = FCM(feat_dim=feat_dim)
        channels = self.head.out_channels

        self.xvector = nn.Sequential(
            OrderedDict(
                [
                    (
                        "tdnn",
                        TDNNLayer(
                            channels,
                            init_channels,
                            5,
                            stride=2,
                            dilation=1,
                            padding=-1,
                            config_str=config_str,
                        ),
                    ),
                ],
            ),
        )
        channels = init_channels
        for i, (num_layers, kernel_size, dilation) in enumerate(
            zip((12, 24, 16), (3, 3, 3), (1, 2, 2), strict=False),
        ):
            block = CAMDenseTDNNBlock(
                num_layers=num_layers,
                in_channels=channels,
                out_channels=growth_rate,
                bn_channels=bn_size * growth_rate,
                kernel_size=kernel_size,
                dilation=dilation,
                config_str=config_str,
                memory_efficient=memory_efficient,
            )
            self.xvector.add_module("block%d" % (i + 1), block)
            channels = channels + num_layers * growth_rate
            self.xvector.add_module(
                "transit%d" % (i + 1),
                TransitLayer(channels, channels // 2, bias=False, config_str=config_str),
            )
            channels //= 2

        self.xvector.add_module("out_nonlinear", get_nonlinear(config_str, channels))

        # self.xvector.add_module('stats', StatsPool())
        # self.xvector.add_module(
        #     'dense',
        #     DenseLayer(channels * 2, embedding_size, config_str='batchnorm_'))
        self.stats = StatsPool()
        self.dense = DenseLayer(channels * 2, embedding_size, config_str="batchnorm_")

        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def load_state_dict(self, state_dict: dict, strict: bool = True) -> None:
        """Load state dictionary with remapped keys.

        Custom load_state_dict that remaps keys from a previous version of the model
        where stats and dense layers were part of xvector.

        Args:
            state_dict: State dictionary to load.
            strict: Whether to strictly enforce that the keys in state_dict match the keys
                returned by this module's state_dict() function.
        """
        new_state_dict = {}

        # Remap keys for compatibility
        for key in state_dict.keys():
            new_key = key
            if key.startswith("xvector.stats"):
                new_key = key.replace("xvector.stats", "stats")
            elif key.startswith("xvector.dense"):
                new_key = key.replace("xvector.dense", "dense")
            new_state_dict[new_key] = state_dict[key]

        # Call the original load_state_dict with the modified state_dict
        super(CAMPPlus, self).load_state_dict(new_state_dict, strict)

    def forward(self, x: torch.Tensor, x_lens: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through CAMPPlus.

        Args:
            x: Input tensor of shape (batch_size, time_steps, feat_dim).
            x_lens: Optional tensor of actual lengths for each sequence in the batch.

        Returns:
            Speaker embedding tensor of shape (batch_size, embedding_size).
        """
        x = x.permute(0, 2, 1)  # (B,T,F) => (B,F,T)
        x = self.head(x)
        x = self.xvector(x)
        x = self.stats(x, x_lens)
        x = self.dense(x)
        return x

# Copyright (C) 2025 Human Dataware Lab.
# Modified from original work by HDL members
#
# Original Copyright (c) 2022 junjun3518 <https://github.com/junjun3518>
# Original source: <https://github.com/junjun3518/alias-free-torch>
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

"""Alias-free-activation implementation."""

from .act import Activation1d
from .filter import LowPassFilter1d
from .resample import DownSample1d, UpSample1d

__all__ = [
    "Activation1d",
    "LowPassFilter1d",
    "UpSample1d",
    "DownSample1d",
]

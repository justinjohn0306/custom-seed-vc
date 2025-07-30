# Copyright (C) 2025 Human Dataware Lab.
# Modified from original work by HDL members
#
# Original Copyright (C) 2020 jik846 <https://github.com/jik876>
# Original source: <https://github.com/jik876/hifi-gan>
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

"""Environment utilities for BigVGAN."""

import os
import shutil
from typing import Any


class AttrDict(dict):
    """Dictionary that supports attribute-style access.

    This allows accessing dictionary items as attributes,
    e.g., d['key'] can be accessed as d.key.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize AttrDict.

        Args:
            *args: Positional arguments passed to dict.
            **kwargs: Keyword arguments passed to dict.
        """
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def build_env(config: str, config_name: str, path: str) -> None:
    """Build environment by copying config file to target path.

    Args:
        config: Path to source config file.
        config_name: Name for the config file.
        path: Target directory path.
    """
    t_path = os.path.join(path, config_name)
    if config != t_path:
        os.makedirs(path, exist_ok=True)
        shutil.copyfile(config, os.path.join(path, config_name))

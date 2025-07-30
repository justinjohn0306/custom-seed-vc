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

"""Hugging Face Hub utilities for Seed-VC."""

import os
from pathlib import Path
from typing import Optional, Tuple, Union

from huggingface_hub import hf_hub_download


def load_custom_model_from_hf(
    repo_id: str,
    model_filename: str = "pytorch_model.bin",
    config_filename: Optional[str] = None,
) -> Union[Path, Tuple[Path, Path]]:
    """Load a custom model from Hugging Face Hub.

    Downloads model weights and optionally config file from a Hugging Face repository
    to a local cache directory.

    Args:
        repo_id: The repository ID on Hugging Face Hub (e.g., "username/model-name").
        model_filename: The filename of the model weights file. Defaults to "pytorch_model.bin".
        config_filename: Optional filename of the config file. If None, only model is returned.

    Returns:
        If config_filename is None:
            Path to the downloaded model file.
        Otherwise:
            Tuple of (model_path, config_path).

    Example:
        >>> model_path = load_custom_model_from_hf("user/my-model")
        >>> model_path, config_path = load_custom_model_from_hf(
        ...     "user/my-model",
        ...     config_filename="config.json"
        ... )
    """
    os.makedirs("./checkpoints", exist_ok=True)
    model_path = Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=model_filename,
            cache_dir="./checkpoints",
        ),
    )
    if config_filename is None:
        return model_path
    config_path = Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=config_filename,
            cache_dir="./checkpoints",
        ),
    )

    return model_path, config_path

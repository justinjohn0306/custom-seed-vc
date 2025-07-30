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

"""Utility functions for BigVGAN."""

import glob
import os
from typing import Optional, Union

import matplotlib
import torch
from torch.nn.utils import weight_norm

matplotlib.use("Agg")
import matplotlib.pylab as plt
from matplotlib.figure import Figure
from scipy.io.wavfile import write

from .meldataset import MAX_WAV_VALUE


def plot_spectrogram(spectrogram: torch.Tensor) -> Figure:
    """Plot a spectrogram.

    Args:
        spectrogram: Spectrogram tensor to plot.

    Returns:
        Matplotlib figure object.
    """
    fig, ax = plt.subplots(figsize=(10, 2))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower", interpolation="none")
    plt.colorbar(im, ax=ax)

    fig.canvas.draw()
    plt.close()

    return fig


def plot_spectrogram_clipped(spectrogram: torch.Tensor, clip_max: float = 2.0) -> Figure:
    """Plot a spectrogram with clipped values.

    Args:
        spectrogram: Spectrogram tensor to plot.
        clip_max: Maximum value for clipping. Default is 2.0.

    Returns:
        Matplotlib figure object.
    """
    fig, ax = plt.subplots(figsize=(10, 2))
    im = ax.imshow(
        spectrogram,
        aspect="auto",
        origin="lower",
        interpolation="none",
        vmin=1e-6,
        vmax=clip_max,
    )
    plt.colorbar(im, ax=ax)

    fig.canvas.draw()
    plt.close()

    return fig


def init_weights(m: torch.nn.Module, mean: float = 0.0, std: float = 0.01) -> None:
    """Initialize weights of convolutional layers.

    Args:
        m: Module to initialize.
        mean: Mean for normal distribution. Default is 0.0.
        std: Standard deviation for normal distribution. Default is 0.01.
    """
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def apply_weight_norm(m: torch.nn.Module) -> None:
    """Apply weight normalization to convolutional layers.

    Args:
        m: Module to apply weight normalization to.
    """
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        weight_norm(m)


def get_padding(kernel_size: int, dilation: int = 1) -> int:
    """Calculate padding for 'same' convolution.

    Args:
        kernel_size: Size of the convolution kernel.
        dilation: Dilation factor. Default is 1.

    Returns:
        Padding size.
    """
    return int((kernel_size * dilation - dilation) / 2)


def load_checkpoint(filepath: str, device: Union[str, torch.device]) -> dict:
    """Load checkpoint from file.

    Args:
        filepath: Path to checkpoint file.
        device: Device to load checkpoint to.

    Returns:
        Checkpoint dictionary.
    """
    assert os.path.isfile(filepath)
    print(f"Loading '{filepath}'")
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict


def save_checkpoint(filepath: str, obj: dict) -> None:
    """Save checkpoint to file.

    Args:
        filepath: Path to save checkpoint to.
        obj: Object/dictionary to save.
    """
    print(f"Saving checkpoint to {filepath}")
    torch.save(obj, filepath)
    print("Complete.")


def scan_checkpoint(cp_dir: str, prefix: str, renamed_file: Optional[str] = None) -> Optional[str]:
    """Scan for checkpoint files in directory.

    Args:
        cp_dir: Directory to scan.
        prefix: Prefix for checkpoint files.
        renamed_file: Optional renamed checkpoint file.

    Returns:
        Path to checkpoint file if found, None otherwise.
    """
    # Fallback to original scanning logic first
    pattern = os.path.join(cp_dir, prefix + "????????")
    cp_list = glob.glob(pattern)

    if len(cp_list) > 0:
        last_checkpoint_path = sorted(cp_list)[-1]
        print(f"[INFO] Resuming from checkpoint: '{last_checkpoint_path}'")
        return last_checkpoint_path

    # If no pattern-based checkpoints are found, check for renamed file
    if renamed_file:
        renamed_path = os.path.join(cp_dir, renamed_file)
        if os.path.isfile(renamed_path):
            print(f"[INFO] Resuming from renamed checkpoint: '{renamed_file}'")
            return renamed_path

    return None


def save_audio(audio: torch.Tensor, path: str, sr: int) -> None:
    """Save audio tensor to WAV file.

    Args:
        audio: Audio tensor with 1D shape.
        path: Path to save WAV file.
        sr: Sample rate.
    """
    # wav: torch with 1d shape
    audio = audio * MAX_WAV_VALUE
    audio = audio.cpu().numpy().astype("int16")
    write(path, sr, audio)

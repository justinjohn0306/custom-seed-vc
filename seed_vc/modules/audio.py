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

"""Audio processing utilities for Seed-VC."""

from typing import Dict, Tuple

import numpy as np
import torch
import torch.utils.data
from librosa.filters import mel as librosa_mel_fn
from scipy.io.wavfile import read

MAX_WAV_VALUE = 32768.0


def load_wav(full_path: str) -> Tuple[np.ndarray, int]:
    """Load a WAV file from disk.

    Args:
        full_path: Path to the WAV file.

    Returns:
        A tuple containing:
            - data: Audio waveform as numpy array.
            - sampling_rate: Sample rate in Hz.
    """
    sampling_rate, data = read(full_path)
    return data, sampling_rate


def dynamic_range_compression(x: np.ndarray, C: float = 1, clip_val: float = 1e-5) -> np.ndarray:
    """Apply dynamic range compression to input.

    Args:
        x: Input array to compress.
        C: Compression constant (default: 1).
        clip_val: Minimum value to clip to avoid log(0) (default: 1e-5).

    Returns:
        Compressed array.
    """
    return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)


def dynamic_range_decompression(x: np.ndarray, C: float = 1) -> np.ndarray:
    """Apply dynamic range decompression to input.

    Args:
        x: Input array to decompress.
        C: Compression constant (default: 1).

    Returns:
        Decompressed array.
    """
    return np.exp(x) / C


def dynamic_range_compression_torch(
    x: torch.Tensor,
    C: float = 1,
    clip_val: float = 1e-5,
) -> torch.Tensor:
    """Apply dynamic range compression to input tensor.

    Args:
        x: Input tensor to compress.
        C: Compression constant (default: 1).
        clip_val: Minimum value to clip to avoid log(0) (default: 1e-5).

    Returns:
        Compressed tensor.
    """
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x: torch.Tensor, C: float = 1) -> torch.Tensor:
    """Apply dynamic range decompression to input tensor.

    Args:
        x: Input tensor to decompress.
        C: Compression constant (default: 1).

    Returns:
        Decompressed tensor.
    """
    return torch.exp(x) / C


def spectral_normalize_torch(magnitudes: torch.Tensor) -> torch.Tensor:
    """Normalize spectral magnitudes using dynamic range compression.

    Args:
        magnitudes: Spectral magnitude tensor.

    Returns:
        Normalized spectral magnitudes.
    """
    output = dynamic_range_compression_torch(magnitudes)
    return output


def spectral_de_normalize_torch(magnitudes: torch.Tensor) -> torch.Tensor:
    """Denormalize spectral magnitudes using dynamic range decompression.

    Args:
        magnitudes: Normalized spectral magnitude tensor.

    Returns:
        Denormalized spectral magnitudes.
    """
    output = dynamic_range_decompression_torch(magnitudes)
    return output


mel_basis: Dict[str, torch.Tensor] = {}
hann_window: Dict[str, torch.Tensor] = {}


def mel_spectrogram(
    y: torch.Tensor,
    n_fft: int,
    num_mels: int,
    sampling_rate: int,
    hop_size: int,
    win_size: int,
    fmin: float,
    fmax: float,
    center: bool = False,
) -> torch.Tensor:
    """Compute mel spectrogram from waveform.

    Args:
        y: Input waveform tensor with values in range [-1, 1].
        n_fft: FFT size.
        num_mels: Number of mel frequency bins.
        sampling_rate: Sample rate in Hz.
        hop_size: Hop size in samples.
        win_size: Window size in samples.
        fmin: Minimum frequency in Hz.
        fmax: Maximum frequency in Hz.
        center: Whether to center the STFT (default: False).

    Returns:
        Normalized mel spectrogram tensor.
    """
    if torch.min(y) < -1.0:
        print("min value is ", torch.min(y))
    if torch.max(y) > 1.0:
        print("max value is ", torch.max(y))

    global mel_basis, hann_window  # pylint: disable=global-statement
    if f"{str(sampling_rate)}_{str(fmax)}_{str(y.device)}" not in mel_basis:
        mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        mel_basis[str(sampling_rate) + "_" + str(fmax) + "_" + str(y.device)] = (
            torch.from_numpy(mel).float().to(y.device)
        )
        hann_window[str(sampling_rate) + "_" + str(y.device)] = torch.hann_window(win_size).to(
            y.device,
        )

    y = torch.nn.functional.pad(
        y.unsqueeze(1),
        (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
        mode="reflect",
    )
    y = y.squeeze(1)

    spec = torch.view_as_real(
        torch.stft(
            y,
            n_fft,
            hop_length=hop_size,
            win_length=win_size,
            window=hann_window[str(sampling_rate) + "_" + str(y.device)],
            center=center,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        ),
    )

    spec = torch.sqrt(spec.pow(2).sum(-1) + (1e-9))

    spec = torch.matmul(mel_basis[str(sampling_rate) + "_" + str(fmax) + "_" + str(y.device)], spec)
    spec = spectral_normalize_torch(spec)

    return spec

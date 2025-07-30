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

"""Mel-spectrogram processing utilities for OpenVoice."""

from typing import Dict

import librosa
import torch
import torch.utils.data
from librosa.filters import mel as librosa_mel_fn

MAX_WAV_VALUE = 32768.0


def dynamic_range_compression_torch(
    x: torch.Tensor,
    C: float = 1,
    clip_val: float = 1e-5,
) -> torch.Tensor:
    """Apply dynamic range compression to the input tensor.

    Args:
        x: Input tensor.
        C: Compression factor.
        clip_val: Minimum value for clamping to avoid log(0).

    Returns:
        Compressed tensor.
    """
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x: torch.Tensor, C: float = 1) -> torch.Tensor:
    """Apply dynamic range decompression to the input tensor.

    Args:
        x: Compressed input tensor.
        C: Compression factor used during compression.

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


def spectrogram_torch(
    y: torch.Tensor,
    n_fft: int,
    sampling_rate: int,
    hop_size: int,
    win_size: int,
    center: bool = False,
) -> torch.Tensor:
    """Compute spectrogram using STFT.

    Args:
        y: Input waveform tensor.
        n_fft: FFT size.
        sampling_rate: Sample rate of the audio.
        hop_size: Hop size for STFT.
        win_size: Window size for STFT.
        center: Whether to center the STFT.

    Returns:
        Magnitude spectrogram.
    """
    # if torch.min(y) < -1.1:
    #     print("min value is ", torch.min(y))
    # if torch.max(y) > 1.1:
    #     print("max value is ", torch.max(y))

    global hann_window
    dtype_device = str(y.dtype) + "_" + str(y.device)
    wnsize_dtype_device = str(win_size) + "_" + dtype_device
    if wnsize_dtype_device not in hann_window:
        hann_window[wnsize_dtype_device] = torch.hann_window(win_size).to(
            dtype=y.dtype,
            device=y.device,
        )

    y = torch.nn.functional.pad(
        y.unsqueeze(1),
        (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
        mode="reflect",
    )
    y = y.squeeze(1)

    spec = torch.stft(
        y,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window[wnsize_dtype_device],
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=False,
    )

    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-6)
    return spec


def spectrogram_torch_conv(
    y: torch.Tensor,
    n_fft: int,
    sampling_rate: int,
    hop_size: int,
    win_size: int,
    center: bool = False,
) -> torch.Tensor:
    """Compute spectrogram using convolution-based STFT.

    This is an alternative implementation using 1D convolution that should
    produce identical results to the standard STFT.

    Args:
        y: Input waveform tensor.
        n_fft: FFT size.
        sampling_rate: Sample rate of the audio.
        hop_size: Hop size for STFT.
        win_size: Window size for STFT.
        center: Whether to center the STFT (must be False).

    Returns:
        Magnitude spectrogram.
    """
    # if torch.min(y) < -1.:
    #     print('min value is ', torch.min(y))
    # if torch.max(y) > 1.:
    #     print('max value is ', torch.max(y))

    global hann_window
    dtype_device = str(y.dtype) + "_" + str(y.device)
    wnsize_dtype_device = str(win_size) + "_" + dtype_device
    if wnsize_dtype_device not in hann_window:
        hann_window[wnsize_dtype_device] = torch.hann_window(win_size).to(
            dtype=y.dtype,
            device=y.device,
        )

    y = torch.nn.functional.pad(
        y.unsqueeze(1),
        (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
        mode="reflect",
    )

    # ******************** original ************************#
    # y = y.squeeze(1)
    # spec1 = torch.stft(
    #     y,
    #     n_fft,
    #     hop_length=hop_size,
    #     win_length=win_size,
    #     window=hann_window[wnsize_dtype_device],
    #     center=center,
    #     pad_mode="reflect",
    #     normalized=False,
    #     onesided=True,
    #     return_complex=False,
    # )

    # ******************** ConvSTFT ************************#
    freq_cutoff = n_fft // 2 + 1
    fourier_basis = torch.view_as_real(torch.fft.fft(torch.eye(n_fft)))
    forward_basis = (
        fourier_basis[:freq_cutoff].permute(2, 0, 1).reshape(-1, 1, fourier_basis.shape[1])
    )
    forward_basis = (
        forward_basis
        * torch.as_tensor(librosa.util.pad_center(torch.hann_window(win_size), size=n_fft)).float()
    )

    import torch.nn.functional as F

    # if center:
    #     signal = F.pad(
    #         y[:, None, None, :],
    #         (n_fft // 2, n_fft // 2, 0, 0),
    #         mode="reflect",
    #     ).squeeze(1)
    assert center is False

    forward_transform_squared = F.conv1d(y, forward_basis.to(y.device), stride=hop_size)
    spec2 = torch.stack(
        [
            forward_transform_squared[:, :freq_cutoff, :],
            forward_transform_squared[:, freq_cutoff:, :],
        ],
        dim=-1,
    )

    # ******************** Verification ************************#
    spec1 = torch.stft(
        y.squeeze(1),
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window[wnsize_dtype_device],
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=False,
    )
    assert torch.allclose(spec1, spec2, atol=1e-4)

    spec = torch.sqrt(spec2.pow(2).sum(-1) + 1e-6)
    return spec


def spec_to_mel_torch(
    spec: torch.Tensor,
    n_fft: int,
    num_mels: int,
    sampling_rate: int,
    fmin: float,
    fmax: float,
) -> torch.Tensor:
    """Convert spectrogram to mel-spectrogram.

    Args:
        spec: Magnitude spectrogram.
        n_fft: FFT size.
        num_mels: Number of mel frequency bins.
        sampling_rate: Sample rate of the audio.
        fmin: Minimum frequency for mel scale.
        fmax: Maximum frequency for mel scale.

    Returns:
        Mel-spectrogram.
    """
    global mel_basis
    dtype_device = str(spec.dtype) + "_" + str(spec.device)
    fmax_dtype_device = str(fmax) + "_" + dtype_device
    if fmax_dtype_device not in mel_basis:
        mel = librosa_mel_fn(sampling_rate, n_fft, num_mels, fmin, fmax)
        mel_basis[fmax_dtype_device] = torch.from_numpy(mel).to(
            dtype=spec.dtype,
            device=spec.device,
        )
    spec = torch.matmul(mel_basis[fmax_dtype_device], spec)
    spec = spectral_normalize_torch(spec)
    return spec


def mel_spectrogram_torch(
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
    """Compute mel-spectrogram from waveform.

    Args:
        y: Input waveform tensor.
        n_fft: FFT size.
        num_mels: Number of mel frequency bins.
        sampling_rate: Sample rate of the audio.
        hop_size: Hop size for STFT.
        win_size: Window size for STFT.
        fmin: Minimum frequency for mel scale.
        fmax: Maximum frequency for mel scale.
        center: Whether to center the STFT.

    Returns:
        Mel-spectrogram.
    """
    if torch.min(y) < -1.0:
        print("min value is ", torch.min(y))
    if torch.max(y) > 1.0:
        print("max value is ", torch.max(y))

    global mel_basis, hann_window
    dtype_device = str(y.dtype) + "_" + str(y.device)
    fmax_dtype_device = str(fmax) + "_" + dtype_device
    wnsize_dtype_device = str(win_size) + "_" + dtype_device
    if fmax_dtype_device not in mel_basis:
        mel = librosa_mel_fn(sampling_rate, n_fft, num_mels, fmin, fmax)
        mel_basis[fmax_dtype_device] = torch.from_numpy(mel).to(dtype=y.dtype, device=y.device)
    if wnsize_dtype_device not in hann_window:
        hann_window[wnsize_dtype_device] = torch.hann_window(win_size).to(
            dtype=y.dtype,
            device=y.device,
        )

    y = torch.nn.functional.pad(
        y.unsqueeze(1),
        (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
        mode="reflect",
    )
    y = y.squeeze(1)

    spec = torch.stft(
        y,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window[wnsize_dtype_device],
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=False,
    )

    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-6)

    spec = torch.matmul(mel_basis[fmax_dtype_device], spec)
    spec = spectral_normalize_torch(spec)

    return spec

# Copyright 2025 Human Dataware Lab.
# Modified from original work by HDL members
#
# Copyright 2024 Plachtaa <https://github.com/Plachtaa>
# Modified from original work by Plachtaa
#
# Original Copyright (c) 2024 NVIDIA CORPORATION.
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

"""Mel-spectrogram dataset utilities for BigVGAN training."""

import math
import os
import pathlib
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.utils.data
from librosa.filters import mel as librosa_mel_fn
from librosa.util import normalize
from scipy.io.wavfile import read
from tqdm import tqdm

MAX_WAV_VALUE = (
    32767.0  # NOTE: 32768.0 -1 to prevent int16 overflow (results in popping sound in corner cases)
)


def load_wav(full_path: Union[str, Path], sr_target: int) -> Tuple[np.ndarray, int]:
    """Load WAV file and validate sampling rate.

    Args:
        full_path: Path to the WAV file.
        sr_target: Target sampling rate in Hz.

    Returns:
        Tuple containing:
            - Audio data as numpy array.
            - Sampling rate of the loaded file.

    Raises:
        RuntimeError: If the sampling rate doesn't match the target.
    """
    sampling_rate, data = read(full_path)
    if sampling_rate != sr_target:
        raise RuntimeError(
            f"Sampling rate of the file {full_path} is {sampling_rate} Hz, "
            "but the model requires {sr_target} Hz",
        )
    return data, sampling_rate


def dynamic_range_compression(x: np.ndarray, C: float = 1, clip_val: float = 1e-5) -> np.ndarray:
    """Apply dynamic range compression to numpy array.

    Args:
        x: Input array.
        C: Compression constant.
        clip_val: Minimum value for clipping to avoid log(0).

    Returns:
        Compressed array.
    """
    return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)


def dynamic_range_decompression(x: np.ndarray, C: float = 1) -> np.ndarray:
    """Apply dynamic range decompression to numpy array.

    Args:
        x: Compressed input array.
        C: Compression constant.

    Returns:
        Decompressed array.
    """
    return np.exp(x) / C


def dynamic_range_compression_torch(
    x: torch.Tensor,
    C: float = 1,
    clip_val: float = 1e-5,
) -> torch.Tensor:
    """Apply dynamic range compression to torch tensor.

    Args:
        x: Input tensor.
        C: Compression constant.
        clip_val: Minimum value for clipping to avoid log(0).

    Returns:
        Compressed tensor.
    """
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x: torch.Tensor, C: float = 1) -> torch.Tensor:
    """Apply dynamic range decompression to torch tensor.

    Args:
        x: Compressed input tensor.
        C: Compression constant.

    Returns:
        Decompressed tensor.
    """
    return torch.exp(x) / C


def spectral_normalize_torch(magnitudes: torch.Tensor) -> torch.Tensor:
    """Apply spectral normalization using dynamic range compression.

    Args:
        magnitudes: Magnitude spectrogram tensor.

    Returns:
        Normalized magnitude spectrogram.
    """
    return dynamic_range_compression_torch(magnitudes)


def spectral_de_normalize_torch(magnitudes: torch.Tensor) -> torch.Tensor:
    """Apply spectral de-normalization using dynamic range decompression.

    Args:
        magnitudes: Normalized magnitude spectrogram tensor.

    Returns:
        De-normalized magnitude spectrogram.
    """
    return dynamic_range_decompression_torch(magnitudes)


mel_basis_cache: Dict[str, torch.Tensor] = {}
hann_window_cache: Dict[str, torch.Tensor] = {}


def mel_spectrogram(
    y: torch.Tensor,
    n_fft: int,
    num_mels: int,
    sampling_rate: int,
    hop_size: int,
    win_size: int,
    fmin: int,
    fmax: Optional[int] = None,
    center: bool = False,
) -> torch.Tensor:
    """Calculate the mel spectrogram of an input signal.

    This function uses slaney norm for the librosa mel filterbank (using librosa.filters.mel)
    and uses Hann window for STFT (using torch.stft).

    Args:
        y (torch.Tensor): Input signal.
        n_fft (int): FFT size.
        num_mels (int): Number of mel bins.
        sampling_rate (int): Sampling rate of the input signal.
        hop_size (int): Hop size for STFT.
        win_size (int): Window size for STFT.
        fmin (int): Minimum frequency for mel filterbank.
        fmax (int): Maximum frequency for mel filterbank.
            If None, defaults to half the sampling rate (fmax = sr / 2.0) inside librosa_mel_fn.
        center (bool): Whether to pad the input to center the frames. Default is False.

    Returns:
        torch.Tensor: Mel spectrogram.
    """
    if torch.min(y) < -1.0:
        print(f"[WARNING] Min value of input waveform signal is {torch.min(y)}")
    if torch.max(y) > 1.0:
        print(f"[WARNING] Max value of input waveform signal is {torch.max(y)}")

    device = y.device
    key = f"{n_fft}_{num_mels}_{sampling_rate}_{hop_size}_{win_size}_{fmin}_{fmax}_{device}"

    if key not in mel_basis_cache:
        mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        mel_basis_cache[key] = torch.from_numpy(mel).float().to(device)
        hann_window_cache[key] = torch.hann_window(win_size).to(device)

    mel_basis = mel_basis_cache[key]
    hann_window = hann_window_cache[key]

    padding = (n_fft - hop_size) // 2
    y = torch.nn.functional.pad(y.unsqueeze(1), (padding, padding), mode="reflect").squeeze(1)

    spec = torch.stft(
        y,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window,
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )
    spec = torch.sqrt(torch.view_as_real(spec).pow(2).sum(-1) + 1e-9)

    mel_spec = torch.matmul(mel_basis, spec)
    mel_spec = spectral_normalize_torch(mel_spec)

    return mel_spec


def get_mel_spectrogram(wav: torch.Tensor, h: Any) -> torch.Tensor:
    """Generate mel spectrogram from a waveform using given hyperparameters.

    Args:
        wav: Input waveform tensor.
        h: Hyperparameters object with attributes n_fft, num_mels, sampling_rate,
           hop_size, win_size, fmin, fmax.

    Returns:
        Mel spectrogram tensor.
    """
    return mel_spectrogram(
        wav,
        h.n_fft,
        h.num_mels,
        h.sampling_rate,
        h.hop_size,
        h.win_size,
        h.fmin,
        h.fmax,
    )


def get_dataset_filelist(a: Any) -> Tuple[List[str], List[str], List[List[str]]]:
    """Get lists of audio files for training and validation.

    Args:
        a: Arguments object with attributes:
            - input_training_file: Path to training file list.
            - input_validation_file: Path to validation file list.
            - input_wavs_dir: Directory containing WAV files.
            - list_input_unseen_validation_file: List of unseen validation file lists.
            - list_input_unseen_wavs_dir: List of directories for unseen validation files.

    Returns:
        Tuple containing:
            - List of training file paths.
            - List of validation file paths.
            - List of lists of unseen validation file paths.
    """
    training_files = []
    validation_files = []
    list_unseen_validation_files = []

    with open(a.input_training_file, "r", encoding="utf-8") as fi:
        training_files = [
            os.path.join(a.input_wavs_dir, x.split("|")[0] + ".wav")
            for x in fi.read().split("\n")
            if len(x) > 0
        ]
        print(f"first training file: {training_files[0]}")

    with open(a.input_validation_file, "r", encoding="utf-8") as fi:
        validation_files = [
            os.path.join(a.input_wavs_dir, x.split("|")[0] + ".wav")
            for x in fi.read().split("\n")
            if len(x) > 0
        ]
        print(f"first validation file: {validation_files[0]}")

    for i in range(len(a.list_input_unseen_validation_file)):
        with open(a.list_input_unseen_validation_file[i], "r", encoding="utf-8") as fi:
            unseen_validation_files = [
                os.path.join(a.list_input_unseen_wavs_dir[i], x.split("|")[0] + ".wav")
                for x in fi.read().split("\n")
                if len(x) > 0
            ]
            print(f"first unseen {i}th validation fileset: {unseen_validation_files[0]}")
            list_unseen_validation_files.append(unseen_validation_files)

    return training_files, validation_files, list_unseen_validation_files


class MelDataset(torch.utils.data.Dataset):
    """Dataset for loading audio files and computing mel spectrograms.

    This dataset loads audio files, optionally segments them, and computes
    mel spectrograms for training or validation of audio models.
    """

    def __init__(
        self,
        training_files: List[str],
        hparams: Any,
        segment_size: int,
        n_fft: int,
        num_mels: int,
        hop_size: int,
        win_size: int,
        sampling_rate: int,
        fmin: int,
        fmax: int,
        split: bool = True,
        shuffle: bool = True,
        n_cache_reuse: int = 1,
        device: Optional[torch.device] = None,
        fmax_loss: Optional[int] = None,
        fine_tuning: bool = False,
        base_mels_path: Optional[Union[str, Path]] = None,
        is_seen: bool = True,
    ) -> None:
        """Initialize MelDataset.

        Args:
            training_files: List of audio file paths.
            hparams: Hyperparameters object.
            segment_size: Size of audio segments in samples.
            n_fft: FFT size for spectrogram computation.
            num_mels: Number of mel frequency bins.
            hop_size: Hop size for STFT.
            win_size: Window size for STFT.
            sampling_rate: Target sampling rate in Hz.
            fmin: Minimum frequency for mel filterbank.
            fmax: Maximum frequency for mel filterbank.
            split: Whether to split audio into segments (True for training).
            shuffle: Whether to shuffle the file list.
            n_cache_reuse: Number of times to reuse cached audio.
            device: Device to use for computation.
            fmax_loss: Maximum frequency for loss computation mel spectrogram.
            fine_tuning: Whether in fine-tuning mode (loads pre-computed mels).
            base_mels_path: Path to pre-computed mel spectrograms (for fine-tuning).
            is_seen: Whether the dataset contains seen speakers.
        """
        self.audio_files: List[str] = training_files
        random.seed(1234)
        if shuffle:
            random.shuffle(self.audio_files)
        self.hparams: Any = hparams
        self.is_seen: bool = is_seen
        if self.is_seen:
            self.name: str = pathlib.Path(self.audio_files[0]).parts[0]
        else:
            self.name: str = "-".join(pathlib.Path(self.audio_files[0]).parts[:2]).strip("/")

        self.segment_size: int = segment_size
        self.sampling_rate: int = sampling_rate
        self.split: bool = split
        self.n_fft: int = n_fft
        self.num_mels: int = num_mels
        self.hop_size: int = hop_size
        self.win_size: int = win_size
        self.fmin: int = fmin
        self.fmax: int = fmax
        self.fmax_loss: Optional[int] = fmax_loss
        self.cached_wav: Optional[np.ndarray] = None
        self.n_cache_reuse: int = n_cache_reuse
        self._cache_ref_count: int = 0
        self.device: Optional[torch.device] = device
        self.fine_tuning: bool = fine_tuning
        self.base_mels_path: Optional[Union[str, Path]] = base_mels_path

        print("[INFO] checking dataset integrity...")
        for i in tqdm(range(len(self.audio_files))):
            assert os.path.exists(self.audio_files[i]), f"{self.audio_files[i]} not found"

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, str, torch.Tensor]:
        """Get a single item from the dataset.

        Args:
            index: Index of the item to retrieve.

        Returns:
            Tuple containing:
                - Mel spectrogram tensor (num_mels, time).
                - Audio waveform tensor (time,).
                - Filename string.
                - Mel spectrogram for loss computation (num_mels, time).
        """
        filename = self.audio_files[index]
        if self._cache_ref_count == 0:
            audio, sampling_rate = load_wav(filename, self.sampling_rate)
            audio = audio / MAX_WAV_VALUE
            if not self.fine_tuning:
                audio = normalize(audio) * 0.95
            self.cached_wav = audio
            if sampling_rate != self.sampling_rate:
                raise ValueError(f"{sampling_rate} SR doesn't match target {self.sampling_rate} SR")
            self._cache_ref_count = self.n_cache_reuse
        else:
            audio = self.cached_wav
            self._cache_ref_count -= 1

        audio = torch.FloatTensor(audio)
        audio = audio.unsqueeze(0)

        if not self.fine_tuning:
            if self.split:
                if audio.size(1) >= self.segment_size:
                    max_audio_start = audio.size(1) - self.segment_size
                    audio_start = random.randint(0, max_audio_start)
                    audio = audio[:, audio_start : audio_start + self.segment_size]
                else:
                    audio = torch.nn.functional.pad(
                        audio,
                        (0, self.segment_size - audio.size(1)),
                        "constant",
                    )

                mel = mel_spectrogram(
                    audio,
                    self.n_fft,
                    self.num_mels,
                    self.sampling_rate,
                    self.hop_size,
                    self.win_size,
                    self.fmin,
                    self.fmax,
                    center=False,
                )
            else:  # Validation step
                # Match audio length to self.hop_size * n for evaluation
                if (audio.size(1) % self.hop_size) != 0:
                    audio = audio[:, : -(audio.size(1) % self.hop_size)]
                mel = mel_spectrogram(
                    audio,
                    self.n_fft,
                    self.num_mels,
                    self.sampling_rate,
                    self.hop_size,
                    self.win_size,
                    self.fmin,
                    self.fmax,
                    center=False,
                )
                assert audio.shape[1] == mel.shape[2] * self.hop_size, (
                    f"audio shape {audio.shape} mel shape {mel.shape}"
                )

        else:
            mel = np.load(
                os.path.join(
                    self.base_mels_path,
                    os.path.splitext(os.path.split(filename)[-1])[0] + ".npy",
                ),
            )
            mel = torch.from_numpy(mel)

            if len(mel.shape) < 3:
                mel = mel.unsqueeze(0)

            if self.split:
                frames_per_seg = math.ceil(self.segment_size / self.hop_size)

                if audio.size(1) >= self.segment_size:
                    mel_start = random.randint(0, mel.size(2) - frames_per_seg - 1)
                    mel = mel[:, :, mel_start : mel_start + frames_per_seg]
                    audio = audio[
                        :,
                        mel_start * self.hop_size : (mel_start + frames_per_seg) * self.hop_size,
                    ]
                else:
                    mel = torch.nn.functional.pad(
                        mel,
                        (0, frames_per_seg - mel.size(2)),
                        "constant",
                    )
                    audio = torch.nn.functional.pad(
                        audio,
                        (0, self.segment_size - audio.size(1)),
                        "constant",
                    )

        mel_loss = mel_spectrogram(
            audio,
            self.n_fft,
            self.num_mels,
            self.sampling_rate,
            self.hop_size,
            self.win_size,
            self.fmin,
            self.fmax_loss,
            center=False,
        )

        return (mel.squeeze(), audio.squeeze(0), filename, mel_loss.squeeze())

    def __len__(self) -> int:
        """Get the length of the dataset.

        Returns:
            Number of audio files in the dataset.
        """
        return len(self.audio_files)

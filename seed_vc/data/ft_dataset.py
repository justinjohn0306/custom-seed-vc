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

"""Fine-tuning dataset module for Seed-VC."""

import os
import random
from typing import Dict, List, Tuple

import librosa
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from seed_vc.modules.audio import mel_spectrogram

duration_setting: Dict[str, float] = {
    "min": 1.0,
    "max": 30.0,
}


# assume single speaker
def to_mel_fn(wave: torch.Tensor, mel_fn_args: Dict[str, any]) -> torch.Tensor:
    """Convert waveform to mel spectrogram.

    Args:
        wave: Input waveform tensor.
        mel_fn_args: Arguments for mel spectrogram computation.

    Returns:
        Mel spectrogram tensor.
    """
    return mel_spectrogram(wave, **mel_fn_args)


class FT_Dataset(Dataset):
    """Fine-tuning dataset for audio processing.

    Loads audio files from a directory and converts them to mel spectrograms
    for training. Supports various audio formats.

    Attributes:
        data_path: Path to the directory containing audio files.
        data: List of audio file paths.
        sr: Sample rate for audio loading.
        mel_fn_args: Arguments for mel spectrogram computation.
    """

    def __init__(
        self,
        data_path: str,
        spect_params: Dict[str, any],
        sr: int = 22050,
        batch_size: int = 1,
    ) -> None:
        """Initialize FT_Dataset.

        Args:
            data_path: Path to directory containing audio files.
            spect_params: Dictionary of spectrogram parameters.
            sr: Target sample rate. Defaults to 22050.
            batch_size: Minimum batch size (dataset will be padded if smaller).

        Raises:
            AssertionError: If no audio files are found in data_path.
        """
        self.data_path = data_path
        self.data: List[str] = []
        for root, _, files in os.walk(data_path):
            for file in files:
                if file.endswith((".wav", ".mp3", ".flac", ".ogg", ".m4a", ".opus")):
                    self.data.append(os.path.join(root, file))

        self.sr = sr
        self.mel_fn_args: Dict[str, any] = {
            "n_fft": spect_params["n_fft"],
            "win_size": spect_params.get("win_length", spect_params.get("win_size", 1024)),
            "hop_size": spect_params.get("hop_length", spect_params.get("hop_size", 256)),
            "num_mels": spect_params.get("n_mels", spect_params.get("num_mels", 80)),
            "sampling_rate": sr,
            "fmin": spect_params["fmin"],
            "fmax": None if spect_params["fmax"] == "None" else spect_params["fmax"],
            "center": False,
        }

        assert len(self.data) != 0
        while len(self.data) < batch_size:
            self.data += self.data

    def __len__(self) -> int:
        """Get dataset length.

        Returns:
            Number of audio files in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a single item from the dataset.

        Args:
            idx: Index of the item to retrieve.

        Returns:
            Tuple of (waveform, mel_spectrogram) tensors.
        """
        idx = idx % len(self.data)
        wav_path = self.data[idx]
        try:
            speech, orig_sr = librosa.load(wav_path, sr=self.sr)
        except Exception as e:
            print(f"Failed to load wav file with error {e}")
            return self.__getitem__(random.randint(0, len(self)))
        if (
            len(speech) < self.sr * duration_setting["min"]
            or len(speech) > self.sr * duration_setting["max"]
        ):
            print(f"Audio {wav_path} is too short or too long, skipping")
            return self.__getitem__(random.randint(0, len(self)))
        if orig_sr != self.sr:
            speech = librosa.resample(speech, orig_sr, self.sr)

        wave = torch.from_numpy(speech).float().unsqueeze(0)
        mel = to_mel_fn(wave, self.mel_fn_args).squeeze(0)

        return wave.squeeze(0), mel


def build_ft_dataloader(
    data_path: str,
    spect_params: Dict[str, any],
    sr: int,
    batch_size: int = 1,
    num_workers: int = 0,
) -> DataLoader:
    """Build a DataLoader for fine-tuning.

    Args:
        data_path: Path to directory containing audio files.
        spect_params: Dictionary of spectrogram parameters.
        sr: Target sample rate.
        batch_size: Batch size for DataLoader. Defaults to 1.
        num_workers: Number of worker processes. Defaults to 0.

    Returns:
        DataLoader instance for the dataset.
    """
    dataset = FT_Dataset(data_path, spect_params, sr, batch_size)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate,
    )
    return dataloader


def collate(
    batch: List[Tuple[torch.Tensor, torch.Tensor]],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collate function for batching audio data.

    Pads waveforms and mel spectrograms to the same length within a batch.
    Sorts batch by mel spectrogram length in descending order.

    Args:
        batch: List of (waveform, mel_spectrogram) tuples.

    Returns:
        Tuple of:
            - waves: Padded waveforms tensor of shape (batch_size, max_wave_length).
            - mels: Padded mel spectrograms tensor of shape (batch_size, n_mels, max_mel_length).
            - wave_lengths: Original waveform lengths tensor.
            - mel_lengths: Original mel spectrogram lengths tensor.
    """
    batch_size = len(batch)

    # sort by mel length
    lengths = [b[1].shape[1] for b in batch]
    batch_indexes = np.argsort(lengths)[::-1]
    batch = [batch[bid] for bid in batch_indexes]

    nmels = batch[0][1].size(0)
    max_mel_length = max([b[1].shape[1] for b in batch])
    max_wave_length = max([b[0].size(0) for b in batch])

    mels = torch.zeros((batch_size, nmels, max_mel_length)).float() - 10
    waves = torch.zeros((batch_size, max_wave_length)).float()

    mel_lengths = torch.zeros(batch_size).long()
    wave_lengths = torch.zeros(batch_size).long()

    for bid, (wave, mel) in enumerate(batch):
        mel_size = mel.size(1)
        mels[bid, :, :mel_size] = mel
        waves[bid, : wave.size(0)] = wave
        mel_lengths[bid] = mel_size
        wave_lengths[bid] = wave.size(0)

    return waves, mels, wave_lengths, mel_lengths


if __name__ == "__main__":
    data_path = "./example/reference"
    sr = 22050
    spect_params: Dict[str, any] = {
        "n_fft": 1024,
        "win_length": 1024,
        "hop_length": 256,
        "n_mels": 80,
        "fmin": 0,
        "fmax": 8000,
    }
    dataloader = build_ft_dataloader(data_path, spect_params, sr, batch_size=2, num_workers=0)
    for idx, batch in enumerate(dataloader):
        wave, mel, wave_lengths, mel_lengths = batch
        print(wave.shape, mel.shape)
        if idx == 10:
            break

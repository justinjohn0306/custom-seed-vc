"""Test audio processing utilities."""

import numpy as np
import torch

from seed_vc.modules.audio import mel_spectrogram


class TestAudioProcessing:
    """Test cases for audio processing functions."""

    def test_mel_spectrogram_shape(self):
        """Test mel spectrogram output shape."""
        # Create dummy audio
        sample_rate = 22050
        duration = 1.0  # 1 second
        n_samples = int(sample_rate * duration)
        audio = torch.randn(1, n_samples)  # batch_size=1

        # Compute mel spectrogram
        mel = mel_spectrogram(
            audio,
            n_fft=1024,
            num_mels=80,
            sampling_rate=sample_rate,
            hop_size=256,
            win_size=1024,
            fmin=0,
            fmax=8000,
        )

        # Check shape
        assert mel.dim() == 3  # [batch, n_mels, time]
        assert mel.shape[0] == 1  # batch size
        assert mel.shape[1] == 80  # n_mels
        assert mel.shape[2] > 0  # time dimension

    def test_mel_spectrogram_values(self):
        """Test mel spectrogram value ranges."""
        # Create sine wave
        sample_rate = 22050
        duration = 0.5
        t = torch.linspace(0, duration, int(sample_rate * duration))
        frequency = 440.0  # A4
        audio = torch.sin(2 * np.pi * frequency * t).unsqueeze(0)

        # Compute mel spectrogram
        mel = mel_spectrogram(
            audio,
            n_fft=1024,
            num_mels=80,
            sampling_rate=sample_rate,
            hop_size=256,
            win_size=1024,
            fmin=0,
            fmax=8000,
        )

        # Check that we have non-zero values
        assert mel.max() > 0
        # Check no NaN or Inf values
        assert not torch.isnan(mel).any()
        assert not torch.isinf(mel).any()

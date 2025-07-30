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

"""RMVPE implementation for Seed-VC."""

import logging
import os
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from librosa.filters import mel
from librosa.util import pad_center
from scipy.signal import get_window

logger = logging.getLogger(__name__)


class STFT(torch.nn.Module):
    """STFT implementation using 1D convolution and transpose convolutions.

    This module implements an STFT using 1D convolution and 1D transpose convolutions.
    Currently optimized for hop lengths that are half the filter length (50% overlap).
    """

    def __init__(
        self,
        filter_length: int = 1024,
        hop_length: int = 512,
        win_length: Optional[int] = None,
        window: str = "hann",
    ) -> None:
        """Initialize STFT module.

        Args:
            filter_length: Length of filters used. Defaults to 1024.
            hop_length: Hop length of STFT. Defaults to 512.
            win_length: Length of window function. Defaults to filter_length.
            window: Window type (bartlett, hann, hamming, blackman, blackmanharris).
        """
        super(STFT, self).__init__()
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length if win_length else filter_length
        self.window = window
        self.forward_transform = None
        self.pad_amount = int(self.filter_length / 2)
        fourier_basis = np.fft.fft(np.eye(self.filter_length))

        cutoff = int((self.filter_length / 2 + 1))
        fourier_basis = np.vstack(
            [np.real(fourier_basis[:cutoff, :]), np.imag(fourier_basis[:cutoff, :])],
        )
        forward_basis = torch.FloatTensor(fourier_basis)
        inverse_basis = torch.FloatTensor(np.linalg.pinv(fourier_basis))

        assert filter_length >= self.win_length
        # get window and zero center pad it to filter_length
        fft_window = get_window(window, self.win_length, fftbins=True)
        fft_window = pad_center(fft_window, size=filter_length)
        fft_window = torch.from_numpy(fft_window).float()

        # window the bases
        forward_basis *= fft_window
        inverse_basis = (inverse_basis.T * fft_window).T

        self.register_buffer("forward_basis", forward_basis.float())
        self.register_buffer("inverse_basis", inverse_basis.float())
        self.register_buffer("fft_window", fft_window.float())

    def transform(
        self,
        input_data: torch.Tensor,
        return_phase: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Transform input audio to STFT domain.

        Args:
            input_data: Audio tensor with shape (num_batch, num_samples).
            return_phase: Whether to return phase information.

        Returns:
            If return_phase is False: Magnitude tensor of shape
                (num_batch, num_frequencies, num_frames).
            If return_phase is True: Tuple of (magnitude, phase) tensors.
        """
        input_data = F.pad(
            input_data,
            (self.pad_amount, self.pad_amount),
            mode="reflect",
        )
        forward_transform = input_data.unfold(1, self.filter_length, self.hop_length).permute(
            0,
            2,
            1,
        )
        forward_transform = torch.matmul(self.forward_basis, forward_transform)
        cutoff = int((self.filter_length / 2) + 1)
        real_part = forward_transform[:, :cutoff, :]
        imag_part = forward_transform[:, cutoff:, :]
        magnitude = torch.sqrt(real_part**2 + imag_part**2)
        if return_phase:
            phase = torch.atan2(imag_part.data, real_part.data)
            return magnitude, phase
        else:
            return magnitude

    def inverse(self, magnitude: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        """Compute inverse STFT (iSTFT).

        Args:
            magnitude: Magnitude tensor with shape (num_batch, num_frequencies, num_frames).
            phase: Phase tensor with shape (num_batch, num_frequencies, num_frames).

        Returns:
            Reconstructed audio tensor of shape (num_batch, num_samples).
        """
        cat = torch.cat([magnitude * torch.cos(phase), magnitude * torch.sin(phase)], dim=1)
        fold = torch.nn.Fold(
            output_size=(1, (cat.size(-1) - 1) * self.hop_length + self.filter_length),
            kernel_size=(1, self.filter_length),
            stride=(1, self.hop_length),
        )
        inverse_transform = torch.matmul(self.inverse_basis, cat)
        inverse_transform = fold(inverse_transform)[:, 0, 0, self.pad_amount : -self.pad_amount]
        window_square_sum = self.fft_window.pow(2).repeat(cat.size(-1), 1).T.unsqueeze(0)
        window_square_sum = fold(window_square_sum)[:, 0, 0, self.pad_amount : -self.pad_amount]
        inverse_transform /= window_square_sum
        return inverse_transform

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        """Transform audio to STFT domain and back.

        Args:
            input_data: Audio tensor with shape (num_batch, num_samples).

        Returns:
            Reconstructed audio tensor of shape (num_batch, num_samples).
        """
        self.magnitude, self.phase = self.transform(input_data, return_phase=True)
        reconstruction = self.inverse(self.magnitude, self.phase)
        return reconstruction


class BiGRU(nn.Module):
    """Bidirectional GRU layer."""

    def __init__(self, input_features: int, hidden_features: int, num_layers: int) -> None:
        """Initialize BiGRU layer.

        Args:
            input_features: Number of input features.
            hidden_features: Number of hidden features.
            num_layers: Number of GRU layers.
        """
        super(BiGRU, self).__init__()
        self.gru = nn.GRU(
            input_features,
            hidden_features,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through BiGRU.

        Args:
            x: Input tensor.

        Returns:
            Output tensor from GRU.
        """
        return self.gru(x)[0]


class ConvBlockRes(nn.Module):
    """Convolutional block with residual connection."""

    def __init__(self, in_channels: int, out_channels: int, momentum: float = 0.01) -> None:
        """Initialize convolutional block with residual connection.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            momentum: Momentum for batch normalization.
        """
        super(ConvBlockRes, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
                bias=False,
            ),
            nn.BatchNorm2d(out_channels, momentum=momentum),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
                bias=False,
            ),
            nn.BatchNorm2d(out_channels, momentum=momentum),
            nn.ReLU(),
        )
        # self.shortcut:Optional[nn.Module] = None
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, (1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through residual conv block.

        Args:
            x: Input tensor.

        Returns:
            Output tensor with residual connection.
        """
        if not hasattr(self, "shortcut"):
            return self.conv(x) + x
        else:
            return self.conv(x) + self.shortcut(x)


class Encoder(nn.Module):
    """Encoder module with multiple residual encoder blocks."""

    def __init__(
        self,
        in_channels: int,
        in_size: int,
        n_encoders: int,
        kernel_size: Tuple[int, int],
        n_blocks: int,
        out_channels: int = 16,
        momentum: float = 0.01,
    ) -> None:
        """Initialize encoder module.

        Args:
            in_channels: Number of input channels.
            in_size: Input size.
            n_encoders: Number of encoder layers.
            kernel_size: Kernel size for pooling.
            n_blocks: Number of residual blocks per encoder.
            out_channels: Initial number of output channels.
            momentum: Momentum for batch normalization.
        """
        super(Encoder, self).__init__()
        self.n_encoders = n_encoders
        self.bn = nn.BatchNorm2d(in_channels, momentum=momentum)
        self.layers = nn.ModuleList()
        self.latent_channels = []
        for _i in range(self.n_encoders):
            self.layers.append(
                ResEncoderBlock(
                    in_channels, out_channels, kernel_size, n_blocks, momentum=momentum
                ),
            )
            self.latent_channels.append([out_channels, in_size])
            in_channels = out_channels
            out_channels *= 2
            in_size //= 2
        self.out_size = in_size
        self.out_channel = out_channels

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Forward pass through encoder.

        Args:
            x: Input tensor.

        Returns:
            Tuple of encoded output and intermediate tensors for skip connections.
        """
        concat_tensors: List[torch.Tensor] = []
        x = self.bn(x)
        for _i, layer in enumerate(self.layers):
            t, x = layer(x)
            concat_tensors.append(t)
        return x, concat_tensors


class ResEncoderBlock(nn.Module):
    """Residual encoder block with optional pooling."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Optional[Tuple[int, int]],
        n_blocks: int = 1,
        momentum: float = 0.01,
    ) -> None:
        """Initialize residual encoder block.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Kernel size for pooling (None to skip pooling).
            n_blocks: Number of residual blocks.
            momentum: Momentum for batch normalization.
        """
        super(ResEncoderBlock, self).__init__()
        self.n_blocks = n_blocks
        self.conv = nn.ModuleList()
        self.conv.append(ConvBlockRes(in_channels, out_channels, momentum))
        for _i in range(n_blocks - 1):
            self.conv.append(ConvBlockRes(out_channels, out_channels, momentum))
        self.kernel_size = kernel_size
        if self.kernel_size is not None:
            self.pool = nn.AvgPool2d(kernel_size=kernel_size)

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass through residual encoder block.

        Args:
            x: Input tensor.

        Returns:
            If kernel_size is None: Output tensor.
            Otherwise: Tuple of (output, pooled_output).
        """
        for _i, conv in enumerate(self.conv):
            x = conv(x)
        if self.kernel_size is not None:
            return x, self.pool(x)
        else:
            return x


class Intermediate(nn.Module):
    """Intermediate processing module between encoder and decoder."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_inters: int,
        n_blocks: int,
        momentum: float = 0.01,
    ) -> None:
        """Initialize intermediate processing module.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            n_inters: Number of intermediate layers.
            n_blocks: Number of residual blocks per layer.
            momentum: Momentum for batch normalization.
        """
        super(Intermediate, self).__init__()
        self.n_inters = n_inters
        self.layers = nn.ModuleList()
        self.layers.append(ResEncoderBlock(in_channels, out_channels, None, n_blocks, momentum))
        for _i in range(self.n_inters - 1):
            self.layers.append(
                ResEncoderBlock(out_channels, out_channels, None, n_blocks, momentum),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through intermediate layers.

        Args:
            x: Input tensor.

        Returns:
            Processed tensor.
        """
        for _i, layer in enumerate(self.layers):
            x = layer(x)
        return x


class ResDecoderBlock(nn.Module):
    """Residual decoder block with transposed convolution."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: Tuple[int, int],
        n_blocks: int = 1,
        momentum: float = 0.01,
    ) -> None:
        """Initialize residual decoder block.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            stride: Stride for transposed convolution.
            n_blocks: Number of residual blocks.
            momentum: Momentum for batch normalization.
        """
        super(ResDecoderBlock, self).__init__()
        out_padding = (0, 1) if stride == (1, 2) else (1, 1)
        self.n_blocks = n_blocks
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(3, 3),
                stride=stride,
                padding=(1, 1),
                output_padding=out_padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels, momentum=momentum),
            nn.ReLU(),
        )
        self.conv2 = nn.ModuleList()
        self.conv2.append(ConvBlockRes(out_channels * 2, out_channels, momentum))
        for _i in range(n_blocks - 1):
            self.conv2.append(ConvBlockRes(out_channels, out_channels, momentum))

    def forward(self, x: torch.Tensor, concat_tensor: torch.Tensor) -> torch.Tensor:
        """Forward pass through decoder block.

        Args:
            x: Input tensor.
            concat_tensor: Tensor from encoder for skip connection.

        Returns:
            Decoded output tensor.
        """
        x = self.conv1(x)
        x = torch.cat((x, concat_tensor), dim=1)
        for _i, conv2 in enumerate(self.conv2):
            x = conv2(x)
        return x


class Decoder(nn.Module):
    """Decoder with multiple residual decoder blocks."""

    def __init__(
        self,
        in_channels: int,
        n_decoders: int,
        stride: Tuple[int, int],
        n_blocks: int,
        momentum: float = 0.01,
    ) -> None:
        """Initialize decoder module.

        Args:
            in_channels: Number of input channels.
            n_decoders: Number of decoder layers.
            stride: Stride for transposed convolutions.
            n_blocks: Number of residual blocks per decoder.
            momentum: Momentum for batch normalization.
        """
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList()
        self.n_decoders = n_decoders
        for _i in range(self.n_decoders):
            out_channels = in_channels // 2
            self.layers.append(
                ResDecoderBlock(in_channels, out_channels, stride, n_blocks, momentum),
            )
            in_channels = out_channels

    def forward(self, x: torch.Tensor, concat_tensors: List[torch.Tensor]) -> torch.Tensor:
        """Forward pass through decoder.

        Args:
            x: Input tensor.
            concat_tensors: List of tensors from encoder for skip connections.

        Returns:
            Decoded output tensor.
        """
        for i, layer in enumerate(self.layers):
            x = layer(x, concat_tensors[-1 - i])
        return x


class DeepUnet(nn.Module):
    """Deep U-Net architecture for feature extraction."""

    def __init__(
        self,
        kernel_size: Tuple[int, int],
        n_blocks: int,
        en_de_layers: int = 5,
        inter_layers: int = 4,
        in_channels: int = 1,
        en_out_channels: int = 16,
    ) -> None:
        """Initialize Deep U-Net architecture.

        Args:
            kernel_size: Kernel size for pooling.
            n_blocks: Number of residual blocks.
            en_de_layers: Number of encoder/decoder layers.
            inter_layers: Number of intermediate layers.
            in_channels: Number of input channels.
            en_out_channels: Initial encoder output channels.
        """
        super(DeepUnet, self).__init__()
        self.encoder = Encoder(
            in_channels,
            128,
            en_de_layers,
            kernel_size,
            n_blocks,
            en_out_channels,
        )
        self.intermediate = Intermediate(
            self.encoder.out_channel // 2,
            self.encoder.out_channel,
            inter_layers,
            n_blocks,
        )
        self.decoder = Decoder(self.encoder.out_channel, en_de_layers, kernel_size, n_blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through Deep U-Net.

        Args:
            x: Input tensor.

        Returns:
            Output tensor after encoding, intermediate processing, and decoding.
        """
        x, concat_tensors = self.encoder(x)
        x = self.intermediate(x)
        x = self.decoder(x, concat_tensors)
        return x


class E2E(nn.Module):
    """End-to-end model for pitch estimation."""

    def __init__(
        self,
        n_blocks: int,
        n_gru: int,
        kernel_size: Tuple[int, int],
        en_de_layers: int = 5,
        inter_layers: int = 4,
        in_channels: int = 1,
        en_out_channels: int = 16,
    ) -> None:
        """Initialize end-to-end model for pitch estimation.

        Args:
            n_blocks: Number of residual blocks.
            n_gru: Number of GRU layers.
            kernel_size: Kernel size for pooling.
            en_de_layers: Number of encoder/decoder layers.
            inter_layers: Number of intermediate layers.
            in_channels: Number of input channels.
            en_out_channels: Initial encoder output channels.
        """
        super(E2E, self).__init__()
        self.unet = DeepUnet(
            kernel_size,
            n_blocks,
            en_de_layers,
            inter_layers,
            in_channels,
            en_out_channels,
        )
        self.cnn = nn.Conv2d(en_out_channels, 3, (3, 3), padding=(1, 1))
        if n_gru:
            self.fc = nn.Sequential(
                BiGRU(3 * 128, 256, n_gru),
                nn.Linear(512, 360),
                nn.Dropout(0.25),
                nn.Sigmoid(),
            )
        else:
            self.fc = nn.Sequential(
                nn.Linear(3 * nn.N_MELS, nn.N_CLASS),
                nn.Dropout(0.25),
                nn.Sigmoid(),
            )

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """Forward pass through E2E model.

        Args:
            mel: Input mel-spectrogram.

        Returns:
            Pitch probabilities.
        """
        # print(mel.shape)
        mel = mel.transpose(-1, -2).unsqueeze(1)
        x = self.cnn(self.unet(mel)).transpose(1, 2).flatten(-2)
        x = self.fc(x)
        # print(x.shape)
        return x


class MelSpectrogram(torch.nn.Module):
    """Mel-spectrogram extraction module."""

    def __init__(
        self,
        is_half: bool,
        n_mel_channels: int,
        sampling_rate: int,
        win_length: int,
        hop_length: int,
        n_fft: Optional[int] = None,
        mel_fmin: int = 0,
        mel_fmax: Optional[int] = None,
        clamp: float = 1e-5,
    ) -> None:
        """Initialize mel-spectrogram extraction module.

        Args:
            is_half: Whether to use half precision.
            n_mel_channels: Number of mel channels.
            sampling_rate: Audio sampling rate.
            win_length: Window length for STFT.
            hop_length: Hop length for STFT.
            n_fft: FFT size (defaults to win_length).
            mel_fmin: Minimum frequency for mel scale.
            mel_fmax: Maximum frequency for mel scale.
            clamp: Minimum value for log clamp.
        """
        super().__init__()
        n_fft = win_length if n_fft is None else n_fft
        self.hann_window = {}
        mel_basis = mel(
            sr=sampling_rate,
            n_fft=n_fft,
            n_mels=n_mel_channels,
            fmin=mel_fmin,
            fmax=mel_fmax,
            htk=True,
        )
        mel_basis = torch.from_numpy(mel_basis).float()
        self.register_buffer("mel_basis", mel_basis)
        self.n_fft = win_length if n_fft is None else n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.sampling_rate = sampling_rate
        self.n_mel_channels = n_mel_channels
        self.clamp = clamp
        self.is_half = is_half

    def forward(
        self,
        audio: torch.Tensor,
        keyshift: int = 0,
        speed: float = 1,
        center: bool = True,
    ) -> torch.Tensor:
        """Extract mel-spectrogram from audio.

        Args:
            audio: Input audio tensor.
            keyshift: Pitch shift in semitones.
            speed: Speed factor for time stretching.
            center: Whether to center the STFT.

        Returns:
            Log mel-spectrogram tensor.
        """
        factor = 2 ** (keyshift / 12)
        n_fft_new = int(np.round(self.n_fft * factor))
        win_length_new = int(np.round(self.win_length * factor))
        hop_length_new = int(np.round(self.hop_length * speed))
        keyshift_key = str(keyshift) + "_" + str(audio.device)
        if keyshift_key not in self.hann_window:
            self.hann_window[keyshift_key] = torch.hann_window(win_length_new).to(audio.device)
        if "privateuseone" in str(audio.device):
            if not hasattr(self, "stft"):
                self.stft = STFT(
                    filter_length=n_fft_new,
                    hop_length=hop_length_new,
                    win_length=win_length_new,
                    window="hann",
                ).to(audio.device)
            magnitude = self.stft.transform(audio)
        else:
            fft = torch.stft(
                audio,
                n_fft=n_fft_new,
                hop_length=hop_length_new,
                win_length=win_length_new,
                window=self.hann_window[keyshift_key],
                center=center,
                return_complex=True,
            )
            magnitude = torch.sqrt(fft.real.pow(2) + fft.imag.pow(2))
        if keyshift != 0:
            size = self.n_fft // 2 + 1
            resize = magnitude.size(1)
            if resize < size:
                magnitude = F.pad(magnitude, (0, 0, 0, size - resize))
            magnitude = magnitude[:, :size, :] * self.win_length / win_length_new
        mel_output = torch.matmul(self.mel_basis, magnitude)
        if self.is_half:
            mel_output = mel_output.half()
        log_mel_spec = torch.log(torch.clamp(mel_output, min=self.clamp))
        return log_mel_spec


class RMVPE:
    """RMVPE pitch estimation model."""

    def __init__(
        self,
        model_path: str,
        is_half: bool,
        device: Optional[Union[str, torch.device]] = None,
        use_jit: bool = False,
    ) -> None:
        """Initialize RMVPE pitch estimation model.

        Args:
            model_path: Path to the model checkpoint.
            is_half: Whether to use half precision.
            device: Device to run the model on.
            use_jit: Whether to use JIT compilation (unused).
        """
        self.resample_kernel = {}
        self.resample_kernel = {}
        self.is_half = is_half
        if device is None:
            # device = "cuda:0" if torch.cuda.is_available() else "cpu"
            if torch.cuda.is_available():
                device = "cuda:0"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device
        self.mel_extractor = MelSpectrogram(is_half, 128, 16000, 1024, 160, None, 30, 8000).to(
            device,
        )
        if "privateuseone" in str(device):
            import onnxruntime as ort

            ort_session = ort.InferenceSession(
                "%s/rmvpe.onnx" % os.environ["rmvpe_root"],
                providers=["DmlExecutionProvider"],
            )
            self.model = ort_session
        else:
            if str(self.device) == "cuda":
                self.device = torch.device("cuda:0")

            def get_default_model():
                model = E2E(4, 1, (2, 2))
                ckpt = torch.load(model_path, map_location="cpu")
                model.load_state_dict(ckpt)
                model.eval()
                if is_half:
                    model = model.half()
                else:
                    model = model.float()
                return model

            self.model = get_default_model()

            self.model = self.model.to(device)
        cents_mapping = 20 * np.arange(360) + 1997.3794084376191
        self.cents_mapping = np.pad(cents_mapping, (4, 4))  # 368

    def mel2hidden(self, mel: torch.Tensor) -> Union[torch.Tensor, np.ndarray]:
        """Convert mel-spectrogram to hidden representation.

        Args:
            mel: Input mel-spectrogram.

        Returns:
            Hidden representation tensor or array.
        """
        with torch.no_grad():
            n_frames = mel.shape[-1]
            n_pad = 32 * ((n_frames - 1) // 32 + 1) - n_frames
            if n_pad > 0:
                mel = F.pad(mel, (0, n_pad), mode="constant")
            if "privateuseone" in str(self.device):
                onnx_input_name = self.model.get_inputs()[0].name
                onnx_outputs_names = self.model.get_outputs()[0].name
                hidden = self.model.run(
                    [onnx_outputs_names],
                    input_feed={onnx_input_name: mel.cpu().numpy()},
                )[0]
            else:
                mel = mel.half() if self.is_half else mel.float()
                hidden = self.model(mel)
            return hidden[:, :n_frames]

    def decode(self, hidden: np.ndarray, thred: float = 0.03) -> np.ndarray:
        """Decode hidden representation to F0 values.

        Args:
            hidden: Hidden representation from model.
            thred: Threshold for pitch detection.

        Returns:
            F0 values in Hz.
        """
        cents_pred = self.to_local_average_cents(hidden, thred=thred)
        f0 = 10 * (2 ** (cents_pred / 1200))
        f0[f0 == 10] = 0
        # f0 = np.array(
        #     [10 * (2 ** (cent_pred / 1200)) if cent_pred else 0 for cent_pred in cents_pred]
        # )
        return f0

    def infer_from_audio(
        self,
        audio: Union[torch.Tensor, np.ndarray],
        thred: float = 0.03,
    ) -> np.ndarray:
        """Infer F0 from audio.

        Args:
            audio: Input audio tensor or array.
            thred: Threshold for pitch detection.

        Returns:
            F0 values in Hz.
        """
        # torch.cuda.synchronize()
        # t0 = ttime()
        if not torch.is_tensor(audio):
            audio = torch.from_numpy(audio)
        mel = self.mel_extractor(audio.float().to(self.device).unsqueeze(0), center=True)
        # print(123123123,mel.device.type)
        # torch.cuda.synchronize()
        # t1 = ttime()
        hidden = self.mel2hidden(mel)
        # torch.cuda.synchronize()
        # t2 = ttime()
        # print(234234,hidden.device.type)
        if "privateuseone" not in str(self.device):
            hidden = hidden.squeeze(0).cpu().numpy()
        else:
            hidden = hidden[0]
        if self.is_half:
            hidden = hidden.astype("float32")

        f0 = self.decode(hidden, thred=thred)
        # torch.cuda.synchronize()
        # t3 = ttime()
        # print("hmvpe:%s\t%s\t%s\t%s"%(t1-t0,t2-t1,t3-t2,t3-t0))
        return f0

    def infer_from_audio_batch(
        self,
        audio: Union[torch.Tensor, np.ndarray],
        thred: float = 0.03,
    ) -> torch.Tensor:
        """Infer F0 from batch of audio.

        Args:
            audio: Batch of audio tensors or arrays.
            thred: Threshold for pitch detection.

        Returns:
            Batch of F0 values as tensor.
        """
        # torch.cuda.synchronize()
        # t0 = ttime()
        if not torch.is_tensor(audio):
            audio = torch.from_numpy(audio)
        mel = self.mel_extractor(audio.float().to(self.device), center=True)
        # print(123123123,mel.device.type)
        # torch.cuda.synchronize()
        # t1 = ttime()
        hidden = self.mel2hidden(mel)
        # torch.cuda.synchronize()
        # t2 = ttime()
        # print(234234,hidden.device.type)
        if "privateuseone" not in str(self.device):
            hidden = hidden.cpu().numpy()
        else:
            pass
        if self.is_half:
            hidden = hidden.astype("float32")

        f0s = []
        for bib in range(hidden.shape[0]):
            f0s.append(self.decode(hidden[bib], thred=thred))
        f0s = np.stack(f0s)
        f0s = torch.from_numpy(f0s).to(self.device)
        # torch.cuda.synchronize()
        # t3 = ttime()
        # print("hmvpe:%s\t%s\t%s\t%s"%(t1-t0,t2-t1,t3-t2,t3-t0))
        return f0s

    def to_local_average_cents(self, salience: np.ndarray, thred: float = 0.05) -> np.ndarray:
        """Convert salience to cents using local weighted average.

        Args:
            salience: Salience matrix from model.
            thred: Threshold for valid pitch detection.

        Returns:
            Pitch values in cents.
        """
        # t0 = ttime()
        center = np.argmax(salience, axis=1)  # 帧长#index
        salience = np.pad(salience, ((0, 0), (4, 4)))  # 帧长,368
        # t1 = ttime()
        center += 4
        todo_salience = []
        todo_cents_mapping = []
        starts = center - 4
        ends = center + 5
        for idx in range(salience.shape[0]):
            todo_salience.append(salience[:, starts[idx] : ends[idx]][idx])
            todo_cents_mapping.append(self.cents_mapping[starts[idx] : ends[idx]])
        # t2 = ttime()
        todo_salience = np.array(todo_salience)  # 帧长，9
        todo_cents_mapping = np.array(todo_cents_mapping)  # 帧长，9
        product_sum = np.sum(todo_salience * todo_cents_mapping, 1)
        weight_sum = np.sum(todo_salience, 1)  # 帧长
        devided = product_sum / weight_sum  # 帧长
        # t3 = ttime()
        maxx = np.max(salience, axis=1)  # 帧长
        devided[maxx <= thred] = 0
        # t4 = ttime()
        # print("decode:%s\t%s\t%s\t%s" % (t1 - t0, t2 - t1, t3 - t2, t4 - t3))
        return devided

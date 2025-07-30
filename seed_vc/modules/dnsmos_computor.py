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

"""DNS MOS (Deep Noise Suppression Mean Opinion Score) computer module."""

# ignore all warning
import warnings
from typing import Dict, List, Tuple, Union

import librosa
import numpy as np
import onnxruntime as ort
import torch
import torchaudio

warnings.filterwarnings("ignore")

SAMPLING_RATE: int = 16000
INPUT_LENGTH: float = 9.01


class DNSMOSComputer:
    """DNS MOS computer for evaluating audio quality.

    This class computes Deep Noise Suppression Mean Opinion Score (DNS MOS) metrics
    for audio signals, including signal quality, background noise, and overall quality scores.

    Attributes:
        onnx_sess: ONNX runtime session for primary model inference.
        p808_onnx_sess: ONNX runtime session for P.808 model inference.
        mel_transform: Mel spectrogram transform for audio feature extraction.
    """

    def __init__(
        self,
        primary_model_path: str,
        p808_model_path: str,
        device: str = "cuda",
        device_id: int = 0,
    ) -> None:
        """Initialize DNS MOS computer with ONNX models.

        Args:
            primary_model_path: Path to the primary DNS MOS ONNX model.
            p808_model_path: Path to the P.808 ONNX model.
            device: Device to use for computation. Defaults to "cuda".
            device_id: CUDA device ID to use. Defaults to 0.
        """
        self.onnx_sess = ort.InferenceSession(
            primary_model_path,
            providers=["CUDAExecutionProvider"],
        )
        self.p808_onnx_sess = ort.InferenceSession(
            p808_model_path,
            providers=["CUDAExecutionProvider"],
        )
        self.onnx_sess.set_providers(["CUDAExecutionProvider"], [{"device_id": device_id}])
        self.p808_onnx_sess.set_providers(["CUDAExecutionProvider"], [{"device_id": device_id}])
        kwargs = {
            "sample_rate": 16000,
            "hop_length": 160,
            "n_fft": 320 + 1,
            "n_mels": 120,
            "mel_scale": "slaney",
        }
        self.mel_transform = torchaudio.transforms.MelSpectrogram(**kwargs).to(f"cuda:{device_id}")

    def audio_melspec(
        self,
        audio: np.ndarray,
        n_mels: int = 120,
        frame_size: int = 320,
        hop_length: int = 160,
        sr: int = 16000,
        to_db: bool = True,
    ) -> np.ndarray:
        """Convert audio to mel spectrogram.

        Args:
            audio: Input audio array.
            n_mels: Number of mel bands. Defaults to 120.
            frame_size: FFT frame size. Defaults to 320.
            hop_length: Hop length for STFT. Defaults to 160.
            sr: Sample rate. Defaults to 16000.
            to_db: Whether to convert to dB scale. Defaults to True.

        Returns:
            Transposed mel spectrogram array.
        """
        mel_specgram = self.mel_transform(torch.Tensor(audio).cuda())
        mel_spec = mel_specgram.cpu()
        if to_db:
            mel_spec = (librosa.power_to_db(mel_spec, ref=np.max) + 40) / 40
        return mel_spec.T

    def get_polyfit_val(
        self,
        sig: float,
        bak: float,
        ovr: float,
        is_personalized_MOS: bool,
    ) -> Tuple[float, float, float]:
        """Apply polynomial fitting to raw MOS scores.

        Args:
            sig: Raw signal quality score.
            bak: Raw background noise score.
            ovr: Raw overall quality score.
            is_personalized_MOS: Whether to use personalized MOS coefficients.

        Returns:
            Tuple of polynomial-fitted scores (sig_poly, bak_poly, ovr_poly).
        """
        if is_personalized_MOS:
            p_ovr = np.poly1d([-0.00533021, 0.005101, 1.18058466, -0.11236046])
            p_sig = np.poly1d([-0.01019296, 0.02751166, 1.19576786, -0.24348726])
            p_bak = np.poly1d([-0.04976499, 0.44276479, -0.1644611, 0.96883132])
        else:
            p_ovr = np.poly1d([-0.06766283, 1.11546468, 0.04602535])
            p_sig = np.poly1d([-0.08397278, 1.22083953, 0.0052439])
            p_bak = np.poly1d([-0.13166888, 1.60915514, -0.39604546])
        sig_poly = p_sig(sig)
        bak_poly = p_bak(bak)
        ovr_poly = p_ovr(ovr)
        return sig_poly, bak_poly, ovr_poly

    def compute(
        self,
        audio: Union[str, np.ndarray],
        sampling_rate: int,
        is_personalized_MOS: bool = False,
    ) -> Dict[str, Union[str, float, int]]:
        """Compute DNS MOS scores for audio.

        Args:
            audio: Audio file path or numpy array of audio samples.
            sampling_rate: Sample rate of the input audio.
            is_personalized_MOS: Whether to use personalized MOS coefficients. Defaults to False.

        Returns:
            Dictionary containing MOS scores and metadata:
                - filename: Audio clip identifier
                - len_in_sec: Audio length in seconds
                - sr: Sample rate
                - num_hops: Number of analysis windows
                - OVRL_raw: Raw overall quality score
                - SIG_raw: Raw signal quality score
                - BAK_raw: Raw background noise score
                - OVRL: Polynomial-fitted overall quality score
                - SIG: Polynomial-fitted signal quality score
                - BAK: Polynomial-fitted background noise score
                - P808_MOS: P.808 MOS score
        """
        fs = SAMPLING_RATE
        if isinstance(audio, str):
            audio, _ = librosa.load(audio, sr=fs)
        elif sampling_rate != fs:
            # resample audio
            audio = librosa.resample(audio, orig_sr=sampling_rate, target_sr=fs)
        actual_audio_len = len(audio)
        len_samples = int(INPUT_LENGTH * fs)
        while len(audio) < len_samples:
            audio = np.append(audio, audio)
        num_hops = int(np.floor(len(audio) / fs) - INPUT_LENGTH) + 1
        hop_len_samples = fs
        predicted_mos_sig_seg_raw: List[float] = []
        predicted_mos_bak_seg_raw: List[float] = []
        predicted_mos_ovr_seg_raw: List[float] = []
        predicted_mos_sig_seg: List[float] = []
        predicted_mos_bak_seg: List[float] = []
        predicted_mos_ovr_seg: List[float] = []
        predicted_p808_mos: List[float] = []

        for idx in range(num_hops):
            audio_seg = audio[
                int(idx * hop_len_samples) : int((idx + INPUT_LENGTH) * hop_len_samples)
            ]
            if len(audio_seg) < len_samples:
                continue
            input_features = np.array(audio_seg).astype("float32")[np.newaxis, :]
            p808_input_features = np.array(self.audio_melspec(audio=audio_seg[:-160])).astype(
                "float32",
            )[np.newaxis, :, :]
            oi = {"input_1": input_features}
            p808_oi = {"input_1": p808_input_features}
            p808_mos = self.p808_onnx_sess.run(None, p808_oi)[0][0][0]
            mos_sig_raw, mos_bak_raw, mos_ovr_raw = self.onnx_sess.run(None, oi)[0][0]
            mos_sig, mos_bak, mos_ovr = self.get_polyfit_val(
                mos_sig_raw,
                mos_bak_raw,
                mos_ovr_raw,
                is_personalized_MOS,
            )
            predicted_mos_sig_seg_raw.append(mos_sig_raw)
            predicted_mos_bak_seg_raw.append(mos_bak_raw)
            predicted_mos_ovr_seg_raw.append(mos_ovr_raw)
            predicted_mos_sig_seg.append(mos_sig)
            predicted_mos_bak_seg.append(mos_bak)
            predicted_mos_ovr_seg.append(mos_ovr)
            predicted_p808_mos.append(p808_mos)
        clip_dict: Dict[str, Union[str, float, int]] = {
            "filename": "audio_clip",
            "len_in_sec": actual_audio_len / fs,
            "sr": fs,
        }
        clip_dict["num_hops"] = num_hops
        clip_dict["OVRL_raw"] = np.mean(predicted_mos_ovr_seg_raw)
        clip_dict["SIG_raw"] = np.mean(predicted_mos_sig_seg_raw)
        clip_dict["BAK_raw"] = np.mean(predicted_mos_bak_seg_raw)
        clip_dict["OVRL"] = np.mean(predicted_mos_ovr_seg)
        clip_dict["SIG"] = np.mean(predicted_mos_sig_seg)
        clip_dict["BAK"] = np.mean(predicted_mos_bak_seg)
        clip_dict["P808_MOS"] = np.mean(predicted_p808_mos)
        return clip_dict

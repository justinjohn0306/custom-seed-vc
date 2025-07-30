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

"""OpenVoice API for voice conversion."""

import re
from typing import List, Optional

import numpy as np
import soundfile
import torch

from . import commons, utils
from .mel_processing import spectrogram_torch
from .models import SynthesizerTrn


class OpenVoiceBaseClass(object):
    """Base class for OpenVoice models.

    This class provides common functionality for loading and managing OpenVoice models.
    """

    def __init__(self, config_path: str, device: str = "cuda:0") -> None:
        """Initialize OpenVoiceBaseClass.

        Args:
            config_path: Path to the configuration file.
            device: Device to run the model on (e.g., 'cuda:0', 'cpu').
        """
        if "cuda" in device:
            assert torch.cuda.is_available()

        hps = utils.get_hparams_from_file(config_path)

        model = SynthesizerTrn(
            len(getattr(hps, "symbols", [])),
            hps.data.filter_length // 2 + 1,
            n_speakers=hps.data.n_speakers,
            **hps.model,
        ).to(device)

        model.eval()
        self.model = model
        self.hps = hps
        self.device = device

    def load_ckpt(self, ckpt_path: str) -> None:
        """Load model checkpoint.

        Args:
            ckpt_path: Path to the checkpoint file.
        """
        checkpoint_dict = torch.load(ckpt_path, map_location=torch.device(self.device))
        a, b = self.model.load_state_dict(checkpoint_dict["model"], strict=False)
        print("Loaded checkpoint '{}'".format(ckpt_path))
        print("missing/unexpected keys:", a, b)


class BaseSpeakerTTS(OpenVoiceBaseClass):
    """Text-to-Speech model with speaker control.

    This class provides TTS functionality with support for multiple speakers and languages.
    """

    language_marks = {
        "english": "EN",
        "chinese": "ZH",
    }

    @staticmethod
    def get_text(text: str, hps, is_symbol: bool) -> torch.Tensor:
        """Convert text to sequence of tokens.

        Args:
            text: Input text to convert.
            hps: Hyperparameters object.
            is_symbol: Whether the text contains phoneme symbols.

        Returns:
            Tensor containing the token sequence.
        """
        from openvoice.text import text_to_sequence  # delayed import to fix F821

        text_norm = text_to_sequence(text, hps.symbols, [] if is_symbol else hps.data.text_cleaners)
        if hps.data.add_blank:
            text_norm = commons.intersperse(text_norm, 0)
        text_norm = torch.LongTensor(text_norm)
        return text_norm

    @staticmethod
    def audio_numpy_concat(
        segment_data_list: List[np.ndarray],
        sr: int,
        speed: float = 1.0,
    ) -> np.ndarray:
        """Concatenate audio segments with silence between them.

        Args:
            segment_data_list: List of audio segments as numpy arrays.
            sr: Sample rate.
            speed: Speed factor for playback.

        Returns:
            Concatenated audio as numpy array.
        """
        audio_segments = []
        for segment_data in segment_data_list:
            audio_segments += segment_data.reshape(-1).tolist()
            audio_segments += [0] * int((sr * 0.05) / speed)
        audio_segments = np.array(audio_segments).astype(np.float32)
        return audio_segments

    @staticmethod
    def split_sentences_into_pieces(text: str, language_str: str) -> List[str]:
        """Split text into sentences based on language.

        Args:
            text: Input text to split.
            language_str: Language identifier string.

        Returns:
            List of sentence pieces.
        """
        texts = utils.split_sentence(text, language_str=language_str)
        print(" > Text splitted to sentences.")
        print("\n".join(texts))
        print(" > ===========================")
        return texts

    def tts(
        self,
        text: str,
        output_path: Optional[str],
        speaker: str,
        language: str = "English",
        speed: float = 1.0,
    ) -> Optional[np.ndarray]:
        """Generate speech from text.

        Args:
            text: Input text to synthesize.
            output_path: Path to save the audio file. If None, returns audio array.
            speaker: Speaker name.
            language: Language for synthesis (default: "English").
            speed: Speed factor for synthesis (default: 1.0).

        Returns:
            Audio array if output_path is None, otherwise None.
        """
        mark = self.language_marks.get(language.lower(), None)
        assert mark is not None, f"language {language} is not supported"

        texts = self.split_sentences_into_pieces(text, mark)

        audio_list = []
        for t in texts:
            t = re.sub(r"([a-z])([A-Z])", r"\1 \2", t)
            t = f"[{mark}]{t}[{mark}]"
            stn_tst = self.get_text(t, self.hps, False)
            device = self.device
            speaker_id = self.hps.speakers[speaker]
            with torch.no_grad():
                x_tst = stn_tst.unsqueeze(0).to(device)
                x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).to(device)
                sid = torch.LongTensor([speaker_id]).to(device)
                audio = (
                    self.model.infer(
                        x_tst,
                        x_tst_lengths,
                        sid=sid,
                        noise_scale=0.667,
                        noise_scale_w=0.6,
                        length_scale=1.0 / speed,
                    )[0][0, 0]
                    .data.cpu()
                    .float()
                    .numpy()
                )
            audio_list.append(audio)
        audio = self.audio_numpy_concat(audio_list, sr=self.hps.data.sampling_rate, speed=speed)

        if output_path is None:
            return audio
        else:
            soundfile.write(output_path, audio, self.hps.data.sampling_rate)


class ToneColorConverter(OpenVoiceBaseClass):
    """Voice conversion model for tone color transfer.

    This class provides functionality to convert voice characteristics from one speaker to another.
    """

    def __init__(self, *args, **kwargs) -> None:
        """Initialize ToneColorConverter.

        Args:
            *args: Positional arguments passed to parent class.
            **kwargs: Keyword arguments passed to parent class.
        """
        super().__init__(*args, **kwargs)

        # if kwargs.get('enable_watermark', True):
        #     import wavmark
        #     self.watermark_model = wavmark.load_model().to(self.device)
        # else:
        #     self.watermark_model = None
        self.version = getattr(self.hps, "_version_", "v1")

    def extract_se(self, waves: torch.Tensor, wave_lengths: torch.Tensor) -> torch.Tensor:
        """Extract speaker embeddings from audio waves.

        Args:
            waves: Audio waveforms as tensor.
            wave_lengths: Lengths of each waveform.

        Returns:
            Speaker embeddings as tensor.
        """
        device = self.device
        hps = self.hps
        gs = []

        for wav_tensor, wav_len in zip(waves, wave_lengths, strict=False):
            y = wav_tensor[:wav_len]
            y = y[None, :]
            y = spectrogram_torch(
                y,
                hps.data.filter_length,
                hps.data.sampling_rate,
                hps.data.hop_length,
                hps.data.win_length,
                center=False,
            ).to(device)
            with torch.no_grad():
                g = self.model.ref_enc(y.transpose(1, 2)).unsqueeze(-1)
                gs.append(g.detach())
        gs = torch.stack(gs)
        gs = gs.squeeze(1).squeeze(-1)
        return gs

    def convert(
        self,
        src_waves: torch.Tensor,
        src_wave_lengths: torch.Tensor,
        src_se: torch.Tensor,
        tgt_se: torch.Tensor,
        tau: float = 0.3,
        message: str = "default",
    ) -> torch.Tensor:
        """Convert source audio to target speaker's voice.

        Args:
            src_waves: Source audio waveforms.
            src_wave_lengths: Lengths of source waveforms.
            src_se: Source speaker embeddings.
            tgt_se: Target speaker embeddings.
            tau: Conversion strength parameter (default: 0.3).
            message: Watermark message (default: "default").

        Returns:
            Converted audio waveform.
        """
        hps = self.hps
        # load audio
        with torch.no_grad():
            y = src_waves
            spec = spectrogram_torch(
                y,
                hps.data.filter_length,
                hps.data.sampling_rate,
                hps.data.hop_length,
                hps.data.win_length,
                center=False,
            ).to(self.device)
            spec_lengths = src_wave_lengths // hps.data.hop_length
            spec_lengths = spec_lengths.clamp(min=1, max=spec.size(2))
            audio = self.model.voice_conversion(
                spec,
                spec_lengths,
                sid_src=src_se.unsqueeze(-1),
                sid_tgt=tgt_se.unsqueeze(-1),
                tau=tau,
            )[0]
        return audio

    def add_watermark(self, audio: np.ndarray, message: str) -> np.ndarray:
        """Add watermark to audio (currently disabled).

        Args:
            audio: Input audio array.
            message: Watermark message.

        Returns:
            Audio array (unchanged as watermarking is disabled).
        """
        # if self.watermark_model is None:
        return audio
        device = self.device
        bits = utils.string_to_bits(message).reshape(-1)
        n_repeat = len(bits) // 32

        K = 16000
        coeff = 2
        for n in range(n_repeat):
            trunck = audio[(coeff * n) * K : (coeff * n + 1) * K]
            if len(trunck) != K:
                print("Audio too short, fail to add watermark")
                break
            message_npy = bits[n * 32 : (n + 1) * 32]

            with torch.no_grad():
                signal = torch.FloatTensor(trunck).to(device)[None]
                message_tensor = torch.FloatTensor(message_npy).to(device)[None]
                signal_wmd_tensor = self.watermark_model.encode(signal, message_tensor)
                signal_wmd_npy = signal_wmd_tensor.detach().cpu().squeeze()
            audio[(coeff * n) * K : (coeff * n + 1) * K] = signal_wmd_npy
        return audio

    def detect_watermark(self, audio: np.ndarray, n_repeat: int) -> str:
        """Detect watermark in audio.

        Args:
            audio: Input audio array.
            n_repeat: Number of repetitions to check.

        Returns:
            Detected watermark message or "Fail" if detection fails.
        """
        bits = []
        K = 16000
        coeff = 2
        for n in range(n_repeat):
            trunck = audio[(coeff * n) * K : (coeff * n + 1) * K]
            if len(trunck) != K:
                print("Audio too short, fail to detect watermark")
                return "Fail"
            with torch.no_grad():
                signal = torch.FloatTensor(trunck).to(self.device).unsqueeze(0)
                message_decoded_npy = (
                    (self.watermark_model.decode(signal) >= 0.5)
                    .int()
                    .detach()
                    .cpu()
                    .numpy()
                    .squeeze()
                )
            bits.append(message_decoded_npy)
        bits = np.stack(bits).reshape(-1, 8)
        message = utils.bits_to_string(bits)
        return message

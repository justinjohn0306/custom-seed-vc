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

"""Main wrapper for Seed-VC voice conversion models."""

from pathlib import Path
from typing import Generator, List, Optional, Tuple, Union

import librosa
import numpy as np
import torch
import torchaudio
import yaml
from pydub import AudioSegment
from transformers import AutoFeatureExtractor, WhisperModel

from seed_vc.hf_utils import load_custom_model_from_hf
from seed_vc.modules.audio import mel_spectrogram
from seed_vc.modules.bigvgan import bigvgan
from seed_vc.modules.campplus.DTDNN import CAMPPlus
from seed_vc.modules.commons import build_model, load_checkpoint, recursive_munch
from seed_vc.modules.rmvpe import RMVPE


class SeedVCWrapper:
    """Wrapper for Seed-VC voice conversion models.

    This class provides a unified interface for voice conversion using Seed-VC models,
    supporting both standard and F0-conditioned voice conversion with streaming capabilities.

    Attributes:
        device: The torch device to use for computations.
        model: The base DiT model for voice conversion.
        model_f0: The F0-conditioned DiT model.
        sr: Sample rate for the base model (22050 Hz).
        sr_f0: Sample rate for the F0 model (44100 Hz).
        hop_length: Hop length for the base model.
        hop_length_f0: Hop length for the F0 model.
        overlap_frame_len: Number of overlapping frames for streaming.
        bitrate: Bitrate for MP3 encoding.

    Example:
        >>> wrapper = SeedVCWrapper(device="cuda")
        >>> # Convert voice from source to target
        >>> audio = wrapper.convert_voice(
        ...     source="path/to/source.wav",
        ...     target="path/to/target.wav",
        ...     stream_output=False
        ... )
    """

    def __init__(self, device: Optional[Union[str, torch.device]] = None) -> None:
        """Initialize the Seed-VC wrapper with all necessary models and configurations.

        Args:
            device: Torch device to use. Can be a string ("cuda", "mps", "cpu") or
                a torch.device object. If None, will be automatically determined
                based on available hardware.
        """
        # Set device
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = device

        # Load base model and configuration
        self._load_base_model()

        # Load F0 conditioned model
        self._load_f0_model()

        # Load additional modules
        self._load_additional_modules()

        # Set streaming parameters
        self.overlap_frame_len = 16
        self.bitrate = "320k"

    def _load_base_model(self) -> None:
        """Load the base DiT model for voice conversion.

        This method loads the base DiT model with Whisper small encoder and
        WaveNet decoder, along with its configuration. It also sets up the
        mel spectrogram extraction function and loads the Whisper model for
        feature extraction.

        The loaded models are stored as instance attributes:
        - self.model: The DiT model dictionary
        - self.whisper_model: The Whisper encoder
        - self.whisper_feature_extractor: The Whisper feature extractor
        - self.to_mel: Mel spectrogram extraction function
        """
        dit_checkpoint_path, dit_config_path = load_custom_model_from_hf(
            "Plachta/Seed-VC",
            "DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan_pruned.pth",
            "config_dit_mel_seed_uvit_whisper_small_wavenet.yml",
        )
        config = yaml.safe_load(open(dit_config_path, "r"))
        model_params = recursive_munch(config["model_params"])
        self.model = build_model(model_params, stage="DiT")
        self.hop_length = config["preprocess_params"]["spect_params"]["hop_length"]
        self.sr = config["preprocess_params"]["sr"]

        # Load checkpoints
        self.model, _, _, _ = load_checkpoint(
            self.model,
            None,
            dit_checkpoint_path,
            load_only_params=True,
            ignore_modules=[],
            is_distributed=False,
        )
        for key in self.model:
            self.model[key].eval()
            self.model[key].to(self.device)
        self.model.cfm.estimator.setup_caches(max_batch_size=1, max_seq_length=8192)

        # Set up mel spectrogram function
        mel_fn_args = {
            "n_fft": config["preprocess_params"]["spect_params"]["n_fft"],
            "win_size": config["preprocess_params"]["spect_params"]["win_length"],
            "hop_size": config["preprocess_params"]["spect_params"]["hop_length"],
            "num_mels": config["preprocess_params"]["spect_params"]["n_mels"],
            "sampling_rate": self.sr,
            "fmin": 0,
            "fmax": None,
            "center": False,
        }
        self.to_mel = lambda x: mel_spectrogram(x, **mel_fn_args)

        # Load whisper model
        whisper_name = (
            model_params.speech_tokenizer.whisper_name
            if hasattr(model_params.speech_tokenizer, "whisper_name")
            else "openai/whisper-small"
        )
        self.whisper_model = WhisperModel.from_pretrained(
            whisper_name,
            torch_dtype=torch.float16,
        ).to(self.device)
        del self.whisper_model.decoder
        self.whisper_feature_extractor = AutoFeatureExtractor.from_pretrained(whisper_name)

    def _load_f0_model(self) -> None:
        """Load the F0 conditioned model for voice conversion.

        This method loads the F0-conditioned DiT model with Whisper base encoder
        at 44kHz sampling rate. The F0 model allows for pitch-controllable voice
        conversion.

        The loaded models are stored as instance attributes:
        - self.model_f0: The F0-conditioned DiT model dictionary
        - self.to_mel_f0: Mel spectrogram extraction function for 44kHz audio
        """
        dit_checkpoint_path, dit_config_path = load_custom_model_from_hf(
            "Plachta/Seed-VC",
            "DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema.pth",
            "config_dit_mel_seed_uvit_whisper_base_f0_44k.yml",
        )
        config = yaml.safe_load(open(dit_config_path, "r"))
        model_params = recursive_munch(config["model_params"])
        self.model_f0 = build_model(model_params, stage="DiT")
        self.hop_length_f0 = config["preprocess_params"]["spect_params"]["hop_length"]
        self.sr_f0 = config["preprocess_params"]["sr"]

        # Load checkpoints
        self.model_f0, _, _, _ = load_checkpoint(
            self.model_f0,
            None,
            dit_checkpoint_path,
            load_only_params=True,
            ignore_modules=[],
            is_distributed=False,
        )
        for key in self.model_f0:
            self.model_f0[key].eval()
            self.model_f0[key].to(self.device)
        self.model_f0.cfm.estimator.setup_caches(max_batch_size=1, max_seq_length=8192)

        # Set up mel spectrogram function for F0 model
        mel_fn_args_f0 = {
            "n_fft": config["preprocess_params"]["spect_params"]["n_fft"],
            "win_size": config["preprocess_params"]["spect_params"]["win_length"],
            "hop_size": config["preprocess_params"]["spect_params"]["hop_length"],
            "num_mels": config["preprocess_params"]["spect_params"]["n_mels"],
            "sampling_rate": self.sr_f0,
            "fmin": 0,
            "fmax": None,
            "center": False,
        }
        self.to_mel_f0 = lambda x: mel_spectrogram(x, **mel_fn_args_f0)

    def _load_additional_modules(self) -> None:
        """Load additional modules like CAMPPlus, BigVGAN, and RMVPE.

        This method loads supporting models:
        - CAMPPlus: For speaker embedding extraction
        - BigVGAN: For waveform generation (both 22kHz and 44kHz versions)
        - RMVPE: For robust F0 extraction

        The loaded models are stored as instance attributes:
        - self.campplus_model: CAMPPlus speaker encoder
        - self.bigvgan_model: BigVGAN vocoder at 22kHz
        - self.bigvgan_44k_model: BigVGAN vocoder at 44kHz
        - self.rmvpe: RMVPE F0 extractor
        """
        # Load CAMPPlus
        campplus_ckpt_path = load_custom_model_from_hf(
            "funasr/campplus",
            "campplus_cn_common.bin",
            config_filename=None,
        )
        self.campplus_model = CAMPPlus(feat_dim=80, embedding_size=192)
        self.campplus_model.load_state_dict(torch.load(campplus_ckpt_path, map_location="cpu"))
        self.campplus_model.eval()
        self.campplus_model.to(self.device)

        # Load BigVGAN models
        self.bigvgan_model = bigvgan.BigVGAN.from_pretrained(
            "nvidia/bigvgan_v2_22khz_80band_256x",
            use_cuda_kernel=False,
        )
        self.bigvgan_model.remove_weight_norm()
        self.bigvgan_model = self.bigvgan_model.eval().to(self.device)

        self.bigvgan_44k_model = bigvgan.BigVGAN.from_pretrained(
            "nvidia/bigvgan_v2_44khz_128band_512x",
            use_cuda_kernel=False,
        )
        self.bigvgan_44k_model.remove_weight_norm()
        self.bigvgan_44k_model = self.bigvgan_44k_model.eval().to(self.device)

        # Load RMVPE for F0 extraction
        model_path = load_custom_model_from_hf("lj1995/VoiceConversionWebUI", "rmvpe.pt", None)
        self.rmvpe = RMVPE(model_path, is_half=False, device=self.device)

    @staticmethod
    def adjust_f0_semitones(f0_sequence: torch.Tensor, n_semitones: float) -> torch.Tensor:
        """Adjust F0 values by a number of semitones.

        Args:
            f0_sequence: Tensor containing F0 values in Hz.
            n_semitones: Number of semitones to shift (positive for higher pitch,
                negative for lower pitch).

        Returns:
            Adjusted F0 sequence with the same shape as input.

        Example:
            >>> f0 = torch.tensor([440.0, 880.0])  # A4 and A5
            >>> shifted = SeedVCWrapper.adjust_f0_semitones(f0, 12)  # Up one octave
            >>> print(shifted)  # tensor([880.0, 1760.0])
        """
        factor = 2 ** (n_semitones / 12)
        return f0_sequence * factor

    @staticmethod
    def crossfade(chunk1: np.ndarray, chunk2: np.ndarray, overlap: int) -> np.ndarray:
        """Apply crossfade between two audio chunks.

        Args:
            chunk1: First audio chunk (numpy array).
            chunk2: Second audio chunk (numpy array).
            overlap: Number of samples to overlap.

        Returns:
            The second chunk with crossfade applied at the beginning.

        Note:
            Uses cosine-squared fade curves for smooth transitions.
        """
        fade_out = np.cos(np.linspace(0, np.pi / 2, overlap)) ** 2
        fade_in = np.cos(np.linspace(np.pi / 2, 0, overlap)) ** 2
        if len(chunk2) < overlap:
            chunk2[:overlap] = (
                chunk2[:overlap] * fade_in[: len(chunk2)]
                + (chunk1[-overlap:] * fade_out)[: len(chunk2)]
            )
        else:
            chunk2[:overlap] = chunk2[:overlap] * fade_in + chunk1[-overlap:] * fade_out
        return chunk2

    def _stream_wave_chunks(
        self,
        vc_wave: torch.Tensor,
        processed_frames: int,
        vc_target: torch.Tensor,
        overlap_wave_len: int,
        generated_wave_chunks: List[np.ndarray],
        previous_chunk: Optional[torch.Tensor],
        is_last_chunk: bool,
        stream_output: bool,
        sr: int,
    ) -> Tuple[
        int,
        Optional[torch.Tensor],
        bool,
        Optional[bytes],
        Optional[Union[np.ndarray, Tuple[int, np.ndarray]]],
    ]:
        """Helper method to handle streaming wave chunks.

        Args:
            vc_wave: The current wave chunk tensor.
            processed_frames: Number of frames processed so far.
            vc_target: The target mel spectrogram tensor.
            overlap_wave_len: Length of overlap between chunks in samples.
            generated_wave_chunks: List to accumulate generated wave chunks.
            previous_chunk: Previous wave chunk for crossfading.
            is_last_chunk: Whether this is the last chunk.
            stream_output: Whether to stream the output as MP3.
            sr: Sample rate in Hz.

        Returns:
            Tuple containing:
                - processed_frames: Updated frame count
                - previous_chunk: Updated previous chunk for next iteration
                - should_break: Whether processing should stop
                - mp3_bytes: MP3 encoded bytes if streaming, None otherwise
                - full_audio: Full audio array or (sr, array) tuple if last chunk,
                    None otherwise
        """
        mp3_bytes = None
        full_audio = None

        if processed_frames == 0:
            if is_last_chunk:
                output_wave = vc_wave[0].cpu().numpy()
                generated_wave_chunks.append(output_wave)

                if stream_output:
                    output_wave_int16 = (output_wave * 32768.0).astype(np.int16)
                    mp3_bytes = (
                        AudioSegment(
                            output_wave_int16.tobytes(),
                            frame_rate=sr,
                            sample_width=output_wave_int16.dtype.itemsize,
                            channels=1,
                        )
                        .export(format="mp3", bitrate=self.bitrate)
                        .read()
                    )
                    full_audio = (sr, np.concatenate(generated_wave_chunks))
                else:
                    return (
                        processed_frames,
                        previous_chunk,
                        True,
                        None,
                        np.concatenate(generated_wave_chunks),
                    )

                return processed_frames, previous_chunk, True, mp3_bytes, full_audio

            output_wave = vc_wave[0, :-overlap_wave_len].cpu().numpy()
            generated_wave_chunks.append(output_wave)
            previous_chunk = vc_wave[0, -overlap_wave_len:]
            processed_frames += vc_target.size(2) - self.overlap_frame_len

            if stream_output:
                output_wave_int16 = (output_wave * 32768.0).astype(np.int16)
                mp3_bytes = (
                    AudioSegment(
                        output_wave_int16.tobytes(),
                        frame_rate=sr,
                        sample_width=output_wave_int16.dtype.itemsize,
                        channels=1,
                    )
                    .export(format="mp3", bitrate=self.bitrate)
                    .read()
                )

        elif is_last_chunk:
            output_wave = self.crossfade(
                previous_chunk.cpu().numpy(),
                vc_wave[0].cpu().numpy(),
                overlap_wave_len,
            )
            generated_wave_chunks.append(output_wave)
            processed_frames += vc_target.size(2) - self.overlap_frame_len

            if stream_output:
                output_wave_int16 = (output_wave * 32768.0).astype(np.int16)
                mp3_bytes = (
                    AudioSegment(
                        output_wave_int16.tobytes(),
                        frame_rate=sr,
                        sample_width=output_wave_int16.dtype.itemsize,
                        channels=1,
                    )
                    .export(format="mp3", bitrate=self.bitrate)
                    .read()
                )
                full_audio = (sr, np.concatenate(generated_wave_chunks))
            else:
                return (
                    processed_frames,
                    previous_chunk,
                    True,
                    None,
                    np.concatenate(generated_wave_chunks),
                )

            return processed_frames, previous_chunk, True, mp3_bytes, full_audio

        else:
            output_wave = self.crossfade(
                previous_chunk.cpu().numpy(),
                vc_wave[0, :-overlap_wave_len].cpu().numpy(),
                overlap_wave_len,
            )
            generated_wave_chunks.append(output_wave)
            previous_chunk = vc_wave[0, -overlap_wave_len:]
            processed_frames += vc_target.size(2) - self.overlap_frame_len

            if stream_output:
                output_wave_int16 = (output_wave * 32768.0).astype(np.int16)
                mp3_bytes = (
                    AudioSegment(
                        output_wave_int16.tobytes(),
                        frame_rate=sr,
                        sample_width=output_wave_int16.dtype.itemsize,
                        channels=1,
                    )
                    .export(format="mp3", bitrate=self.bitrate)
                    .read()
                )

        return processed_frames, previous_chunk, False, mp3_bytes, full_audio

    def _process_whisper_features(
        self,
        audio_16k: torch.Tensor,
        is_source: bool = True,
    ) -> torch.Tensor:
        """Process audio through Whisper model to extract features.

        Args:
            audio_16k: Audio tensor at 16kHz sampling rate, shape (1, samples).
            is_source: Whether this is source audio (unused but kept for compatibility).

        Returns:
            Extracted features tensor with shape (1, time_steps, feature_dim).

        Note:
            For audio longer than 30 seconds, processes in overlapping chunks
            of 30 seconds with 5 seconds overlap to maintain continuity.
        """
        if audio_16k.size(-1) <= 16000 * 30:
            # If audio is short enough, process in one go
            inputs = self.whisper_feature_extractor(
                [audio_16k.squeeze(0).cpu().numpy()],
                return_tensors="pt",
                return_attention_mask=True,
                sampling_rate=16000,
            )
            input_features = self.whisper_model._mask_input_features(
                inputs.input_features,
                attention_mask=inputs.attention_mask,
            ).to(self.device)
            outputs = self.whisper_model.encoder(
                input_features.to(self.whisper_model.encoder.dtype),
                head_mask=None,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
            )
            features = outputs.last_hidden_state.to(torch.float32)
            features = features[:, : audio_16k.size(-1) // 320 + 1]
        else:
            # Process long audio in chunks
            overlapping_time = 5  # 5 seconds
            features_list = []
            buffer = None
            traversed_time = 0
            while traversed_time < audio_16k.size(-1):
                if buffer is None:  # first chunk
                    chunk = audio_16k[:, traversed_time : traversed_time + 16000 * 30]
                else:
                    chunk = torch.cat(
                        [
                            buffer,
                            audio_16k[
                                :,
                                traversed_time : traversed_time + 16000 * (30 - overlapping_time),
                            ],
                        ],
                        dim=-1,
                    )
                inputs = self.whisper_feature_extractor(
                    [chunk.squeeze(0).cpu().numpy()],
                    return_tensors="pt",
                    return_attention_mask=True,
                    sampling_rate=16000,
                )
                input_features = self.whisper_model._mask_input_features(
                    inputs.input_features,
                    attention_mask=inputs.attention_mask,
                ).to(self.device)
                outputs = self.whisper_model.encoder(
                    input_features.to(self.whisper_model.encoder.dtype),
                    head_mask=None,
                    output_attentions=False,
                    output_hidden_states=False,
                    return_dict=True,
                )
                chunk_features = outputs.last_hidden_state.to(torch.float32)
                chunk_features = chunk_features[:, : chunk.size(-1) // 320 + 1]
                if traversed_time == 0:
                    features_list.append(chunk_features)
                else:
                    features_list.append(chunk_features[:, 50 * overlapping_time :])
                buffer = chunk[:, -16000 * overlapping_time :]
                traversed_time += (
                    30 * 16000 if traversed_time == 0 else chunk.size(-1) - 16000 * overlapping_time
                )
            features = torch.cat(features_list, dim=1)

        return features

    @torch.no_grad()
    @torch.inference_mode()
    def convert_voice(
        self,
        source: Union[str, Path],
        target: Union[str, Path],
        diffusion_steps: int = 10,
        length_adjust: float = 1.0,
        inference_cfg_rate: float = 0.7,
        f0_condition: bool = False,
        auto_f0_adjust: bool = True,
        pitch_shift: float = 0,
        stream_output: bool = True,
    ) -> Union[np.ndarray, Generator[Tuple[bytes, Optional[Tuple[int, np.ndarray]]], None, None]]:
        """Convert both timbre and voice from source to target.

        Args:
            source: Path to source audio file. Supports common formats (wav, mp3, etc).
            target: Path to target/reference audio file for voice cloning.
            diffusion_steps: Number of diffusion steps. More steps = better quality but slower.
                Recommended range: 5-50, default: 10.
            length_adjust: Length adjustment factor. Values > 1.0 make output longer,
                < 1.0 make it shorter. Default: 1.0.
            inference_cfg_rate: Classifier-free guidance rate. Higher values follow the
                target voice more closely. Range: 0.0-1.0, default: 0.7.
            f0_condition: Whether to use F0 (pitch) conditioning. Enables pitch control
                but requires more computation. Default: False.
            auto_f0_adjust: Whether to automatically adjust F0 to match target speaker's
                pitch range. Only used when f0_condition=True. Default: True.
            pitch_shift: Manual pitch shift in semitones. Positive values increase pitch,
                negative values decrease it. Only used when f0_condition=True. Default: 0.
            stream_output: Whether to stream the output as MP3 chunks for real-time
                playback. If False, returns complete audio array. Default: True.

        Returns:
            If stream_output is True:
                Generator yielding tuples of (mp3_bytes, full_audio) where:
                - mp3_bytes: Encoded MP3 audio chunk as bytes
                - full_audio: None until the last chunk, then (sample_rate, audio_array)

            If stream_output is False:
                Complete audio as numpy array with shape (samples,)

        Raises:
            RuntimeError: If model loading fails or CUDA out of memory.
            FileNotFoundError: If source or target audio files don't exist.

        Example:
            >>> # Non-streaming conversion
            >>> audio = wrapper.convert_voice("input.wav", "reference.wav", stream_output=False)
            >>>
            >>> # Streaming conversion
            >>> for mp3_chunk, full_audio in wrapper.convert_voice("input.wav", "reference.wav"):
            ...     if full_audio:
            ...         sr, audio = full_audio
            ...         # Process complete audio
        """
        # Select appropriate models based on F0 condition
        inference_module = self.model if not f0_condition else self.model_f0
        mel_fn = self.to_mel if not f0_condition else self.to_mel_f0
        bigvgan_fn = self.bigvgan_model if not f0_condition else self.bigvgan_44k_model
        sr = 22050 if not f0_condition else 44100
        hop_length = 256 if not f0_condition else 512
        max_context_window = sr // hop_length * 30
        overlap_wave_len = self.overlap_frame_len * hop_length

        # Load audio
        source_audio = librosa.load(source, sr=sr)[0]
        ref_audio = librosa.load(target, sr=sr)[0]

        # Process audio
        source_audio = torch.tensor(source_audio).unsqueeze(0).float().to(self.device)
        ref_audio = torch.tensor(ref_audio[: sr * 25]).unsqueeze(0).float().to(self.device)

        # Resample to 16kHz for feature extraction
        ref_waves_16k = torchaudio.functional.resample(ref_audio, sr, 16000)
        converted_waves_16k = torchaudio.functional.resample(source_audio, sr, 16000)

        # Extract Whisper features
        S_alt = self._process_whisper_features(converted_waves_16k, is_source=True)
        S_ori = self._process_whisper_features(ref_waves_16k, is_source=False)

        # Compute mel spectrograms
        mel = mel_fn(source_audio.to(self.device).float())
        mel2 = mel_fn(ref_audio.to(self.device).float())

        # Set target lengths
        target_lengths = torch.LongTensor([int(mel.size(2) * length_adjust)]).to(mel.device)
        target2_lengths = torch.LongTensor([mel2.size(2)]).to(mel2.device)

        # Compute style features
        feat2 = torchaudio.compliance.kaldi.fbank(
            ref_waves_16k,
            num_mel_bins=80,
            dither=0,
            sample_frequency=16000,
        )
        feat2 = feat2 - feat2.mean(dim=0, keepdim=True)
        style2 = self.campplus_model(feat2.unsqueeze(0))

        # Process F0 if needed
        if f0_condition:
            F0_ori = self.rmvpe.infer_from_audio(ref_waves_16k[0], thred=0.03)
            F0_alt = self.rmvpe.infer_from_audio(converted_waves_16k[0], thred=0.03)

            if self.device == "mps":
                F0_ori = torch.from_numpy(F0_ori).float().to(self.device)[None]
                F0_alt = torch.from_numpy(F0_alt).float().to(self.device)[None]
            else:
                F0_ori = torch.from_numpy(F0_ori).to(self.device)[None]
                F0_alt = torch.from_numpy(F0_alt).to(self.device)[None]

            voiced_F0_ori = F0_ori[F0_ori > 1]
            voiced_F0_alt = F0_alt[F0_alt > 1]

            log_f0_alt = torch.log(F0_alt + 1e-5)
            voiced_log_f0_ori = torch.log(voiced_F0_ori + 1e-5)
            voiced_log_f0_alt = torch.log(voiced_F0_alt + 1e-5)
            median_log_f0_ori = torch.median(voiced_log_f0_ori)
            median_log_f0_alt = torch.median(voiced_log_f0_alt)

            # Shift alt log f0 level to ori log f0 level
            shifted_log_f0_alt = log_f0_alt.clone()
            if auto_f0_adjust:
                shifted_log_f0_alt[F0_alt > 1] = (
                    log_f0_alt[F0_alt > 1] - median_log_f0_alt + median_log_f0_ori
                )
            shifted_f0_alt = torch.exp(shifted_log_f0_alt)
            if pitch_shift != 0:
                shifted_f0_alt[F0_alt > 1] = self.adjust_f0_semitones(
                    shifted_f0_alt[F0_alt > 1],
                    pitch_shift,
                )
        else:
            F0_ori = None
            F0_alt = None
            shifted_f0_alt = None

        # Length regulation
        cond, _, codes, commitment_loss, codebook_loss = inference_module.length_regulator(
            S_alt,
            ylens=target_lengths,
            n_quantizers=3,
            f0=shifted_f0_alt,
        )
        prompt_condition, _, codes, commitment_loss, codebook_loss = (
            inference_module.length_regulator(
                S_ori,
                ylens=target2_lengths,
                n_quantizers=3,
                f0=F0_ori,
            )
        )

        # Process in chunks for streaming
        max_source_window = max_context_window - mel2.size(2)
        processed_frames = 0
        generated_wave_chunks = []
        previous_chunk = None

        # Generate chunk by chunk and stream the output
        while processed_frames < cond.size(1):
            chunk_cond = cond[:, processed_frames : processed_frames + max_source_window]
            is_last_chunk = processed_frames + max_source_window >= cond.size(1)
            cat_condition = torch.cat([prompt_condition, chunk_cond], dim=1)

            with torch.autocast(device_type=self.device.type, dtype=torch.float16):
                # Voice Conversion
                vc_target = inference_module.cfm.inference(
                    cat_condition,
                    torch.LongTensor([cat_condition.size(1)]).to(mel2.device),
                    mel2,
                    style2,
                    None,
                    diffusion_steps,
                    inference_cfg_rate=inference_cfg_rate,
                )
                vc_target = vc_target[:, :, mel2.size(-1) :]

            vc_wave = bigvgan_fn(vc_target.float())[0]

            processed_frames, previous_chunk, should_break, mp3_bytes, full_audio = (
                self._stream_wave_chunks(
                    vc_wave,
                    processed_frames,
                    vc_target,
                    overlap_wave_len,
                    generated_wave_chunks,
                    previous_chunk,
                    is_last_chunk,
                    stream_output,
                    sr,
                )
            )

            if stream_output and mp3_bytes is not None:
                yield mp3_bytes, full_audio

            if should_break:
                if not stream_output:
                    return full_audio
                break

        if not stream_output:
            return np.concatenate(generated_wave_chunks)

        return None, None

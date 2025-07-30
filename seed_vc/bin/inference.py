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

import argparse
import glob
import os
import time
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import librosa
import numpy as np
import torch
import torchaudio
import yaml
from torch import nn
from tqdm import tqdm

from seed_vc.hf_utils import load_custom_model_from_hf
from seed_vc.modules.commons import build_model, load_checkpoint, recursive_munch, str2bool

os.environ["HF_HUB_CACHE"] = "./checkpoints/hf_cache"
warnings.simplefilter("ignore")

# Load model and configuration
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

fp16: bool = False


def load_models(
    args: argparse.Namespace,
) -> Tuple[
    Dict[str, nn.Module],
    Callable[[torch.Tensor], torch.Tensor],
    Optional[Callable[[np.ndarray, float], np.ndarray]],
    Union[nn.Module, Callable[[torch.Tensor], torch.Tensor]],
    nn.Module,
    Callable[[torch.Tensor], torch.Tensor],
    Dict[str, Any],
]:
    """Load and initialize all models required for voice conversion.

    Args:
        args: Command line arguments containing model configuration.
            - fp16: Whether to use half precision.
            - f0_condition: Whether to use F0 conditioning.
            - checkpoint: Optional path to checkpoint file.
            - config: Optional path to config file.

    Returns:
        A tuple containing:
            - model: Dictionary of model components.
            - semantic_fn: Function to extract semantic features from audio.
            - f0_fn: Function to extract F0 from audio (None if not using F0).
            - vocoder_fn: Vocoder model or function for waveform generation.
            - campplus_model: Speaker embedding model.
            - to_mel: Function to convert audio to mel spectrogram.
            - mel_fn_args: Dictionary of mel spectrogram parameters.

    Raises:
        ValueError: If unknown vocoder or speech tokenizer type is specified.
    """
    global fp16
    fp16 = args.fp16
    if not args.f0_condition:
        if args.checkpoint is None:
            dit_checkpoint_path, dit_config_path = load_custom_model_from_hf(
                "Plachta/Seed-VC",
                "DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan_pruned.pth",
                "config_dit_mel_seed_uvit_whisper_small_wavenet.yml",
            )
        else:
            dit_checkpoint_path = args.checkpoint
            dit_config_path = args.config
        f0_fn = None
    else:
        if args.checkpoint is None:
            dit_checkpoint_path, dit_config_path = load_custom_model_from_hf(
                "Plachta/Seed-VC",
                "DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema_v2.pth",
                "config_dit_mel_seed_uvit_whisper_base_f0_44k.yml",
            )
        else:
            dit_checkpoint_path = args.checkpoint
            dit_config_path = args.config
        # f0 extractor
        from seed_vc.modules.rmvpe import RMVPE

        model_path = load_custom_model_from_hf("lj1995/VoiceConversionWebUI", "rmvpe.pt", None)
        f0_extractor = RMVPE(model_path, is_half=False, device=device)
        f0_fn = f0_extractor.infer_from_audio

    config: Dict[str, Any] = yaml.safe_load(open(dit_config_path, "r"))
    model_params = recursive_munch(config["model_params"])
    model_params.dit_type = "DiT"
    model: Dict[str, nn.Module] = build_model(model_params, stage="DiT")
    sr: int = config["preprocess_params"]["sr"]

    # Load checkpoints
    model, _, _, _ = load_checkpoint(
        model,
        None,
        dit_checkpoint_path,
        load_only_params=True,
        ignore_modules=[],
        is_distributed=False,
    )
    for key in model:
        model[key].eval()
        model[key].to(device)
    model.cfm.estimator.setup_caches(max_batch_size=1, max_seq_length=8192)

    # Load additional modules
    from seed_vc.modules.campplus.DTDNN import CAMPPlus

    campplus_ckpt_path: str = load_custom_model_from_hf(
        "funasr/campplus",
        "campplus_cn_common.bin",
        config_filename=None,
    )
    campplus_model: nn.Module = CAMPPlus(feat_dim=80, embedding_size=192)
    campplus_model.load_state_dict(torch.load(campplus_ckpt_path, map_location="cpu"))
    campplus_model.eval()
    campplus_model.to(device)

    vocoder_type: str = model_params.vocoder.type

    if vocoder_type == "bigvgan":
        from seed_vc.modules.bigvgan import bigvgan

        bigvgan_name = model_params.vocoder.name
        bigvgan_model = bigvgan.BigVGAN.from_pretrained(bigvgan_name, use_cuda_kernel=False)
        # remove weight norm in the model and set to eval mode
        bigvgan_model.remove_weight_norm()
        bigvgan_model = bigvgan_model.eval().to(device)
        vocoder_fn = bigvgan_model
    elif vocoder_type == "hifigan":
        from seed_vc.modules.hifigan.f0_predictor import ConvRNNF0Predictor
        from seed_vc.modules.hifigan.generator import HiFTGenerator

        hift_config = yaml.safe_load(open("configs/hifigan.yml", "r"))
        hift_gen = HiFTGenerator(
            **hift_config["hift"],
            f0_predictor=ConvRNNF0Predictor(**hift_config["f0_predictor"]),
        )
        hift_path = load_custom_model_from_hf("FunAudioLLM/CosyVoice-300M", "hift.pt", None)
        hift_gen.load_state_dict(torch.load(hift_path, map_location="cpu"))
        hift_gen.eval()
        hift_gen.to(device)
        vocoder_fn = hift_gen
    elif vocoder_type == "vocos":
        vocos_config = yaml.safe_load(open(model_params.vocoder.vocos.config, "r"))
        vocos_path = model_params.vocoder.vocos.path
        vocos_model_params = recursive_munch(vocos_config["model_params"])
        vocos = build_model(vocos_model_params, stage="mel_vocos")
        vocos_checkpoint_path = vocos_path
        vocos, _, _, _ = load_checkpoint(
            vocos,
            None,
            vocos_checkpoint_path,
            load_only_params=True,
            ignore_modules=[],
            is_distributed=False,
        )
        _ = [vocos[key].eval().to(device) for key in vocos]
        _ = [vocos[key].to(device) for key in vocos]
        total_params = sum(
            sum(p.numel() for p in vocos[key].parameters() if p.requires_grad)
            for key in vocos.keys()
        )
        print(f"Vocoder model total parameters: {total_params / 1_000_000:.2f}M")
        vocoder_fn = vocos.decoder
    else:
        raise ValueError(f"Unknown vocoder type: {vocoder_type}")

    speech_tokenizer_type: str = model_params.speech_tokenizer.type
    if speech_tokenizer_type == "whisper":
        # whisper
        from transformers import AutoFeatureExtractor, WhisperModel

        whisper_name = model_params.speech_tokenizer.name
        whisper_model = WhisperModel.from_pretrained(whisper_name, torch_dtype=torch.float16).to(
            device,
        )
        del whisper_model.decoder
        whisper_feature_extractor = AutoFeatureExtractor.from_pretrained(whisper_name)

        def semantic_fn(waves_16k: torch.Tensor) -> torch.Tensor:
            """Extract semantic features using Whisper encoder.

            Args:
                waves_16k: Input waveform tensor at 16kHz sample rate.

            Returns:
                Semantic feature tensor.
            """
            ori_inputs = whisper_feature_extractor(
                [waves_16k.squeeze(0).cpu().numpy()],
                return_tensors="pt",
                return_attention_mask=True,
            )
            ori_input_features = whisper_model._mask_input_features(
                ori_inputs.input_features,
                attention_mask=ori_inputs.attention_mask,
            ).to(device)
            with torch.no_grad():
                ori_outputs = whisper_model.encoder(
                    ori_input_features.to(whisper_model.encoder.dtype),
                    head_mask=None,
                    output_attentions=False,
                    output_hidden_states=False,
                    return_dict=True,
                )
            S_ori = ori_outputs.last_hidden_state.to(torch.float32)
            S_ori = S_ori[:, : waves_16k.size(-1) // 320 + 1]
            return S_ori
    elif speech_tokenizer_type == "cnhubert":
        from transformers import HubertModel, Wav2Vec2FeatureExtractor

        hubert_model_name = config["model_params"]["speech_tokenizer"]["name"]
        hubert_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(hubert_model_name)
        hubert_model = HubertModel.from_pretrained(hubert_model_name)
        hubert_model = hubert_model.to(device)
        hubert_model = hubert_model.eval()
        hubert_model = hubert_model.half()

        def semantic_fn(waves_16k: torch.Tensor) -> torch.Tensor:
            """Extract semantic features using CNHubert model.

            Args:
                waves_16k: Input waveform tensor at 16kHz sample rate.

            Returns:
                Semantic feature tensor.
            """
            ori_waves_16k_input_list = [
                waves_16k[bib].cpu().numpy() for bib in range(len(waves_16k))
            ]
            ori_inputs = hubert_feature_extractor(
                ori_waves_16k_input_list,
                return_tensors="pt",
                return_attention_mask=True,
                padding=True,
                sampling_rate=16000,
            ).to(device)
            with torch.no_grad():
                ori_outputs = hubert_model(
                    ori_inputs.input_values.half(),
                )
            S_ori = ori_outputs.last_hidden_state.float()
            return S_ori
    elif speech_tokenizer_type == "xlsr":
        from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

        model_name = config["model_params"]["speech_tokenizer"]["name"]
        output_layer = config["model_params"]["speech_tokenizer"]["output_layer"]
        wav2vec_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        wav2vec_model = Wav2Vec2Model.from_pretrained(model_name)
        wav2vec_model.encoder.layers = wav2vec_model.encoder.layers[:output_layer]
        wav2vec_model = wav2vec_model.to(device)
        wav2vec_model = wav2vec_model.eval()
        wav2vec_model = wav2vec_model.half()

        def semantic_fn(waves_16k: torch.Tensor) -> torch.Tensor:
            """Extract semantic features using XLSR model.

            Args:
                waves_16k: Input waveform tensor at 16kHz sample rate.

            Returns:
                Semantic feature tensor.
            """
            ori_waves_16k_input_list = [
                waves_16k[bib].cpu().numpy() for bib in range(len(waves_16k))
            ]
            ori_inputs = wav2vec_feature_extractor(
                ori_waves_16k_input_list,
                return_tensors="pt",
                return_attention_mask=True,
                padding=True,
                sampling_rate=16000,
            ).to(device)
            with torch.no_grad():
                ori_outputs = wav2vec_model(
                    ori_inputs.input_values.half(),
                )
            S_ori = ori_outputs.last_hidden_state.float()
            return S_ori
    else:
        raise ValueError(f"Unknown speech tokenizer type: {speech_tokenizer_type}")
    # Generate mel spectrograms
    mel_fn_args: Dict[str, Any] = {
        "n_fft": config["preprocess_params"]["spect_params"]["n_fft"],
        "win_size": config["preprocess_params"]["spect_params"]["win_length"],
        "hop_size": config["preprocess_params"]["spect_params"]["hop_length"],
        "num_mels": config["preprocess_params"]["spect_params"]["n_mels"],
        "sampling_rate": sr,
        "fmin": config["preprocess_params"]["spect_params"].get("fmin", 0),
        "fmax": None
        if config["preprocess_params"]["spect_params"].get("fmax", "None") == "None"
        else 8000,
        "center": False,
    }
    from seed_vc.modules.audio import mel_spectrogram

    to_mel: Callable[[torch.Tensor], torch.Tensor] = lambda x: mel_spectrogram(x, **mel_fn_args)

    return (
        model,
        semantic_fn,
        f0_fn,
        vocoder_fn,
        campplus_model,
        to_mel,
        mel_fn_args,
    )


def adjust_f0_semitones(
    f0_sequence: Union[torch.Tensor, np.ndarray],
    n_semitones: int,
) -> Union[torch.Tensor, np.ndarray]:
    """Adjust F0 sequence by a given number of semitones.

    Args:
        f0_sequence: Input F0 sequence as tensor or numpy array.
        n_semitones: Number of semitones to shift (positive for higher, negative for lower).

    Returns:
        Adjusted F0 sequence with same type as input.
    """
    factor: float = 2 ** (n_semitones / 12)
    return f0_sequence * factor


def crossfade(chunk1: np.ndarray, chunk2: np.ndarray, overlap: int) -> np.ndarray:
    """Apply crossfade between two audio chunks.

    Args:
        chunk1: First audio chunk (previous chunk).
        chunk2: Second audio chunk (current chunk).
        overlap: Number of samples to overlap and crossfade.

    Returns:
        Crossfaded audio chunk with smooth transition.
    """
    fade_out: np.ndarray = np.cos(np.linspace(0, np.pi / 2, overlap)) ** 2
    fade_in: np.ndarray = np.cos(np.linspace(np.pi / 2, 0, overlap)) ** 2
    if len(chunk2) < overlap:
        chunk2[:overlap] = (
            chunk2[:overlap] * fade_in[: len(chunk2)]
            + (chunk1[-overlap:] * fade_out)[: len(chunk2)]
        )
    else:
        chunk2[:overlap] = chunk2[:overlap] * fade_in + chunk1[-overlap:] * fade_out
    return chunk2


def find_audio_files(source_path: str, recursive: bool = True) -> List[str]:
    """Find all audio files in the given path.

    Args:
        source_path: Path to a file or directory.
        recursive: Whether to search recursively in subdirectories.

    Returns:
        List of audio file paths.
    """
    # Supported audio extensions
    audio_extensions = (".wav", ".mp3", ".flac")

    audio_files = []

    if os.path.isfile(source_path):
        # Single file
        if any(source_path.endswith(ext) for ext in audio_extensions):
            audio_files.append(source_path)
        else:
            supported_formats = ", ".join(sorted(set(ext.lower() for ext in audio_extensions)))
            raise ValueError(
                "Source file must be an audio file "
                f"(supported: {supported_formats}), got: {source_path}"
            )
    elif os.path.isdir(source_path):
        # Directory
        for ext in audio_extensions:
            pattern = os.path.join(source_path, "**", f"*{ext}")
            audio_files.extend(glob.glob(pattern, recursive=recursive))
            pattern = os.path.join(source_path, "**", f"*{ext}.upper()")
            audio_files.extend(glob.glob(pattern, recursive=recursive))
        audio_files = sorted(list(set(audio_files)))  # Remove duplicates and sort
    else:
        raise ValueError(f"Source path does not exist: {source_path}")

    if not audio_files:
        raise ValueError(f"No audio files found in: {source_path}")

    return audio_files


@torch.no_grad()
def process_single_file(
    source_file: str,
    target_file: str,
    output_dir: str,
    model: Dict[str, nn.Module],
    semantic_fn: Callable[[torch.Tensor], torch.Tensor],
    f0_fn: Optional[Callable[[np.ndarray, float], np.ndarray]],
    vocoder_fn: Union[nn.Module, Callable[[torch.Tensor], torch.Tensor]],
    campplus_model: nn.Module,
    mel_fn: Callable[[torch.Tensor], torch.Tensor],
    mel_fn_args: Dict[str, Any],
    args: argparse.Namespace,
) -> str:
    """Process a single audio file for voice conversion.

    Args:
        source_file: Path to source audio file.
        target_file: Path to target reference audio file.
        output_dir: Output directory for converted audio.
        model: Dictionary of model components.
        semantic_fn: Function to extract semantic features.
        f0_fn: Function to extract F0 (if using F0 conditioning).
        vocoder_fn: Vocoder model or function.
        campplus_model: Speaker embedding model.
        mel_fn: Function to compute mel spectrogram.
        mel_fn_args: Mel spectrogram parameters.
        args: Command line arguments.

    Returns:
        Path to the output file.
    """
    sr: int = mel_fn_args["sampling_rate"]
    f0_condition: bool = args.f0_condition
    auto_f0_adjust: bool = args.auto_f0_adjust
    pitch_shift: int = args.semi_tone_shift
    diffusion_steps: int = args.diffusion_steps
    length_adjust: float = args.length_adjust
    inference_cfg_rate: float = args.inference_cfg_rate

    # Load audio files
    source_audio: np.ndarray = librosa.load(source_file, sr=sr)[0]
    ref_audio: np.ndarray = librosa.load(target_file, sr=sr)[0]

    sr = 22050 if not f0_condition else 44100
    hop_length = 256 if not f0_condition else 512
    max_context_window: int = sr // hop_length * 30
    overlap_frame_len: int = 16
    overlap_wave_len: int = overlap_frame_len * hop_length

    # Process audio
    source_audio: torch.Tensor = torch.tensor(source_audio).unsqueeze(0).float().to(device)
    ref_audio: torch.Tensor = torch.tensor(ref_audio[: sr * 25]).unsqueeze(0).float().to(device)

    time_vc_start: float = time.time()
    # Resample
    converted_waves_16k: torch.Tensor = torchaudio.functional.resample(source_audio, sr, 16000)
    # if source audio less than 30 seconds, whisper can handle in one forward
    if converted_waves_16k.size(-1) <= 16000 * 30:
        S_alt = semantic_fn(converted_waves_16k)
    else:
        overlapping_time = 5  # 5 seconds
        S_alt_list = []
        buffer = None
        traversed_time = 0
        while traversed_time < converted_waves_16k.size(-1):
            if buffer is None:  # first chunk
                chunk = converted_waves_16k[:, traversed_time : traversed_time + 16000 * 30]
            else:
                chunk = torch.cat(
                    [
                        buffer,
                        converted_waves_16k[
                            :,
                            traversed_time : traversed_time + 16000 * (30 - overlapping_time),
                        ],
                    ],
                    dim=-1,
                )
            S_alt = semantic_fn(chunk)
            if traversed_time == 0:
                S_alt_list.append(S_alt)
            else:
                S_alt_list.append(S_alt[:, 50 * overlapping_time :])
            buffer = chunk[:, -16000 * overlapping_time :]
            traversed_time += (
                30 * 16000 if traversed_time == 0 else chunk.size(-1) - 16000 * overlapping_time
            )
        S_alt = torch.cat(S_alt_list, dim=1)

    ori_waves_16k: torch.Tensor = torchaudio.functional.resample(ref_audio, sr, 16000)
    S_ori: torch.Tensor = semantic_fn(ori_waves_16k)

    mel: torch.Tensor = mel_fn(source_audio.to(device).float())
    mel2: torch.Tensor = mel_fn(ref_audio.to(device).float())

    target_lengths: torch.Tensor = torch.LongTensor([int(mel.size(2) * length_adjust)]).to(
        mel.device,
    )
    target2_lengths: torch.Tensor = torch.LongTensor([mel2.size(2)]).to(mel2.device)

    feat2: torch.Tensor = torchaudio.compliance.kaldi.fbank(
        ori_waves_16k,
        num_mel_bins=80,
        dither=0,
        sample_frequency=16000,
    )
    feat2 = feat2 - feat2.mean(dim=0, keepdim=True)
    style2: torch.Tensor = campplus_model(feat2.unsqueeze(0))

    if f0_condition:
        F0_ori = f0_fn(ori_waves_16k[0], thred=0.03)
        F0_alt = f0_fn(converted_waves_16k[0], thred=0.03)

        F0_ori = torch.from_numpy(F0_ori).to(device)[None]
        F0_alt = torch.from_numpy(F0_alt).to(device)[None]

        voiced_F0_ori = F0_ori[F0_ori > 1]
        voiced_F0_alt = F0_alt[F0_alt > 1]

        log_f0_alt = torch.log(F0_alt + 1e-5)
        voiced_log_f0_ori = torch.log(voiced_F0_ori + 1e-5)
        voiced_log_f0_alt = torch.log(voiced_F0_alt + 1e-5)
        median_log_f0_ori = torch.median(voiced_log_f0_ori)
        median_log_f0_alt = torch.median(voiced_log_f0_alt)

        # shift alt log f0 level to ori log f0 level
        shifted_log_f0_alt = log_f0_alt.clone()
        if auto_f0_adjust:
            shifted_log_f0_alt[F0_alt > 1] = (
                log_f0_alt[F0_alt > 1] - median_log_f0_alt + median_log_f0_ori
            )
        shifted_f0_alt = torch.exp(shifted_log_f0_alt)
        if pitch_shift != 0:
            shifted_f0_alt[F0_alt > 1] = adjust_f0_semitones(
                shifted_f0_alt[F0_alt > 1],
                pitch_shift,
            )
    else:
        F0_ori = None
        F0_alt = None
        shifted_f0_alt = None

    # Length regulation
    cond, _, codes, commitment_loss, codebook_loss = model.length_regulator(
        S_alt,
        ylens=target_lengths,
        n_quantizers=3,
        f0=shifted_f0_alt,
    )
    prompt_condition, _, codes, commitment_loss, codebook_loss = model.length_regulator(
        S_ori,
        ylens=target2_lengths,
        n_quantizers=3,
        f0=F0_ori,
    )

    max_source_window = max_context_window - mel2.size(2)
    # split source condition (cond) into chunks
    processed_frames = 0
    generated_wave_chunks = []
    # generate chunk by chunk and stream the output
    while processed_frames < cond.size(1):
        chunk_cond = cond[:, processed_frames : processed_frames + max_source_window]
        is_last_chunk = processed_frames + max_source_window >= cond.size(1)
        cat_condition = torch.cat([prompt_condition, chunk_cond], dim=1)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16 if fp16 else torch.float32,
        ):
            # Voice Conversion
            vc_target = model.cfm.inference(
                cat_condition,
                torch.LongTensor([cat_condition.size(1)]).to(mel2.device),
                mel2,
                style2,
                None,
                diffusion_steps,
                inference_cfg_rate=inference_cfg_rate,
            )
            vc_target = vc_target[:, :, mel2.size(-1) :]
        vc_wave = vocoder_fn(vc_target.float()).squeeze()
        vc_wave = vc_wave[None, :]
        if processed_frames == 0:
            if is_last_chunk:
                output_wave = vc_wave[0].cpu().numpy()
                generated_wave_chunks.append(output_wave)
                break
            output_wave = vc_wave[0, :-overlap_wave_len].cpu().numpy()
            generated_wave_chunks.append(output_wave)
            previous_chunk = vc_wave[0, -overlap_wave_len:]
            processed_frames += vc_target.size(2) - overlap_frame_len
        elif is_last_chunk:
            output_wave = crossfade(
                previous_chunk.cpu().numpy(),
                vc_wave[0].cpu().numpy(),
                overlap_wave_len,
            )
            generated_wave_chunks.append(output_wave)
            processed_frames += vc_target.size(2) - overlap_frame_len
            break
        else:
            output_wave = crossfade(
                previous_chunk.cpu().numpy(),
                vc_wave[0, :-overlap_wave_len].cpu().numpy(),
                overlap_wave_len,
            )
            generated_wave_chunks.append(output_wave)
            previous_chunk = vc_wave[0, -overlap_wave_len:]
            processed_frames += vc_target.size(2) - overlap_frame_len
    vc_wave: torch.Tensor = torch.tensor(np.concatenate(generated_wave_chunks))[None, :].float()

    time_vc_end: float = time.time()
    rtf = (time_vc_end - time_vc_start) / (vc_wave.size(-1) / sr)
    print(f"  RTF: {rtf:.3f} (processing time / audio duration)")

    # Save output
    source_name: str = os.path.basename(source_file).split(".")[0]
    target_name: str = os.path.basename(target_file).split(".")[0]
    output_file = os.path.join(
        output_dir,
        f"vc_{source_name}_{target_name}_{length_adjust}_{diffusion_steps}_{inference_cfg_rate}.wav",
    )
    torchaudio.save(output_file, vc_wave.cpu(), sr)

    return output_file


@torch.no_grad()
def main(args: argparse.Namespace) -> None:
    """Main function for voice conversion inference.

    Performs voice conversion from source audio to target speaker's voice.
    Supports both non-F0 and F0-conditioned models with various options
    for pitch adjustment and length regulation.

    Args:
        args: Command line arguments containing:
            - source: Path to source audio file or directory.
            - target: Path to target reference audio file.
            - output: Output directory for converted audio.
            - diffusion_steps: Number of diffusion steps.
            - length_adjust: Factor to adjust output length.
            - inference_cfg_rate: Classifier-free guidance rate.
            - f0_condition: Whether to use F0 conditioning.
            - auto_f0_adjust: Whether to automatically adjust F0.
            - semi_tone_shift: Semitone shift for pitch adjustment.
            - checkpoint: Optional path to model checkpoint.
            - config: Optional path to model config.
            - fp16: Whether to use half precision.

    Returns:
        None. Saves the converted audio to the specified output directory.
    """
    # Load models once
    print("Loading models...")
    model, semantic_fn, f0_fn, vocoder_fn, campplus_model, mel_fn, mel_fn_args = load_models(args)

    # Find all source files
    try:
        source_files = find_audio_files(args.source)
        print(f"Found {len(source_files)} audio files to process")
    except ValueError as e:
        print(f"Error: {e}")
        return

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Process statistics
    successful = 0
    failed = 0
    failed_files = []

    # Process each file
    for source_file in tqdm(source_files, desc="Processing files"):
        try:
            output_file = process_single_file(
                source_file=source_file,
                target_file=args.target,
                output_dir=args.output,
                model=model,
                semantic_fn=semantic_fn,
                f0_fn=f0_fn,
                vocoder_fn=vocoder_fn,
                campplus_model=campplus_model,
                mel_fn=mel_fn,
                mel_fn_args=mel_fn_args,
                args=args,
            )
            successful += 1
            print(f"✓ Processed: {source_file} -> {output_file}")
        except Exception as e:
            failed += 1
            failed_files.append((source_file, str(e)))
            print(f"✗ Failed: {source_file} - {e}")

    # Print summary
    print("\n" + "=" * 50)
    print("Processing complete!")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")

    if failed_files:
        print("\nFailed files:")
        for file, error in failed_files:
            print(f"  - {file}: {error}")

    # Calculate and print RTF if single file
    if len(source_files) == 1 and successful == 1:
        print("\nNote: RTF calculation is included in single file processing.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Perform voice conversion using Seed-VC model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source",
        type=str,
        default="./assets/examples/source/source_s1.wav",
        help="Path to source audio file or directory containing audio files",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="./assets/examples/reference/s1p1.wav",
        help="Path to target reference audio file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./reconstructed",
        help="Output directory for converted audio files",
    )
    parser.add_argument(
        "--diffusion-steps",
        type=int,
        default=30,
        help="Number of diffusion steps for generation",
    )
    parser.add_argument(
        "--length-adjust",
        type=float,
        default=1.0,
        help="Factor to adjust output length (1.0 = same as source)",
    )
    parser.add_argument(
        "--inference-cfg-rate",
        type=float,
        default=0.7,
        help="Classifier-free guidance rate for inference",
    )
    parser.add_argument(
        "--f0-condition",
        type=str2bool,
        default=False,
        help="Whether to use F0 conditioning",
    )
    parser.add_argument(
        "--auto-f0-adjust",
        type=str2bool,
        default=False,
        help="Whether to automatically adjust F0 to match target speaker",
    )
    parser.add_argument(
        "--semi-tone-shift",
        type=int,
        default=0,
        help="Pitch shift in semitones (positive for higher, negative for lower)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to the checkpoint file (optional, uses default if not specified)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to the config file (optional, uses default if not specified)",
    )
    parser.add_argument(
        "--fp16",
        type=str2bool,
        default=True,
        help="Whether to use half precision (FP16) for faster inference",
    )
    args = parser.parse_args()
    main(args)

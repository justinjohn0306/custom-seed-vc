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

import os

os.environ["HF_HUB_CACHE"] = "./checkpoints/hf_cache"
import argparse
from typing import Any, Callable, Dict, Generator, Optional, Tuple, Union

import gradio as gr
import librosa
import numpy as np
import torch
import torchaudio
import yaml
from pydub import AudioSegment

from seed_vc.hf_utils import load_custom_model_from_hf
from seed_vc.modules.commons import build_model, load_checkpoint, recursive_munch, str2bool

# Load model and configuration

fp16: bool = False
device: Optional[torch.device] = None


def load_models(
    args: argparse.Namespace,
) -> Tuple[
    Dict[str, Any],
    Callable[[torch.Tensor], torch.Tensor],
    Union[Callable, torch.nn.Module],
    torch.nn.Module,
    Callable[[torch.Tensor], torch.Tensor],
    Dict[str, Any],
    Callable[[np.ndarray, float], np.ndarray],
]:
    """Load all models and configurations required for voice conversion.

    Args:
        args: Command line arguments containing model paths and configurations.

    Returns:
        A tuple containing:
            - model: The main DiT model dictionary
            - semantic_fn: Function to extract semantic features from audio
            - vocoder_fn: Vocoder model or function for audio synthesis
            - campplus_model: CAMPPlus model for speaker embedding
            - to_mel: Function to convert audio to mel spectrogram
            - mel_fn_args: Arguments for mel spectrogram computation
            - f0_fn: Function to extract F0 (fundamental frequency) from audio
    """
    global sr, hop_length, fp16
    fp16 = args.fp16
    print(f"Using device: {device}")
    print(f"Using fp16: {fp16}")
    # f0 conditioned model
    if args.checkpoint is None or args.checkpoint == "":
        dit_checkpoint_path, dit_config_path = load_custom_model_from_hf(
            "Plachta/Seed-VC",
            "DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema_v2.pth",
            "config_dit_mel_seed_uvit_whisper_base_f0_44k.yml",
        )
    else:
        print(f"Using custom checkpoint: {args.checkpoint}")
        dit_checkpoint_path = args.checkpoint
        dit_config_path = args.config
    config = yaml.safe_load(open(dit_config_path, "r"))
    model_params = recursive_munch(config["model_params"])
    model_params.dit_type = "DiT"
    model = build_model(model_params, stage="DiT")
    hop_length = config["preprocess_params"]["spect_params"]["hop_length"]
    sr = config["preprocess_params"]["sr"]

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

    campplus_ckpt_path = load_custom_model_from_hf(
        "funasr/campplus",
        "campplus_cn_common.bin",
        config_filename=None,
    )
    campplus_model = CAMPPlus(feat_dim=80, embedding_size=192)
    campplus_model.load_state_dict(torch.load(campplus_ckpt_path, map_location="cpu"))
    campplus_model.eval()
    campplus_model.to(device)

    vocoder_type = model_params.vocoder.type

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

    speech_tokenizer_type = model_params.speech_tokenizer.type
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
    mel_fn_args = {
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
    # f0 extractor
    from seed_vc.modules.rmvpe import RMVPE

    model_path = load_custom_model_from_hf("lj1995/VoiceConversionWebUI", "rmvpe.pt", None)
    rmvpe = RMVPE(model_path, is_half=False, device=device)
    f0_fn = rmvpe.infer_from_audio

    return (
        model,
        semantic_fn,
        vocoder_fn,
        campplus_model,
        to_mel,
        mel_fn_args,
        f0_fn,
    )


def adjust_f0_semitones(
    f0_sequence: Union[torch.Tensor, np.ndarray],
    n_semitones: int,
) -> Union[torch.Tensor, np.ndarray]:
    """Adjust F0 sequence by a given number of semitones.

    Args:
        f0_sequence: Input F0 sequence as tensor or numpy array.
        n_semitones: Number of semitones to shift (positive = higher pitch, negative = lower pitch).

    Returns:
        Pitch-shifted F0 sequence with the same type as input.
    """
    factor = 2 ** (n_semitones / 12)
    return f0_sequence * factor


def crossfade(chunk1: np.ndarray, chunk2: np.ndarray, overlap: int) -> np.ndarray:
    """Apply crossfade between two audio chunks.

    Args:
        chunk1: First audio chunk (will fade out).
        chunk2: Second audio chunk (will fade in).
        overlap: Number of samples to overlap.

    Returns:
        The second chunk with crossfade applied at the beginning.
    """
    fade_out = np.cos(np.linspace(0, np.pi / 2, overlap)) ** 2
    fade_in = np.cos(np.linspace(np.pi / 2, 0, overlap)) ** 2
    chunk2[:overlap] = chunk2[:overlap] * fade_in + chunk1[-overlap:] * fade_out
    return chunk2


# streaming and chunk processing related params
# max_context_window = sr // hop_length * 30
# overlap_frame_len = 16
# overlap_wave_len = overlap_frame_len * hop_length
bitrate: str = "320k"

model_f0: Optional[Dict[str, Any]] = None
semantic_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None
vocoder_fn: Optional[Union[Callable, torch.nn.Module]] = None
campplus_model: Optional[torch.nn.Module] = None
to_mel_f0: Optional[Callable[[torch.Tensor], torch.Tensor]] = None
mel_fn_args: Optional[Dict[str, Any]] = None
f0_fn: Optional[Callable[[np.ndarray, float], np.ndarray]] = None
overlap_wave_len: Optional[int] = None
max_context_window: Optional[int] = None
sr: Optional[int] = None
hop_length: Optional[int] = None
overlap_frame_len: int = 16


@torch.no_grad()
@torch.inference_mode()
def voice_conversion(
    source: str,
    target: str,
    diffusion_steps: int,
    length_adjust: float,
    inference_cfg_rate: float,
    auto_f0_adjust: bool,
    pitch_shift: int,
) -> Generator[Tuple[bytes, Optional[Tuple[int, np.ndarray]]], None, None]:
    """Perform voice conversion from source to target voice.

    This function performs streaming voice conversion, yielding audio chunks as they are generated.

    Args:
        source: Path to source audio file.
        target: Path to target/reference audio file.
        diffusion_steps: Number of diffusion steps for generation (higher = better quality but slower).
        length_adjust: Factor to adjust output length (<1.0 speeds up, >1.0 slows down).
        inference_cfg_rate: Classifier-free guidance rate (0.0-1.0).
        auto_f0_adjust: Whether to automatically adjust F0 to match target voice.
        pitch_shift: Pitch shift in semitones (-24 to +24).

    Yields:
        Tuples of (mp3_bytes, optional_full_audio) where:
            - mp3_bytes: Encoded MP3 audio chunk
            - optional_full_audio: None for intermediate chunks, (sample_rate, audio_array) for final chunk
    """
    inference_module = model_f0
    mel_fn = to_mel_f0
    # Load audio
    source_audio = librosa.load(source, sr=sr)[0]
    ref_audio = librosa.load(target, sr=sr)[0]

    # Process audio
    source_audio = torch.tensor(source_audio).unsqueeze(0).float().to(device)
    ref_audio = torch.tensor(ref_audio[: sr * 25]).unsqueeze(0).float().to(device)

    # Resample
    ref_waves_16k = torchaudio.functional.resample(ref_audio, sr, 16000)
    converted_waves_16k = torchaudio.functional.resample(source_audio, sr, 16000)
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

    ori_waves_16k = torchaudio.functional.resample(ref_audio, sr, 16000)
    S_ori = semantic_fn(ori_waves_16k)

    mel = mel_fn(source_audio.to(device).float())
    mel2 = mel_fn(ref_audio.to(device).float())

    target_lengths = torch.LongTensor([int(mel.size(2) * length_adjust)]).to(mel.device)
    target2_lengths = torch.LongTensor([mel2.size(2)]).to(mel2.device)

    feat2 = torchaudio.compliance.kaldi.fbank(
        ref_waves_16k,
        num_mel_bins=80,
        dither=0,
        sample_frequency=16000,
    )
    feat2 = feat2 - feat2.mean(dim=0, keepdim=True)
    style2 = campplus_model(feat2.unsqueeze(0))

    F0_ori = f0_fn(ref_waves_16k[0], thred=0.03)
    F0_alt = f0_fn(converted_waves_16k[0], thred=0.03)

    if device.type == "mps":
        F0_ori = torch.from_numpy(F0_ori).float().to(device)[None]
        F0_alt = torch.from_numpy(F0_alt).float().to(device)[None]
    else:
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
        shifted_f0_alt[F0_alt > 1] = adjust_f0_semitones(shifted_f0_alt[F0_alt > 1], pitch_shift)

    # Length regulation
    cond, _, codes, commitment_loss, codebook_loss = inference_module.length_regulator(
        S_alt,
        ylens=target_lengths,
        n_quantizers=3,
        f0=shifted_f0_alt,
    )
    prompt_condition, _, codes, commitment_loss, codebook_loss = inference_module.length_regulator(
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
        vc_wave = vocoder_fn(vc_target.float()).squeeze().cpu()
        if vc_wave.ndim == 1:
            vc_wave = vc_wave.unsqueeze(0)
        if processed_frames == 0:
            if is_last_chunk:
                output_wave = vc_wave[0].cpu().numpy()
                generated_wave_chunks.append(output_wave)
                output_wave = (output_wave * 32768.0).astype(np.int16)
                mp3_bytes = (
                    AudioSegment(
                        output_wave.tobytes(),
                        frame_rate=sr,
                        sample_width=output_wave.dtype.itemsize,
                        channels=1,
                    )
                    .export(format="mp3", bitrate=bitrate)
                    .read()
                )
                yield mp3_bytes, (sr, np.concatenate(generated_wave_chunks))
                break
            output_wave = vc_wave[0, :-overlap_wave_len].cpu().numpy()
            generated_wave_chunks.append(output_wave)
            previous_chunk = vc_wave[0, -overlap_wave_len:]
            processed_frames += vc_target.size(2) - overlap_frame_len
            output_wave = (output_wave * 32768.0).astype(np.int16)
            mp3_bytes = (
                AudioSegment(
                    output_wave.tobytes(),
                    frame_rate=sr,
                    sample_width=output_wave.dtype.itemsize,
                    channels=1,
                )
                .export(format="mp3", bitrate=bitrate)
                .read()
            )
            yield mp3_bytes, None
        elif is_last_chunk:
            output_wave = crossfade(
                previous_chunk.cpu().numpy(),
                vc_wave[0].cpu().numpy(),
                overlap_wave_len,
            )
            generated_wave_chunks.append(output_wave)
            processed_frames += vc_target.size(2) - overlap_frame_len
            output_wave = (output_wave * 32768.0).astype(np.int16)
            mp3_bytes = (
                AudioSegment(
                    output_wave.tobytes(),
                    frame_rate=sr,
                    sample_width=output_wave.dtype.itemsize,
                    channels=1,
                )
                .export(format="mp3", bitrate=bitrate)
                .read()
            )
            yield mp3_bytes, (sr, np.concatenate(generated_wave_chunks))
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
            output_wave = (output_wave * 32768.0).astype(np.int16)
            mp3_bytes = (
                AudioSegment(
                    output_wave.tobytes(),
                    frame_rate=sr,
                    sample_width=output_wave.dtype.itemsize,
                    channels=1,
                )
                .export(format="mp3", bitrate=bitrate)
                .read()
            )
            yield mp3_bytes, None


def main(args: argparse.Namespace) -> None:
    """Main function to set up and launch the Gradio interface.

    Args:
        args: Command line arguments for configuration.
    """
    global model_f0, semantic_fn, vocoder_fn, campplus_model, to_mel_f0, mel_fn_args, f0_fn
    global overlap_wave_len, max_context_window, sr, hop_length
    model_f0, semantic_fn, vocoder_fn, campplus_model, to_mel_f0, mel_fn_args, f0_fn = load_models(
        args,
    )
    # streaming and chunk processing related params
    max_context_window = sr // hop_length * 30
    overlap_wave_len = overlap_frame_len * hop_length
    description = (
        "Zero-shot voice conversion with in-context learning. For local deployment please check [GitHub repository](https://github.com/Plachtaa/seed-vc) "
        "for details and updates.<br>Note that any reference audio will be forcefully clipped to 25s if beyond this length.<br> "
        "If total duration of source and reference audio exceeds 30s, source audio will be processed in chunks.<br> "
        "无需训练的 zero-shot 语音/歌声转换模型，若需本地部署查看[GitHub页面](https://github.com/Plachtaa/seed-vc)<br>"
        "请注意，参考音频若超过 25 秒，则会被自动裁剪至此长度。<br>若源音频和参考音频的总时长超过 30 秒，源音频将被分段处理。"
    )
    inputs = [
        gr.Audio(type="filepath", label="Source Audio / 源音频"),
        gr.Audio(type="filepath", label="Reference Audio / 参考音频"),
        gr.Slider(
            minimum=1,
            maximum=200,
            value=10,
            step=1,
            label="Diffusion Steps / 扩散步数",
            info="10 by default, 50~100 for best quality / 默认为 10，50~100 为最佳质量",
        ),
        gr.Slider(
            minimum=0.5,
            maximum=2.0,
            step=0.1,
            value=1.0,
            label="Length Adjust / 长度调整",
            info="<1.0 for speed-up speech, >1.0 for slow-down speech / <1.0 加速语速，>1.0 减慢语速",
        ),
        gr.Slider(
            minimum=0.0,
            maximum=1.0,
            step=0.1,
            value=0.7,
            label="Inference CFG Rate",
            info="has subtle influence / 有微小影响",
        ),
        gr.Checkbox(
            label="Auto F0 adjust / 自动F0调整",
            value=True,
            info="Roughly adjust F0 to match target voice. Only works when F0 conditioned model is used. / 粗略调整 F0 以匹配目标音色，仅在勾选 '启用F0输入' 时生效",
        ),
        gr.Slider(
            label="Pitch shift / 音调变换",
            minimum=-24,
            maximum=24,
            step=1,
            value=0,
            info="Pitch shift in semitones, only works when F0 conditioned model is used / 半音数的音高变换，仅在勾选 '启用F0输入' 时生效",
        ),
    ]

    examples = [
        [
            "assets/examples/source/yae_0.wav",
            "assets/examples/reference/dingzhen_0.wav",
            25,
            1.0,
            0.7,
            True,
            0,
        ],
        [
            "assets/examples/source/jay_0.wav",
            "assets/examples/reference/azuma_0.wav",
            25,
            1.0,
            0.7,
            True,
            0,
        ],
        [
            "assets/examples/source/Wiz Khalifa,Charlie Puth - See You Again [vocals]_[cut_28sec].wav",
            "assets/examples/reference/teio_0.wav",
            50,
            1.0,
            0.7,
            False,
            0,
        ],
        [
            "assets/examples/source/TECHNOPOLIS - 2085 [vocals]_[cut_14sec].wav",
            "assets/examples/reference/trump_0.wav",
            50,
            1.0,
            0.7,
            False,
            -12,
        ],
    ]

    outputs = [
        gr.Audio(label="Stream Output Audio / 流式输出", streaming=True, format="mp3"),
        gr.Audio(label="Full Output Audio / 完整输出", streaming=False, format="wav"),
    ]

    gr.Interface(
        fn=voice_conversion,
        description=description,
        inputs=inputs,
        outputs=outputs,
        title="Seed Voice Conversion",
        examples=examples,
        cache_examples=False,
    ).launch(
        share=args.share,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, help="Path to the checkpoint file", default=None)
    parser.add_argument("--config", type=str, help="Path to the config file", default=None)
    parser.add_argument(
        "--share",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="Whether to share the app",
    )
    parser.add_argument(
        "--fp16",
        type=str2bool,
        nargs="?",
        const=True,
        help="Whether to use fp16",
        default=True,
    )
    parser.add_argument("--gpu", type=int, help="Which GPU id to use", default=0)
    args = parser.parse_args()
    cuda_target = f"cuda:{args.gpu}" if args.gpu else "cuda"

    if torch.cuda.is_available():
        device = torch.device(cuda_target)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    main(args)

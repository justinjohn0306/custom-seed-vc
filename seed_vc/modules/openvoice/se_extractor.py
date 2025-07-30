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

"""Speaker embedding extractor for OpenVoice."""

import base64
import hashlib
import os
from typing import Optional, Tuple

import librosa
import numpy as np
from faster_whisper import WhisperModel
from pydub import AudioSegment

model_size: str = "medium"
# Run on GPU with FP16
model: Optional[WhisperModel] = None


def split_audio_whisper(audio_path: str, audio_name: str, target_dir: str = "processed") -> str:
    """Split audio into segments using Whisper for speech detection.

    Args:
        audio_path: Path to the input audio file.
        audio_name: Name for the output directory.
        target_dir: Base directory for processed files.

    Returns:
        Path to the folder containing audio segments.
    """
    global model
    if model is None:
        model = WhisperModel(model_size, device="cuda", compute_type="float16")
    audio = AudioSegment.from_file(audio_path)
    max_len = len(audio)

    target_folder = os.path.join(target_dir, audio_name)

    segments, info = model.transcribe(audio_path, beam_size=5, word_timestamps=True)
    segments = list(segments)

    # create directory
    os.makedirs(target_folder, exist_ok=True)
    wavs_folder = os.path.join(target_folder, "wavs")
    os.makedirs(wavs_folder, exist_ok=True)

    # segments
    s_ind = 0
    start_time = None

    for k, w in enumerate(segments):
        # process with the time
        if k == 0:
            start_time = max(0, w.start)

        end_time = w.end

        # clean text
        text = w.text.replace("...", "")

        # left 0.08s for each audios
        audio_seg = audio[int(start_time * 1000) : min(max_len, int(end_time * 1000) + 80)]

        # segment file name
        fname = f"{audio_name}_seg{s_ind}.wav"

        # filter out the segment shorter than 1.5s and longer than 20s
        save = (
            audio_seg.duration_seconds > 1.5
            and audio_seg.duration_seconds < 20.0
            and len(text) >= 2
            and len(text) < 200
        )

        if save:
            output_file = os.path.join(wavs_folder, fname)
            audio_seg.export(output_file, format="wav")

        if k < len(segments) - 1:
            start_time = max(0, segments[k + 1].start - 0.08)

        s_ind = s_ind + 1
    return wavs_folder


def split_audio_vad(
    audio_path: str,
    audio_name: str,
    target_dir: str,
    split_seconds: float = 10.0,
) -> str:
    """Split audio into segments using Voice Activity Detection.

    Args:
        audio_path: Path to the input audio file.
        audio_name: Name for the output directory.
        target_dir: Base directory for processed files.
        split_seconds: Target duration for each segment.

    Returns:
        Path to the folder containing audio segments.
    """
    from whisper_timestamped.transcribe import (  # delayed import to fix F821
        get_audio_tensor,
        get_vad_segments,
    )

    SAMPLE_RATE = 16000
    audio_vad = get_audio_tensor(audio_path)
    segments = get_vad_segments(
        audio_vad,
        output_sample=True,
        min_speech_duration=0.1,
        min_silence_duration=1,
        method="silero",
    )
    segments = [(seg["start"], seg["end"]) for seg in segments]
    segments = [(float(s) / SAMPLE_RATE, float(e) / SAMPLE_RATE) for s, e in segments]
    print(segments)
    audio_active = AudioSegment.silent(duration=0)
    audio = AudioSegment.from_file(audio_path)

    for start_time, end_time in segments:
        audio_active += audio[int(start_time * 1000) : int(end_time * 1000)]

    audio_dur = audio_active.duration_seconds
    print(f"after vad: dur = {audio_dur}")
    target_folder = os.path.join(target_dir, audio_name)
    wavs_folder = os.path.join(target_folder, "wavs")
    os.makedirs(wavs_folder, exist_ok=True)
    start_time = 0.0
    count = 0
    num_splits = int(np.round(audio_dur / split_seconds))
    assert num_splits > 0, "input audio is too short"
    interval = audio_dur / num_splits

    for i in range(num_splits):
        end_time = min(start_time + interval, audio_dur)
        if i == num_splits - 1:
            end_time = audio_dur
        output_file = f"{wavs_folder}/{audio_name}_seg{count}.wav"
        audio_seg = audio_active[int(start_time * 1000) : int(end_time * 1000)]
        audio_seg.export(output_file, format="wav")
        start_time = end_time
        count += 1
    return wavs_folder


def hash_numpy_array(audio_path: str) -> str:
    """Generate a hash for audio file content.

    Args:
        audio_path: Path to the audio file.

    Returns:
        Base64-encoded hash string (first 16 characters).
    """
    array, _ = librosa.load(audio_path, sr=None, mono=True)
    # Convert the array to bytes
    array_bytes = array.tobytes()
    # Calculate the hash of the array bytes
    hash_object = hashlib.sha256(array_bytes)
    hash_value = hash_object.digest()
    # Convert the hash value to base64
    base64_value = base64.b64encode(hash_value)
    return base64_value.decode("utf-8")[:16].replace("/", "_^")


def get_se(audio_path: str, vc_model, target_dir: str = "processed", vad: bool = True) -> Tuple:
    """Extract speaker embedding from audio file.

    Args:
        audio_path: Path to the input audio file.
        vc_model: Voice conversion model instance.
        target_dir: Directory for saving processed files.
        vad: Whether to use VAD for audio splitting.

    Returns:
        Tuple of (speaker embedding, audio name).
    """
    version = vc_model.version
    print("OpenVoice version:", version)

    audio_name = (
        f"{os.path.basename(audio_path).rsplit('.', 1)[0]}_{version}_{hash_numpy_array(audio_path)}"
    )
    se_path = os.path.join(target_dir, audio_name, "se.pth")

    # if os.path.isfile(se_path):
    #     se = torch.load(se_path).to(device)
    #     return se, audio_name
    # if os.path.isdir(audio_path):
    #     wavs_folder = audio_path

    # if vad:
    #     wavs_folder = split_audio_vad(
    #         audio_path,
    #         target_dir=target_dir,
    #         audio_name=audio_name,
    #     )
    # else:
    #     wavs_folder = split_audio_whisper(
    #         audio_path,
    #         target_dir=target_dir,
    #         audio_name=audio_name,
    #     )

    # audio_segs = glob(f'{wavs_folder}/*.wav')
    # if len(audio_segs) == 0:
    #     raise NotImplementedError('No audio segments found!')

    return vc_model.extract_se([audio_path], se_save_path=se_path), audio_name

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
from typing import Any, Generator, Optional, Tuple

import gradio as gr
import torch
import yaml

# Set up device and torch configurations
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

dtype = torch.float16

# Global variables to store model instances
vc_wrapper_v1: Optional[Any] = None
vc_wrapper_v2: Optional[Any] = None


def load_v2_models(args: argparse.Namespace) -> Any:
    """Load V2 models and configure them.

    Args:
        args: Command line arguments containing compile flag.

    Returns:
        Configured V2 voice conversion wrapper.
    """
    from hydra.utils import instantiate
    from omegaconf import DictConfig

    cfg = DictConfig(yaml.safe_load(open("configs/v2/vc_wrapper.yaml", "r")))
    vc_wrapper = instantiate(cfg)
    vc_wrapper.load_checkpoints()
    vc_wrapper.to(device)
    vc_wrapper.eval()

    vc_wrapper.setup_ar_caches(max_batch_size=1, max_seq_len=4096, dtype=dtype, device=device)

    if args.compile:
        print("Compiling model with torch.compile...")
        torch._inductor.config.coordinate_descent_tuning = True
        torch._inductor.config.triton.unique_kernel_names = True

        if hasattr(torch._inductor.config, "fx_graph_cache"):
            # Experimental feature to reduce compilation times, will be on by default in future
            torch._inductor.config.fx_graph_cache = True
        vc_wrapper.compile_ar()
        # vc_wrapper.compile_cfm()

    return vc_wrapper


# Wrapper functions for GPU decoration
def convert_voice_v1_wrapper(
    source_audio_path: str,
    target_audio_path: str,
    diffusion_steps: int = 10,
    length_adjust: float = 1.0,
    inference_cfg_rate: float = 0.7,
    f0_condition: bool = False,
    auto_f0_adjust: bool = True,
    pitch_shift: int = 0,
    stream_output: bool = True,
) -> Generator[Tuple[int, bytes], None, None]:
    """Wrapper function for vc_wrapper.convert_voice that can be decorated with @spaces.GPU.

    Args:
        source_audio_path: Path to source audio file.
        target_audio_path: Path to target/reference audio file.
        diffusion_steps: Number of diffusion steps for quality control.
        length_adjust: Speech speed adjustment factor.
        inference_cfg_rate: Inference configuration rate.
        f0_condition: Whether to use F0 conditioned model.
        auto_f0_adjust: Whether to auto-adjust F0 to match target.
        pitch_shift: Pitch shift in semitones.
        stream_output: Whether to stream the output.

    Yields:
        Tuple of sample rate and audio bytes.
    """
    global vc_wrapper_v1
    from seed_vc.wrapper import SeedVCWrapper

    if vc_wrapper_v1 is None:
        vc_wrapper_v1 = SeedVCWrapper()

    # Use yield from to properly handle the generator
    yield from vc_wrapper_v1.convert_voice(
        source=source_audio_path,
        target=target_audio_path,
        diffusion_steps=diffusion_steps,
        length_adjust=length_adjust,
        inference_cfg_rate=inference_cfg_rate,
        f0_condition=f0_condition,
        auto_f0_adjust=auto_f0_adjust,
        pitch_shift=pitch_shift,
        stream_output=stream_output,
    )


def convert_voice_v2_wrapper(
    source_audio_path: str,
    target_audio_path: str,
    diffusion_steps: int = 30,
    length_adjust: float = 1.0,
    intelligebility_cfg_rate: float = 0.7,
    similarity_cfg_rate: float = 0.7,
    top_p: float = 0.7,
    temperature: float = 0.7,
    repetition_penalty: float = 1.5,
    convert_style: bool = False,
    anonymization_only: bool = False,
    stream_output: bool = True,
) -> Generator[Tuple[int, bytes], None, None]:
    """Wrapper function for vc_wrapper.convert_voice_with_streaming that can be decorated with @spaces.GPU.

    Args:
        source_audio_path: Path to source audio file.
        target_audio_path: Path to target/reference audio file.
        diffusion_steps: Number of diffusion steps for quality control.
        length_adjust: Speech speed adjustment factor.
        intelligebility_cfg_rate: Controls pronunciation intelligibility.
        similarity_cfg_rate: Controls similarity to reference audio.
        top_p: AR model sampling top P.
        temperature: AR model sampling temperature.
        repetition_penalty: AR model sampling repetition penalty.
        convert_style: Whether to convert style/emotion/accent.
        anonymization_only: Whether to perform anonymization only.
        stream_output: Whether to stream the output.

    Yields:
        Tuple of sample rate and audio bytes.
    """
    global vc_wrapper_v2

    # Use yield from to properly handle the generator
    yield from vc_wrapper_v2.convert_voice_with_streaming(
        source_audio_path=source_audio_path,
        target_audio_path=target_audio_path,
        diffusion_steps=diffusion_steps,
        length_adjust=length_adjust,
        intelligebility_cfg_rate=intelligebility_cfg_rate,
        similarity_cfg_rate=similarity_cfg_rate,
        top_p=top_p,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        convert_style=convert_style,
        anonymization_only=anonymization_only,
        device=device,
        dtype=dtype,
        stream_output=stream_output,
    )


def create_v1_interface() -> gr.Interface:
    """Create Gradio interface for V1 voice conversion.

    Returns:
        Configured Gradio interface for V1.
    """
    # Set up Gradio interface
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
            label="Use F0 conditioned model / 启用F0输入",
            value=False,
            info="Must set to true for singing voice conversion / 歌声转换时必须勾选",
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
            False,
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
            True,
            0,
        ],
        [
            "assets/examples/source/Wiz Khalifa,Charlie Puth - See You Again [vocals]_[cut_28sec].wav",
            "assets/examples/reference/teio_0.wav",
            100,
            1.0,
            0.7,
            True,
            False,
            0,
        ],
        [
            "assets/examples/source/TECHNOPOLIS - 2085 [vocals]_[cut_14sec].wav",
            "assets/examples/reference/trump_0.wav",
            50,
            1.0,
            0.7,
            True,
            False,
            -12,
        ],
    ]

    outputs = [
        gr.Audio(label="Stream Output Audio / 流式输出", streaming=True, format="mp3"),
        gr.Audio(label="Full Output Audio / 完整输出", streaming=False, format="wav"),
    ]

    return gr.Interface(
        fn=convert_voice_v1_wrapper,
        description=description,
        inputs=inputs,
        outputs=outputs,
        title="Seed Voice Conversion V1 (Voice & Singing Voice Conversion)",
        examples=examples,
        cache_examples=False,
    )


def create_v2_interface() -> gr.Interface:
    """Create Gradio interface for V2 voice conversion.

    Returns:
        Configured Gradio interface for V2.
    """
    # Set up Gradio interface
    description = (
        "Zero-shot voice/style conversion with in-context learning. For local deployment please check [GitHub repository](https://github.com/Plachtaa/seed-vc) "
        "for details and updates.<br>Note that any reference audio will be forcefully clipped to 25s if beyond this length.<br> "
        "If total duration of source and reference audio exceeds 30s, source audio will be processed in chunks.<br> "
        "Please click the 'convert style/emotion/accent' checkbox to convert the style, emotion, or accent of the source audio, or else only timbre conversion will be performed.<br> "
        "Click the 'anonymization only' checkbox will ignore reference audio but convert source to an 'average voice' determined by model itself.<br> "
        "无需训练的 zero-shot 语音/口音转换模型，若需本地部署查看[GitHub页面](https://github.com/Plachtaa/seed-vc)<br>"
        "请注意，参考音频若超过 25 秒，则会被自动裁剪至此长度。<br>若源音频和参考音频的总时长超过 30 秒，源音频将被分段处理。"
        "<br>请勾选 'convert style/emotion/accent' 以转换源音频的风格、情感或口音，否则仅执行音色转换。<br>"
        "勾选 'anonymization only' 会无视参考音频而将源音频转换为某种由模型自身决定的 '平均音色'。<br>"
        "Credits to [Vevo](https://github.com/open-mmlab/Amphion/tree/main/models/vc/vevo)"
    )
    inputs = [
        gr.Audio(type="filepath", label="Source Audio / 源音频"),
        gr.Audio(type="filepath", label="Reference Audio / 参考音频"),
        gr.Slider(
            minimum=1,
            maximum=200,
            value=30,
            step=1,
            label="Diffusion Steps / 扩散步数",
            info="30 by default, 50~100 for best quality / 默认为 30，50~100 为最佳质量",
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
            value=0.0,
            label="Intelligibility CFG Rate",
            info="controls pronunciation intelligibility / 控制发音清晰度",
        ),
        gr.Slider(
            minimum=0.0,
            maximum=1.0,
            step=0.1,
            value=0.7,
            label="Similarity CFG Rate",
            info="controls similarity to reference audio / 控制与参考音频的相似度",
        ),
        gr.Slider(
            minimum=0.1,
            maximum=1.0,
            step=0.1,
            value=0.9,
            label="Top-p",
            info="AR model sampling top P",
        ),
        gr.Slider(
            minimum=0.1,
            maximum=2.0,
            step=0.1,
            value=1.0,
            label="Temperature",
            info="AR model sampling temperature",
        ),
        gr.Slider(
            minimum=1.0,
            maximum=3.0,
            step=0.1,
            value=1.0,
            label="Repetition Penalty",
            info="AR model sampling repetition penalty",
        ),
        gr.Checkbox(label="convert style/emotion/accent", value=False),
        gr.Checkbox(label="anonymization only", value=False),
    ]

    examples = [
        [
            "assets/examples/source/yae_0.wav",
            "assets/examples/reference/dingzhen_0.wav",
            50,
            1.0,
            0.0,
            0.7,
            0.9,
            1.0,
            1.0,
            False,
            False,
        ],
        [
            "assets/examples/source/jay_0.wav",
            "assets/examples/reference/azuma_0.wav",
            50,
            1.0,
            0.0,
            0.7,
            0.9,
            1.0,
            1.0,
            False,
            False,
        ],
    ]

    outputs = [
        gr.Audio(label="Stream Output Audio / 流式输出", streaming=True, format="mp3"),
        gr.Audio(label="Full Output Audio / 完整输出", streaming=False, format="wav"),
    ]

    return gr.Interface(
        fn=convert_voice_v2_wrapper,
        description=description,
        inputs=inputs,
        outputs=outputs,
        title="Seed Voice Conversion V2 (Voice & Style Conversion)",
        examples=examples,
        cache_examples=False,
    )


def main(args: argparse.Namespace) -> None:
    """Main function to set up and launch the Gradio application.

    Args:
        args: Command line arguments with flags for enabling V1/V2 and compilation.
    """
    global vc_wrapper_v1, vc_wrapper_v2
    # Create interfaces based on enabled versions
    interfaces = []

    # Load V2 models if enabled
    if args.enable_v2:
        print("Loading V2 models...")
        vc_wrapper_v2 = load_v2_models(args)
        v2_interface = create_v2_interface()
        interfaces.append(("V2 - Voice & Style Conversion", v2_interface))

    # Create V1 interface if enabled
    if args.enable_v1:
        print("Creating V1 interface...")
        v1_interface = create_v1_interface()
        interfaces.append(("V1 - Voice & Singing Voice Conversion", v1_interface))

    # Check if at least one version is enabled
    if not interfaces:
        print("Error: At least one version (V1 or V2) must be enabled.")
        return

    # Create tabs
    with gr.Blocks(title="Seed Voice Conversion") as demo:
        gr.Markdown("# Seed Voice Conversion")

        if len(interfaces) > 1:
            gr.Markdown(
                "Choose between V1 (Voice & Singing Voice Conversion) or V2 (Voice & Style Conversion)",
            )

            with gr.Tabs():
                for tab_name, interface in interfaces:
                    with gr.TabItem(tab_name):
                        interface.render()
        else:
            # If only one version is enabled, don't use tabs
            for _, interface in interfaces:
                interface.render()

    # Launch the combined interface
    demo.launch()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Compile the model using torch.compile",
    )
    parser.add_argument(
        "--enable-v1",
        action="store_true",
        help="Enable V1 (Voice & Singing Voice Conversion)",
    )
    parser.add_argument(
        "--enable-v2",
        action="store_true",
        help="Enable V2 (Voice & Style Conversion)",
    )
    args = parser.parse_args()
    main(args)

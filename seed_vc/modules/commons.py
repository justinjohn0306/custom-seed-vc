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

"""Common utilities and helper functions for Seed-VC."""

import argparse
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from munch import Munch
from torch.nn import functional as F


def str2bool(v: Union[bool, str]) -> bool:
    """Convert string to boolean value.

    Args:
        v: Value to convert. Can be boolean or string.

    Returns:
        Boolean representation of the input.

    Raises:
        argparse.ArgumentTypeError: If the string cannot be converted to boolean.
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


class AttrDict(dict):
    """Dictionary with attribute-style access."""

    def __init__(self, *args, **kwargs):
        """Initialize AttrDict with dictionary items accessible as attributes.

        Args:
            *args: Positional arguments passed to dict constructor.
            **kwargs: Keyword arguments passed to dict constructor.
        """
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def init_weights(m: torch.nn.Module, mean: float = 0.0, std: float = 0.01) -> None:
    """Initialize weights of convolutional layers.

    Args:
        m: PyTorch module to initialize.
        mean: Mean of the normal distribution for weight initialization.
        std: Standard deviation of the normal distribution.
    """
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def get_padding(kernel_size: int, dilation: int = 1) -> int:
    """Calculate padding size for 'same' convolution.

    Args:
        kernel_size: Size of the convolutional kernel.
        dilation: Dilation factor for the convolution.

    Returns:
        Padding size to maintain input dimensions.
    """
    return int((kernel_size * dilation - dilation) / 2)


def convert_pad_shape(pad_shape: List[List[int]]) -> List[int]:
    """Convert padding shape to PyTorch format.

    Args:
        pad_shape: Nested list of padding values.

    Returns:
        Flattened padding values in PyTorch format.
    """
    l = pad_shape[::-1]
    pad_shape = [item for sublist in l for item in sublist]
    return pad_shape


def intersperse(lst: List[Any], item: Any) -> List[Any]:
    """Intersperse an item between elements of a list.

    Args:
        lst: List to intersperse items into.
        item: Item to insert between list elements.

    Returns:
        New list with item interspersed between original elements.
    """
    result = [item] * (len(lst) * 2 + 1)
    result[1::2] = lst
    return result


def kl_divergence(
    m_p: torch.Tensor,
    logs_p: torch.Tensor,
    m_q: torch.Tensor,
    logs_q: torch.Tensor,
) -> torch.Tensor:
    """Calculate KL divergence between two Gaussian distributions.

    Computes KL(P||Q) where P and Q are Gaussian distributions.

    Args:
        m_p: Mean of distribution P.
        logs_p: Log standard deviation of distribution P.
        m_q: Mean of distribution Q.
        logs_q: Log standard deviation of distribution Q.

    Returns:
        KL divergence between P and Q.
    """
    kl = (logs_q - logs_p) - 0.5
    kl += 0.5 * (torch.exp(2.0 * logs_p) + ((m_p - m_q) ** 2)) * torch.exp(-2.0 * logs_q)
    return kl


def rand_gumbel(shape: Union[torch.Size, List[int], Tuple[int, ...]]) -> torch.Tensor:
    """Sample from the Gumbel distribution.

    Samples from Gumbel distribution while protecting from numerical overflows.

    Args:
        shape: Shape of the output tensor.

    Returns:
        Tensor of Gumbel-distributed random values.
    """
    uniform_samples = torch.rand(shape) * 0.99998 + 0.00001
    return -torch.log(-torch.log(uniform_samples))


def rand_gumbel_like(x: torch.Tensor) -> torch.Tensor:
    """Sample from Gumbel distribution with same shape as input.

    Args:
        x: Tensor to match shape and device.

    Returns:
        Gumbel-distributed tensor with same shape, dtype, and device as x.
    """
    g = rand_gumbel(x.size()).to(dtype=x.dtype, device=x.device)
    return g


def slice_segments(x: torch.Tensor, ids_str: torch.Tensor, segment_size: int = 4) -> torch.Tensor:
    """Extract segments from tensor based on start indices.

    Args:
        x: Input tensor of shape (batch, channels, time).
        ids_str: Start indices for each batch element.
        segment_size: Length of segments to extract.

    Returns:
        Tensor of extracted segments with shape (batch, channels, segment_size).
    """
    ret = torch.zeros_like(x[:, :, :segment_size])
    for i in range(x.size(0)):
        idx_str = ids_str[i]
        idx_end = idx_str + segment_size
        ret[i] = x[i, :, idx_str:idx_end]
    return ret


def slice_segments_audio(
    x: torch.Tensor,
    ids_str: torch.Tensor,
    segment_size: int = 4,
) -> torch.Tensor:
    """Extract audio segments from tensor based on start indices.

    Args:
        x: Input audio tensor of shape (batch, time).
        ids_str: Start indices for each batch element.
        segment_size: Length of segments to extract.

    Returns:
        Tensor of extracted audio segments with shape (batch, segment_size).
    """
    ret = torch.zeros_like(x[:, :segment_size])
    for i in range(x.size(0)):
        idx_str = ids_str[i]
        idx_end = idx_str + segment_size
        ret[i] = x[i, idx_str:idx_end]
    return ret


def rand_slice_segments(
    x: torch.Tensor,
    x_lengths: Optional[torch.Tensor] = None,
    segment_size: int = 4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Randomly slice segments from tensor.

    Args:
        x: Input tensor of shape (batch, channels, time).
        x_lengths: Length of each sequence in batch. If None, uses full length.
        segment_size: Length of segments to extract.

    Returns:
        Tuple of (sliced segments, start indices used).
    """
    b, d, t = x.size()
    if x_lengths is None:
        x_lengths = t
    ids_str_max = x_lengths - segment_size + 1
    ids_str = ((torch.rand([b]).to(device=x.device) * ids_str_max).clip(0)).to(dtype=torch.long)
    ret = slice_segments(x, ids_str, segment_size)
    return ret, ids_str


def get_timing_signal_1d(
    length: int,
    channels: int,
    min_timescale: float = 1.0,
    max_timescale: float = 1.0e4,
) -> torch.Tensor:
    """Generate sinusoidal positional embeddings.

    Args:
        length: Sequence length.
        channels: Number of channels (must be even).
        min_timescale: Minimum timescale for sinusoidal frequencies.
        max_timescale: Maximum timescale for sinusoidal frequencies.

    Returns:
        Positional embedding tensor of shape (1, channels, length).
    """
    position = torch.arange(length, dtype=torch.float)
    num_timescales = channels // 2
    log_timescale_increment = math.log(float(max_timescale) / float(min_timescale)) / (
        num_timescales - 1
    )
    inv_timescales = min_timescale * torch.exp(
        torch.arange(num_timescales, dtype=torch.float) * -log_timescale_increment,
    )
    scaled_time = position.unsqueeze(0) * inv_timescales.unsqueeze(1)
    signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], 0)
    signal = F.pad(signal, [0, 0, 0, channels % 2])
    signal = signal.view(1, channels, length)
    return signal


def add_timing_signal_1d(
    x: torch.Tensor,
    min_timescale: float = 1.0,
    max_timescale: float = 1.0e4,
) -> torch.Tensor:
    """Add sinusoidal positional embeddings to input tensor.

    Args:
        x: Input tensor of shape (batch, channels, length).
        min_timescale: Minimum timescale for sinusoidal frequencies.
        max_timescale: Maximum timescale for sinusoidal frequencies.

    Returns:
        Input tensor with added positional embeddings.
    """
    b, channels, length = x.size()
    signal = get_timing_signal_1d(length, channels, min_timescale, max_timescale)
    return x + signal.to(dtype=x.dtype, device=x.device)


def cat_timing_signal_1d(
    x: torch.Tensor,
    min_timescale: float = 1.0,
    max_timescale: float = 1.0e4,
    axis: int = 1,
) -> torch.Tensor:
    """Concatenate sinusoidal positional embeddings to input tensor.

    Args:
        x: Input tensor of shape (batch, channels, length).
        min_timescale: Minimum timescale for sinusoidal frequencies.
        max_timescale: Maximum timescale for sinusoidal frequencies.
        axis: Axis along which to concatenate.

    Returns:
        Tensor with concatenated positional embeddings.
    """
    b, channels, length = x.size()
    signal = get_timing_signal_1d(length, channels, min_timescale, max_timescale)
    return torch.cat([x, signal.to(dtype=x.dtype, device=x.device)], axis)


def subsequent_mask(length: int) -> torch.Tensor:
    """Generate a mask for autoregressive models.

    Creates a lower triangular mask to prevent attending to future positions.

    Args:
        length: Sequence length.

    Returns:
        Mask tensor of shape (1, 1, length, length).
    """
    mask = torch.tril(torch.ones(length, length)).unsqueeze(0).unsqueeze(0)
    return mask


@torch.jit.script
def fused_add_tanh_sigmoid_multiply(
    input_a: torch.Tensor,
    input_b: torch.Tensor,
    n_channels: torch.Tensor,
) -> torch.Tensor:
    """Fused operation for gated activation.

    Performs: tanh(input[:n_channels]) * sigmoid(input[n_channels:]).

    Args:
        input_a: First input tensor.
        input_b: Second input tensor to add.
        n_channels: Number of channels for tanh activation.

    Returns:
        Result of gated activation.
    """
    n_channels_int = n_channels[0]
    in_act = input_a + input_b
    t_act = torch.tanh(in_act[:, :n_channels_int, :])
    s_act = torch.sigmoid(in_act[:, n_channels_int:, :])
    acts = t_act * s_act
    return acts


def shift_1d(x: torch.Tensor) -> torch.Tensor:
    """Shift tensor by one position along time dimension.

    Args:
        x: Input tensor.

    Returns:
        Shifted tensor with same shape as input.
    """
    x = F.pad(x, convert_pad_shape([[0, 0], [0, 0], [1, 0]]))[:, :, :-1]
    return x


def sequence_mask(length: torch.Tensor, max_length: Optional[int] = None) -> torch.Tensor:
    """Generate sequence mask from lengths.

    Args:
        length: Tensor of sequence lengths.
        max_length: Maximum sequence length. If None, uses max of length tensor.

    Returns:
        Boolean mask tensor indicating valid positions.
    """
    if max_length is None:
        max_length = length.max()
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)


def avg_with_mask(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Compute masked average of tensor.

    Args:
        x: Input tensor to average.
        mask: Binary mask tensor (1 for valid, 0 for invalid).

    Returns:
        Averaged value considering only masked positions.

    Raises:
        AssertionError: If mask is not float dtype.
    """
    assert mask.dtype == torch.float, "Mask should be float"

    if mask.ndim == 2:
        mask = mask.unsqueeze(1)

    if mask.shape[1] == 1:
        mask = mask.expand_as(x)

    return (x * mask).sum() / mask.sum()


def generate_path(duration: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Generate alignment path from duration predictions.

    Args:
        duration: Duration predictions of shape (batch, 1, t_x).
        mask: Attention mask of shape (batch, 1, t_y, t_x).

    Returns:
        Alignment path tensor of shape (batch, 1, t_y, t_x).
    """
    b, _, t_y, t_x = mask.shape
    cum_duration = torch.cumsum(duration, -1)

    cum_duration_flat = cum_duration.view(b * t_x)
    path = sequence_mask(cum_duration_flat, t_y).to(mask.dtype)
    path = path.view(b, t_x, t_y)
    path = path - F.pad(path, convert_pad_shape([[0, 0], [1, 0], [0, 0]]))[:, :-1]
    path = path.unsqueeze(1).transpose(2, 3) * mask
    return path


def clip_grad_value_(
    parameters: Union[torch.Tensor, List[torch.Tensor]],
    clip_value: Optional[float],
    norm_type: float = 2,
) -> float:
    """Clip gradients by value and compute gradient norm.

    Args:
        parameters: Model parameters or list of parameters.
        clip_value: Maximum absolute value for gradients. If None, no clipping.
        norm_type: Type of norm to compute.

    Returns:
        Total gradient norm.
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    if clip_value is not None:
        clip_value = float(clip_value)

    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
        if clip_value is not None:
            p.grad.data.clamp_(min=-clip_value, max=clip_value)
    total_norm = total_norm ** (1.0 / norm_type)
    return total_norm


def log_norm(x: torch.Tensor, mean: float = -4, std: float = 4, dim: int = 2) -> torch.Tensor:
    """Apply log normalization to mel spectrogram.

    Converts normalized log mel to mel, computes norm, then takes log.

    Args:
        x: Input log mel spectrogram.
        mean: Mean for denormalization.
        std: Standard deviation for denormalization.
        dim: Dimension along which to compute norm.

    Returns:
        Log-normalized tensor.
    """
    x = torch.log(torch.exp(x * std + mean).norm(dim=dim))
    return x


def load_F0_models(path: Union[str, Path]) -> torch.nn.Module:
    """Load F0 (fundamental frequency) prediction model.

    Args:
        path: Path to model checkpoint file.

    Returns:
        Loaded JDCNet model for F0 prediction.
    """
    # load F0 model
    from .JDC.model import JDCNet

    F0_model = JDCNet(num_class=1, seq_len=192)
    params = torch.load(path, map_location="cpu")["net"]
    F0_model.load_state_dict(params)
    _ = F0_model.train()

    return F0_model


def modify_w2v_forward(self: Any, output_layer: int = 15) -> Any:
    """Modify wav2vec2 forward method to return intermediate layer outputs.

    Args:
        self: Wav2vec2 encoder instance.
        output_layer: Layer number to extract features from (1-based indexing).

    Returns:
        Modified forward function that extracts intermediate representations.
    """
    from transformers.modeling_outputs import BaseModelOutput

    def forward(
        hidden_states,
        attention_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        conv_attention_mask = attention_mask
        if attention_mask is not None:
            # make sure padded tokens output 0
            hidden_states = hidden_states.masked_fill(~attention_mask.bool().unsqueeze(-1), 0.0)

            # extend attention_mask
            attention_mask = 1.0 - attention_mask[:, None, None, :].to(dtype=hidden_states.dtype)
            attention_mask = attention_mask * torch.finfo(hidden_states.dtype).min
            attention_mask = attention_mask.expand(
                attention_mask.shape[0],
                1,
                attention_mask.shape[-1],
                attention_mask.shape[-1],
            )

        hidden_states = self.dropout(hidden_states)

        if self.embed_positions is not None:
            relative_position_embeddings = self.embed_positions(hidden_states)
        else:
            relative_position_embeddings = None

        deepspeed_zero3_is_enabled = False

        for i, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = torch.rand([])

            skip_the_layer = (
                True if self.training and (dropout_probability < self.config.layerdrop) else False
            )
            if not skip_the_layer or deepspeed_zero3_is_enabled:
                # under deepspeed zero3 all gpus must run in sync
                if self.gradient_checkpointing and self.training:
                    layer_outputs = self._gradient_checkpointing_func(
                        layer.__call__,
                        hidden_states,
                        attention_mask,
                        relative_position_embeddings,
                        output_attentions,
                        conv_attention_mask,
                    )
                else:
                    layer_outputs = layer(
                        hidden_states,
                        attention_mask=attention_mask,
                        relative_position_embeddings=relative_position_embeddings,
                        output_attentions=output_attentions,
                        conv_attention_mask=conv_attention_mask,
                    )
                hidden_states = layer_outputs[0]

            if skip_the_layer:
                layer_outputs = (None, None)

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

            if i == output_layer - 1:
                break

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None
            )
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )

    return forward


MATPLOTLIB_FLAG = False


def plot_spectrogram_to_numpy(spectrogram: torch.Tensor) -> np.ndarray:
    """Convert spectrogram tensor to numpy array visualization.

    Args:
        spectrogram: Spectrogram tensor to visualize.

    Returns:
        RGB numpy array of the spectrogram plot.
    """
    global MATPLOTLIB_FLAG
    if not MATPLOTLIB_FLAG:
        import logging

        import matplotlib

        matplotlib.use("Agg")
        MATPLOTLIB_FLAG = True
        mpl_logger = logging.getLogger("matplotlib")
        mpl_logger.setLevel(logging.WARNING)
    import matplotlib.pylab as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(10, 2))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower", interpolation="none")
    plt.colorbar(im, ax=ax)
    plt.xlabel("Frames")
    plt.ylabel("Channels")
    plt.tight_layout()

    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return data


def normalize_f0(f0_sequence: np.ndarray) -> np.ndarray:
    """Normalize F0 sequence using log scale and z-score normalization.

    Args:
        f0_sequence: F0 values in Hz. Unvoiced frames should be <= 0.

    Returns:
        Normalized F0 sequence with unvoiced frames marked as -1.
    """
    # Remove unvoiced frames (replace with -1)
    voiced_indices = np.where(f0_sequence > 0)[0]
    f0_voiced = f0_sequence[voiced_indices]

    # Convert to log scale
    log_f0 = np.log2(f0_voiced)

    # Calculate mean and standard deviation
    mean_f0 = np.mean(log_f0)
    std_f0 = np.std(log_f0)

    # Normalize the F0 sequence
    normalized_f0 = (log_f0 - mean_f0) / std_f0

    # Create the normalized F0 sequence with unvoiced frames
    normalized_sequence = np.zeros_like(f0_sequence)
    normalized_sequence[voiced_indices] = normalized_f0
    normalized_sequence[f0_sequence <= 0] = -1  # Assign -1 to unvoiced frames

    return normalized_sequence


def build_model(args: Union[Dict[str, Any], Munch], stage: str = "DiT") -> Munch:
    """Build model based on configuration and stage.

    Args:
        args: Model configuration object containing hyperparameters.
        stage: Model stage to build. Currently only supports "DiT".

    Returns:
        Munch object containing initialized model components.

    Raises:
        ValueError: If unknown stage is specified.
    """
    if stage == "DiT":
        from seed_vc.modules.flow_matching import CFM
        from seed_vc.modules.length_regulator import InterpolateRegulator

        length_regulator = InterpolateRegulator(
            channels=args.length_regulator.channels,
            sampling_ratios=args.length_regulator.sampling_ratios,
            is_discrete=args.length_regulator.is_discrete,
            in_channels=args.length_regulator.in_channels
            if hasattr(args.length_regulator, "in_channels")
            else None,
            codebook_size=args.length_regulator.content_codebook_size,
            f0_condition=args.length_regulator.f0_condition
            if hasattr(args.length_regulator, "f0_condition")
            else False,
            n_f0_bins=args.length_regulator.n_f0_bins
            if hasattr(args.length_regulator, "n_f0_bins")
            else 512,
        )
        cfm = CFM(args)
        nets = Munch(
            cfm=cfm,
            length_regulator=length_regulator,
        )
    else:
        raise ValueError(f"Unknown stage: {stage}")

    return nets


def load_checkpoint(
    model: Dict[str, torch.nn.Module],
    optimizer: Any,
    path: Union[str, Path],
    load_only_params: bool = True,
    ignore_modules: Optional[List[str]] = None,
    is_distributed: bool = False,
    load_ema: bool = False,
) -> Tuple[Dict[str, torch.nn.Module], Any, int, int]:
    """Load model checkpoint from file.

    Args:
        model: Dictionary of model components to load.
        optimizer: Optimizer instance to load state into.
        path: Path to checkpoint file.
        load_only_params: If True, only load model parameters, not optimizer state.
        ignore_modules: List of module names to skip loading.
        is_distributed: Whether model is distributed (DDP).
        load_ema: Whether to load exponential moving average weights.

    Returns:
        Tuple of (model dict, optimizer, epoch, iteration count).
    """
    if ignore_modules is None:
        ignore_modules = []
    state = torch.load(path, map_location="cpu")
    params = state["net"]
    if load_ema and "ema" in state:
        print("Loading EMA")
        for key in model:
            i = 0
            for param_name in params[key]:
                if "input_pos" in param_name:
                    continue
                assert params[key][param_name].shape == state["ema"][key][0][i].shape
                params[key][param_name] = state["ema"][key][0][i].clone()
                i += 1
    for key in model:
        if key in params and key not in ignore_modules:
            if not is_distributed:
                # Strip prefix of DDP (module.), create a new OrderedDict
                # that does not contain the prefix
                for k in list(params[key].keys()):
                    if k.startswith("module."):
                        params[key][k[len("module.") :]] = params[key][k]
                        del params[key][k]
            model_state_dict = model[key].state_dict()
            # 过滤出形状匹配的键值对
            filtered_state_dict = {
                k: v
                for k, v in params[key].items()
                if k in model_state_dict and v.shape == model_state_dict[k].shape
            }
            skipped_keys = set(params[key].keys()) - set(filtered_state_dict.keys())
            if skipped_keys:
                print(f"Warning: Skipped loading some keys due to shape mismatch: {skipped_keys}")
            print("%s loaded" % key)
            model[key].load_state_dict(filtered_state_dict, strict=False)
    _ = [model[key].eval() for key in model]

    if not load_only_params:
        epoch = state["epoch"] + 1
        iters = state["iters"]
        optimizer.load_state_dict(state["optimizer"])
        optimizer.load_scheduler_state_dict(state["scheduler"])

    else:
        epoch = 0
        iters = 0

    return model, optimizer, epoch, iters


def recursive_munch(d: Union[Dict[str, Any], List[Any], Any]) -> Union[Munch, List[Any], Any]:
    """Recursively convert dictionaries to Munch objects.

    Args:
        d: Dictionary, list, or any other value to convert.

    Returns:
        Munch object if input was dict, recursively processed list if input was list,
        or unchanged value otherwise.
    """
    if isinstance(d, dict):
        return Munch((k, recursive_munch(v)) for k, v in d.items())
    elif isinstance(d, list):
        return [recursive_munch(v) for v in d]
    else:
        return d

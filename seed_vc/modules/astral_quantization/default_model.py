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

"""Default model implementation for ASTRAL quantization."""

from typing import Optional, Tuple

import torch
from transformers import AutoModel, AutoTokenizer, Wav2Vec2FeatureExtractor


class AstralQuantizer(torch.nn.Module):
    """ASTRAL Quantizer for speech representation learning.

    Combines SSL model (e.g., Wav2Vec2, HuBERT) with encoder and quantizer
    for discrete speech representation.

    Attributes:
        encoder: Encoder module.
        quantizer: Quantizer module (e.g., BSQ).
        tokenizer_name: Name of tokenizer to load.
        tokenizer: Loaded tokenizer.
        ssl_model_name: Name of SSL model.
        ssl_output_layer: Which layer output to use from SSL model.
        ssl_feature_extractor: Feature extractor for SSL model.
        ssl_model: SSL model (if not skipped).
    """

    def __init__(
        self,
        tokenizer_name: str,
        ssl_model_name: str,
        ssl_output_layer: int,
        encoder: torch.nn.Module,
        quantizer: torch.nn.Module,
        skip_ssl: bool = False,
    ) -> None:
        """Initialize ASTRAL quantizer.

        Args:
            tokenizer_name: HuggingFace tokenizer name.
            ssl_model_name: HuggingFace SSL model name.
            ssl_output_layer: Layer index to extract features from.
            encoder: Encoder module to process SSL features.
            quantizer: Quantizer module for discrete representation.
            skip_ssl: Whether to skip loading SSL model (use external).
        """
        super().__init__()
        self.encoder = encoder
        self.quantizer = quantizer
        self.tokenizer_name = tokenizer_name
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        # Load SSL model from Huggingface
        self.ssl_model_name = ssl_model_name
        self.ssl_output_layer = ssl_output_layer
        self.ssl_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(ssl_model_name)

        if skip_ssl:  # in case the same SSL model has been loaded somewhere else
            self.ssl_model = None
        else:
            self.ssl_model = AutoModel.from_pretrained(ssl_model_name).eval()
            self.ssl_model.encoder.layers = self.ssl_model.encoder.layers[:ssl_output_layer]
            self.ssl_model.encoder.layer_norm = torch.nn.Identity()

    def load_separate_checkpoint(self, checkpoint_path: str) -> None:
        """Load checkpoint with separate component weights.

        Args:
            checkpoint_path: Path to checkpoint file.
        """
        params = torch.load(checkpoint_path, map_location="cpu")["net"]
        for key in params.keys():
            for k in list(params[key].keys()):
                if k.startswith("module."):
                    params[key][k[len("module.") :]] = params[key][k]
                    del params[key][k]
        self.encoder.load_state_dict(params["encoder"])
        self.quantizer.load_state_dict(params["vq"])
        if hasattr(self, "decoder") and self.decoder is not None:
            self.decoder.load_state_dict(params["decoder"])
        if hasattr(self, "asr_decoder") and self.asr_decoder is not None:
            self.asr_decoder.load_state_dict(params["predictor"], strict=False)

    def forward(
        self,
        waves_16k: torch.Tensor,
        wave_16k_lens: torch.Tensor,
        ssl_model: Optional[torch.nn.Module] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through ASTRAL quantizer.

        Args:
            waves_16k: Input waveforms at 16kHz [B, T].
            wave_16k_lens: Length of each waveform [B].
            ssl_model: External SSL model to use (if skip_ssl=True).

        Returns:
            Tuple of:
                - Quantized features [B, D, T'].
                - Quantization indices [B, T'].
                - Feature lengths [B].
        """
        ssl_fn = self.ssl_model if self.ssl_model else ssl_model
        assert ssl_fn is not None, (
            "In case in-class SSL model loading is skipped, external ssl_model must be provided"
        )
        waves_16k_input_list = [
            waves_16k[bib, : wave_16k_lens[bib]].cpu().numpy() for bib in range(len(waves_16k))
        ]
        alt_inputs = self.ssl_feature_extractor(
            waves_16k_input_list,
            return_tensors="pt",
            return_attention_mask=True,
            padding=True,
            sampling_rate=16000,
        ).to(waves_16k.device)
        feature_lens = (
            alt_inputs.data["attention_mask"].sum(-1) // 320
        )  # frame rate of hubert is 50 Hz

        outputs = ssl_fn(
            alt_inputs.input_values,
            attention_mask=alt_inputs.attention_mask,
        )
        last_hidden_states = outputs.last_hidden_state
        last_hidden_states = last_hidden_states[:, : feature_lens.max(), :]
        feature_lens = feature_lens.clamp(max=last_hidden_states.size(1))
        last_hidden_states = last_hidden_states.transpose(1, 2)
        x_hidden = self.encoder(last_hidden_states, feature_lens)
        x_hidden = x_hidden.transpose(1, 2)
        x_quantized, indices = self.quantizer(x_hidden)[:2]
        return x_quantized, indices, feature_lens

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
import shutil
import sys
import time
from typing import Any, Dict, List, Optional

import torch
import torch.multiprocessing as mp
import torchaudio
import torchaudio.compliance.kaldi as kaldi
import yaml
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

os.environ["HF_HUB_CACHE"] = "./checkpoints/hf_cache"

from seed_vc.data.ft_dataset import build_ft_dataloader
from seed_vc.hf_utils import load_custom_model_from_hf
from seed_vc.modules.commons import build_model, load_checkpoint, recursive_munch
from seed_vc.optimizers import build_optimizer


class Trainer:
    """Trainer class for Seed-VC model fine-tuning.

    This class handles the training process for the Seed-VC voice conversion model,
    including data loading, model initialization, and training loop management.

    Attributes:
        device (str): Device to run training on.
        log_dir (str): Directory for saving logs and checkpoints.
        max_steps (int): Maximum number of training steps.
        n_epochs (int): Maximum number of training epochs.
        log_interval (int): Interval for logging training progress.
        save_interval (int): Interval for saving checkpoints.
        sr (int): Sample rate for audio processing.
        hop_length (int): Hop length for spectrogram computation.
        win_length (int): Window length for spectrogram computation.
        n_fft (int): FFT size for spectrogram computation.
        f0_condition (bool): Whether to use F0 conditioning.
        train_dataloader (DataLoader): DataLoader for training data.
        campplus_model (nn.Module): Speaker verification model.
        sv_fn (Callable): Speaker verification function.
        rmvpe (Optional[nn.Module]): F0 extraction model.
        f0_fn (Optional[Callable]): F0 extraction function.
        tone_color_converter (nn.Module): Voice conversion model.
        se_db (torch.Tensor): Speaker embedding database.
        bigvgan_model (Optional[nn.Module]): BigVGAN vocoder model.
        hift_gen (Optional[nn.Module]): HiFi-GAN vocoder model.
        vocoder_fn (nn.Module): Vocoder function.
        whisper_model (Optional[nn.Module]): Whisper model for semantic features.
        whisper_feature_extractor (Optional[Any]): Whisper feature extractor.
        wav2vec_model (Optional[nn.Module]): Wav2Vec2 model for semantic features.
        wav2vec_feature_extractor (Optional[Any]): Wav2Vec2 feature extractor.
        semantic_fn (Callable): Semantic feature extraction function.
        model_params (Dict[str, Any]): Model configuration parameters.
        model (Dict[str, nn.Module]): Dictionary of model components.
        optimizer (Any): Optimizer for training.
        epoch (int): Current epoch number.
        iters (int): Current iteration number.
        ema_loss (float): Exponential moving average of loss.
        loss_smoothing_rate (float): Smoothing rate for EMA loss.
    """

    def __init__(
        self,
        # Paths
        log_dir: str,
        data_dir: str,
        run_name: str,
        config_path: str,
        pretrained_ckpt_path: Optional[str],
        pretrained_model: Optional[str],
        # Training parameters
        batch_size: int,
        num_workers: int,
        max_steps: int,
        save_interval: int,
        max_epochs: int,
        log_interval: int,
        device: str,
        # Audio parameters
        sr: int,
        spect_params: Dict[str, Any],
        # Model parameters
        f0_condition: bool,
        speech_tokenizer_config: Dict[str, Any],
        vocoder_config: Dict[str, Any],
        model_params: Dict[str, Any],
        # Learning rate
        base_lr: float,
    ) -> None:
        """Initialize the Trainer with explicit parameters.

        Args:
            log_dir: Base directory for logs.
            data_dir: Directory containing training data.
            run_name: Name for this training run.
            config_path: Path to the configuration file (for copying).
            pretrained_ckpt_path: Path to pretrained checkpoint (optional).
            pretrained_model: Name of pretrained model from HF (optional).
            batch_size: Batch size for training.
            num_workers: Number of workers for data loading.
            max_steps: Maximum number of training steps.
            save_interval: Interval for saving checkpoints.
            max_epochs: Maximum number of training epochs.
            log_interval: Interval for logging training progress.
            device: Device to run training on.
            sr: Sample rate for audio processing.
            spect_params: Spectrogram parameters.
                (n_fft, hop_length, win_length, n_mels, fmin, fmax).
            f0_condition: Whether to use F0 conditioning.
            speech_tokenizer_config: Configuration for speech tokenizer.
            vocoder_config: Configuration for vocoder.
            model_params: Model configuration parameters.
            base_lr: Base learning rate.
        """
        self.device = device
        self.log_dir = os.path.join(log_dir, run_name)
        os.makedirs(self.log_dir, exist_ok=True)
        # copy config file to log dir
        shutil.copyfile(config_path, os.path.join(self.log_dir, os.path.basename(config_path)))

        # Initialize TensorBoard SummaryWriter
        self.writer = SummaryWriter(log_dir=os.path.join(self.log_dir, "tensorboard"))

        self.max_steps = max_steps
        self.n_epochs = max_epochs
        self.log_interval = log_interval
        self.save_interval = save_interval

        self.sr = sr
        self.hop_length = spect_params.get("hop_length", 256)
        self.win_length = spect_params.get("win_length", 1024)
        self.n_fft = spect_params.get("n_fft", 1024)
        self.f0_condition = f0_condition

        self.train_dataloader = build_ft_dataloader(
            data_dir,
            spect_params,
            self.sr,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        # Build config-like dictionaries for model building functions
        config = {
            "model_params": {
                "speech_tokenizer": speech_tokenizer_config,
                "vocoder": vocoder_config,
                "DiT": model_params.get("DiT", {}),
            }
        }

        self.build_sv_model(device, config)
        self.build_semantic_fn(device, config)
        if self.f0_condition:
            self.build_f0_fn(device, config)
        self.build_converter(device, config)
        self.build_vocoder(device, config)

        scheduler_params = {
            "warmup_steps": 0,
            "base_lr": base_lr,
        }

        self.model_params = recursive_munch(model_params)
        self.model = build_model(self.model_params, stage="DiT")

        _ = [self.model[key].to(device) for key in self.model]
        self.model.cfm.estimator.setup_caches(max_batch_size=batch_size, max_seq_length=8192)

        # initialize optimizers after preparing models for compatibility with FSDP
        self.optimizer = build_optimizer(
            {key: self.model[key] for key in self.model},
            lr=float(scheduler_params["base_lr"]),
        )

        if pretrained_ckpt_path is None:
            # find latest checkpoint
            available_checkpoints = glob.glob(os.path.join(self.log_dir, "DiT_epoch_*_step_*.pth"))
            if len(available_checkpoints) > 0:
                latest_checkpoint = max(
                    available_checkpoints,
                    key=lambda x: int(x.split("_")[-1].split(".")[0]),
                )
                earliest_checkpoint = min(
                    available_checkpoints,
                    key=lambda x: int(x.split("_")[-1].split(".")[0]),
                )
                # delete the earliest checkpoint if we have more than 2
                if earliest_checkpoint != latest_checkpoint and len(available_checkpoints) > 2:
                    os.remove(earliest_checkpoint)
                    print(f"Removed {earliest_checkpoint}")
            elif pretrained_model:
                latest_checkpoint = load_custom_model_from_hf(
                    "Plachta/Seed-VC",
                    pretrained_model,
                    None,
                )
            else:
                latest_checkpoint = ""
        else:
            assert os.path.exists(pretrained_ckpt_path), (
                f"Pretrained checkpoint {pretrained_ckpt_path} not found"
            )
            latest_checkpoint = pretrained_ckpt_path

        if os.path.exists(latest_checkpoint):
            self.model, self.optimizer, self.epoch, self.iters = load_checkpoint(
                self.model,
                self.optimizer,
                latest_checkpoint,
                load_only_params=True,
                ignore_modules=[],
                is_distributed=False,
            )
            print(f"Loaded checkpoint from {latest_checkpoint}")
        else:
            self.epoch, self.iters = 0, 0
            print("Failed to load any checkpoint, training from scratch.")

    def build_sv_model(self, device: str, config: Dict[str, Any]) -> None:
        """Build speaker verification model.

        Args:
            device: Device to load the model on.
            config: Configuration dictionary.
        """
        from seed_vc.modules.campplus.DTDNN import CAMPPlus

        self.campplus_model = CAMPPlus(feat_dim=80, embedding_size=192)
        campplus_sd_path = load_custom_model_from_hf(
            "funasr/campplus",
            "campplus_cn_common.bin",
            config_filename=None,
        )
        campplus_sd = torch.load(campplus_sd_path, map_location="cpu")
        self.campplus_model.load_state_dict(campplus_sd)
        self.campplus_model.eval()
        self.campplus_model.to(device)
        self.sv_fn = self.campplus_model

    def build_f0_fn(self, device: str, config: Dict[str, Any]) -> None:
        """Build F0 extraction function.

        Args:
            device: Device to load the model on.
            config: Configuration dictionary.
        """
        from seed_vc.modules.rmvpe import RMVPE

        model_path = load_custom_model_from_hf("lj1995/VoiceConversionWebUI", "rmvpe.pt", None)
        self.rmvpe = RMVPE(model_path, is_half=False, device=device)
        self.f0_fn = self.rmvpe

    def build_converter(self, device: str, config: Dict[str, Any]) -> None:
        """Build voice conversion model.

        Args:
            device: Device to load the model on.
            config: Configuration dictionary.
        """
        from seed_vc.modules.openvoice.api import ToneColorConverter

        ckpt_converter, config_converter = load_custom_model_from_hf(
            "myshell-ai/OpenVoiceV2",
            "converter/checkpoint.pth",
            "converter/config.json",
        )
        self.tone_color_converter = ToneColorConverter(config_converter, device=device)
        self.tone_color_converter.load_ckpt(ckpt_converter)
        self.tone_color_converter.model.eval()
        se_db_path = load_custom_model_from_hf("Plachta/Seed-VC", "se_db.pt", None)
        self.se_db = torch.load(se_db_path, map_location="cpu")

    def build_vocoder(self, device: str, config: Dict[str, Any]) -> None:
        """Build vocoder model.

        Args:
            device: Device to load the model on.
            config: Configuration dictionary.

        Raises:
            ValueError: If vocoder type is not supported.
        """
        vocoder_type = config["model_params"]["vocoder"]["type"]
        vocoder_name = config["model_params"]["vocoder"].get("name", None)
        if vocoder_type == "bigvgan":
            from seed_vc.modules.bigvgan import bigvgan

            self.bigvgan_model = bigvgan.BigVGAN.from_pretrained(
                vocoder_name,
                use_cuda_kernel=False,
            )
            self.bigvgan_model.remove_weight_norm()
            self.bigvgan_model = self.bigvgan_model.eval().to(device)
            vocoder_fn = self.bigvgan_model
        elif vocoder_type == "hifigan":
            from seed_vc.modules.hifigan.f0_predictor import ConvRNNF0Predictor
            from seed_vc.modules.hifigan.generator import HiFTGenerator

            hift_config = yaml.safe_load(open("configs/hifigan.yml", "r"))
            hift_path = load_custom_model_from_hf("FunAudioLLM/CosyVoice-300M", "hift.pt", None)
            self.hift_gen = HiFTGenerator(
                **hift_config["hift"],
                f0_predictor=ConvRNNF0Predictor(**hift_config["f0_predictor"]),
            )
            self.hift_gen.load_state_dict(torch.load(hift_path, map_location="cpu"))
            self.hift_gen.eval()
            self.hift_gen.to(device)
            vocoder_fn = self.hift_gen
        else:
            raise ValueError(f"Unsupported vocoder type: {vocoder_type}")
        self.vocoder_fn = vocoder_fn

    def build_semantic_fn(self, device: str, config: Dict[str, Any]) -> None:
        """Build semantic feature extraction function.

        Args:
            device: Device to load the model on.
            config: Configuration dictionary.

        Raises:
            ValueError: If speech tokenizer type is not supported.
        """
        speech_tokenizer_type = config["model_params"]["speech_tokenizer"].get("type", "cosyvoice")
        if speech_tokenizer_type == "whisper":
            from transformers import AutoFeatureExtractor, WhisperModel

            whisper_model_name = config["model_params"]["speech_tokenizer"]["name"]
            self.whisper_model = WhisperModel.from_pretrained(whisper_model_name).to(device)
            self.whisper_feature_extractor = AutoFeatureExtractor.from_pretrained(
                whisper_model_name,
            )
            # remove decoder to save memory
            del self.whisper_model.decoder

            def semantic_fn(waves_16k: torch.Tensor) -> torch.Tensor:
                """Extract semantic features using Whisper model.

                Args:
                    waves_16k: Input waveforms at 16kHz, shape (B, T).

                Returns:
                    Semantic features, shape (B, T', D).
                """
                ori_inputs = self.whisper_feature_extractor(
                    [w16k.cpu().numpy() for w16k in waves_16k],
                    return_tensors="pt",
                    return_attention_mask=True,
                    sampling_rate=16000,
                )
                ori_input_features = self.whisper_model._mask_input_features(
                    ori_inputs.input_features,
                    attention_mask=ori_inputs.attention_mask,
                ).to(device)
                with torch.no_grad():
                    ori_outputs = self.whisper_model.encoder(
                        ori_input_features.to(self.whisper_model.encoder.dtype),
                        head_mask=None,
                        output_attentions=False,
                        output_hidden_states=False,
                        return_dict=True,
                    )
                S_ori = ori_outputs.last_hidden_state.to(torch.float32)
                S_ori = S_ori[:, : waves_16k.size(-1) // 320 + 1]
                return S_ori

        elif speech_tokenizer_type == "xlsr":
            from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

            model_name = config["model_params"]["speech_tokenizer"]["name"]
            output_layer = config["model_params"]["speech_tokenizer"]["output_layer"]
            self.wav2vec_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
            self.wav2vec_model = Wav2Vec2Model.from_pretrained(model_name)
            self.wav2vec_model.encoder.layers = self.wav2vec_model.encoder.layers[:output_layer]
            self.wav2vec_model = self.wav2vec_model.to(device)
            self.wav2vec_model = self.wav2vec_model.eval()
            self.wav2vec_model = self.wav2vec_model.half()

            def semantic_fn(waves_16k: torch.Tensor) -> torch.Tensor:
                """Extract semantic features using XLSR model.

                Args:
                    waves_16k: Input waveforms at 16kHz, shape (B, T).

                Returns:
                    Semantic features, shape (B, T', D).
                """
                ori_waves_16k_input_list = [
                    waves_16k[bib].cpu().numpy() for bib in range(len(waves_16k))
                ]
                ori_inputs = self.wav2vec_feature_extractor(
                    ori_waves_16k_input_list,
                    return_tensors="pt",
                    return_attention_mask=True,
                    padding=True,
                    sampling_rate=16000,
                ).to(device)
                with torch.no_grad():
                    ori_outputs = self.wav2vec_model(
                        ori_inputs.input_values.half(),
                    )
                S_ori = ori_outputs.last_hidden_state.float()
                return S_ori
        else:
            raise ValueError(f"Unsupported speech tokenizer type: {speech_tokenizer_type}")
        self.semantic_fn = semantic_fn

    def train_one_step(self, batch: List[torch.Tensor]) -> Dict[str, float]:
        """Train one step.

        Args:
            batch: Batch of data containing waves, mels, wave_lengths, mel_input_length.

        Returns:
            Dictionary containing loss values for this step.
        """
        waves, mels, wave_lengths, mel_input_length = batch

        B = waves.size(0)
        target_size = mels.size(2)
        target = mels
        target_lengths = mel_input_length

        # get speaker embedding
        if self.sr != 22050:
            waves_22k = torchaudio.functional.resample(waves, self.sr, 22050)
            wave_lengths_22k = (wave_lengths.float() * 22050 / self.sr).long()
        else:
            waves_22k = waves
            wave_lengths_22k = wave_lengths
        se_batch = self.tone_color_converter.extract_se(waves_22k, wave_lengths_22k)

        ref_se_idx = torch.randint(0, len(self.se_db), (B,))
        ref_se = self.se_db[ref_se_idx].to(self.device)

        # convert
        converted_waves_22k = self.tone_color_converter.convert(
            waves_22k,
            wave_lengths_22k,
            se_batch,
            ref_se,
        ).squeeze(1)

        if self.sr != 22050:
            converted_waves = torchaudio.functional.resample(converted_waves_22k, 22050, self.sr)
        else:
            converted_waves = converted_waves_22k

        waves_16k = torchaudio.functional.resample(waves, self.sr, 16000)
        wave_lengths_16k = (wave_lengths.float() * 16000 / self.sr).long()
        converted_waves_16k = torchaudio.functional.resample(converted_waves, self.sr, 16000)

        # extract S_alt (perturbed speech tokens)
        S_ori = self.semantic_fn(waves_16k)
        S_alt = self.semantic_fn(converted_waves_16k)

        if self.f0_condition:
            F0_ori = self.rmvpe.infer_from_audio_batch(waves_16k)
        else:
            F0_ori = None

        # interpolate speech token to match acoustic feature length
        alt_cond, _, alt_codes, alt_commitment_loss, alt_codebook_loss = (
            self.model.length_regulator(S_alt, ylens=target_lengths, f0=F0_ori)
        )
        ori_cond, _, ori_codes, ori_commitment_loss, ori_codebook_loss = (
            self.model.length_regulator(S_ori, ylens=target_lengths, f0=F0_ori)
        )
        if alt_commitment_loss is None:
            alt_commitment_loss = 0
            alt_codebook_loss = 0
            ori_commitment_loss = 0
            ori_codebook_loss = 0

        # randomly set a length as prompt
        prompt_len_max = target_lengths - 1
        prompt_len = (torch.rand([B], device=alt_cond.device) * prompt_len_max).floor().long()
        prompt_len[torch.rand([B], device=alt_cond.device) < 0.1] = 0

        # for prompt cond token, use ori_cond instead of alt_cond
        cond = alt_cond.clone()
        for bib in range(B):
            cond[bib, : prompt_len[bib]] = ori_cond[bib, : prompt_len[bib]]

        # diffusion target
        common_min_len = min(target_size, cond.size(1))
        target = target[:, :, :common_min_len]
        cond = cond[:, :common_min_len]
        target_lengths = torch.clamp(target_lengths, max=common_min_len)
        x = target

        # style vectors are extracted from the prompt only
        feat_list = []
        for bib in range(B):
            feat = kaldi.fbank(
                waves_16k[bib : bib + 1, : wave_lengths_16k[bib]],
                num_mel_bins=80,
                dither=0,
                sample_frequency=16000,
            )
            feat = feat - feat.mean(dim=0, keepdim=True)
            feat_list.append(feat)
        y_list = []
        with torch.no_grad():
            for feat in feat_list:
                y = self.sv_fn(feat.unsqueeze(0))
                y_list.append(y)
        y = torch.cat(y_list, dim=0)

        loss, _ = self.model.cfm(x, target_lengths, prompt_len, cond, y)

        commitment_loss = (alt_commitment_loss + ori_commitment_loss) * 0.05
        codebook_loss = (ori_codebook_loss + alt_codebook_loss) * 0.15
        loss_total = loss + commitment_loss + codebook_loss

        self.optimizer.zero_grad()
        loss_total.backward()
        torch.nn.utils.clip_grad_norm_(self.model.cfm.parameters(), 10.0)
        torch.nn.utils.clip_grad_norm_(self.model.length_regulator.parameters(), 10.0)
        self.optimizer.step("cfm")
        self.optimizer.step("length_regulator")
        self.optimizer.scheduler(key="cfm")
        self.optimizer.scheduler(key="length_regulator")

        return {
            "cfm_loss": loss.detach().item(),
            "commitment_loss": commitment_loss.detach().item()
            if isinstance(commitment_loss, torch.Tensor)
            else commitment_loss,
            "codebook_loss": codebook_loss.detach().item()
            if isinstance(codebook_loss, torch.Tensor)
            else codebook_loss,
            "total_loss": loss_total.detach().item(),
        }

    def train_one_epoch(self) -> None:
        """Train one epoch.

        Trains the model for one epoch, saving checkpoints at regular intervals.
        """
        _ = [self.model[key].train() for key in self.model]
        for _i, batch in enumerate(tqdm(self.train_dataloader)):
            batch = [b.to(self.device) for b in batch]
            losses = self.train_one_step(batch)

            # Update EMA loss with total loss
            current_loss = losses["total_loss"]
            self.ema_loss = (
                self.ema_loss * self.loss_smoothing_rate
                + current_loss * (1 - self.loss_smoothing_rate)
                if self.iters > 0
                else current_loss
            )

            # Log to TensorBoard
            self.writer.add_scalar("train/total_loss", losses["total_loss"], self.iters)
            self.writer.add_scalar("train/cfm_loss", losses["cfm_loss"], self.iters)
            self.writer.add_scalar("train/commitment_loss", losses["commitment_loss"], self.iters)
            self.writer.add_scalar("train/codebook_loss", losses["codebook_loss"], self.iters)
            self.writer.add_scalar("train/ema_loss", self.ema_loss, self.iters)

            # Log learning rate
            lr_cfm = self.optimizer.optimizers["cfm"].param_groups[0]["lr"]
            lr_lr = self.optimizer.optimizers["length_regulator"].param_groups[0]["lr"]
            self.writer.add_scalar("train/learning_rate_cfm", lr_cfm, self.iters)
            self.writer.add_scalar("train/learning_rate_length_regulator", lr_lr, self.iters)

            if self.iters % self.log_interval == 0:
                print(f"epoch {self.epoch}, step {self.iters}, loss: {self.ema_loss}")
            self.iters += 1

            if self.iters >= self.max_steps:
                break

            if self.iters % self.save_interval == 0:
                print("Saving..")
                state = {
                    "net": {key: self.model[key].state_dict() for key in self.model},
                    "optimizer": self.optimizer.state_dict(),
                    "scheduler": self.optimizer.scheduler_state_dict(),
                    "iters": self.iters,
                    "epoch": self.epoch,
                }
                save_path = os.path.join(
                    self.log_dir,
                    f"DiT_epoch_{self.epoch:05d}_step_{self.iters:05d}.pth",
                )
                torch.save(state, save_path)

                # find all checkpoints and remove old ones
                checkpoints = glob.glob(os.path.join(self.log_dir, "DiT_epoch_*.pth"))
                if len(checkpoints) > 2:
                    checkpoints.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
                    for cp in checkpoints[:-2]:
                        os.remove(cp)

    def train(self) -> None:
        """Main training loop.

        Runs the training process for the specified number of epochs or steps,
        and saves the final model.
        """
        self.ema_loss = 0
        self.loss_smoothing_rate = 0.99
        for epoch in range(self.n_epochs):
            self.epoch = epoch
            self.train_one_epoch()
            if self.iters >= self.max_steps:
                break

        print("Saving final model..")
        state = {
            "net": {key: self.model[key].state_dict() for key in self.model},
        }
        os.makedirs(self.log_dir, exist_ok=True)
        save_path = os.path.join(self.log_dir, "ft_model.pth")
        torch.save(state, save_path)
        print(f"Final model saved at {save_path}")

        # Close TensorBoard writer
        self.writer.close()


def main(args: argparse.Namespace) -> None:
    """Main function to initialize and run training.

    Args:
        args: Command line arguments containing training configuration.
    """
    # Load config file
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Override config values with command line arguments
    # Only override if the argument was explicitly provided
    if args.batch_size is not None and args.batch_size > 0:
        config["batch_size"] = args.batch_size
    if args.log_interval is not None:
        config["log_interval"] = args.log_interval
    if args.save_every is not None:
        config["save_interval"] = args.save_every
    if args.max_epochs is not None:
        config["epochs"] = args.max_epochs

    # Add command line only arguments to config
    config["pretrained_ckpt_path"] = args.pretrained_ckpt
    config["max_steps"] = args.max_steps
    config["num_workers"] = args.num_workers
    config["dataset_dir"] = args.dataset_dir
    config["run_name"] = args.run_name
    config["device"] = args.device
    config["config_path"] = args.config

    # Define the trainer
    trainer = Trainer(
        # Paths
        log_dir=config["log_dir"],
        data_dir=config["dataset_dir"],
        run_name=config["run_name"],
        config_path=config["config_path"],
        pretrained_ckpt_path=config["pretrained_ckpt_path"],
        pretrained_model=config["pretrained_model"],
        # Training parameters
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        max_steps=config["max_steps"],
        max_epochs=config.get("epochs", config.get("max_epochs", 10000)),
        save_interval=config["save_interval"],
        log_interval=config["log_interval"],
        device=config["device"],
        # Audio parameters
        sr=config["preprocess_params"]["sr"],
        spect_params=config["preprocess_params"]["spect_params"],
        # Model parameters
        f0_condition=config["model_params"]["DiT"].get("f0_condition", False),
        speech_tokenizer_config=config["model_params"]["speech_tokenizer"],
        vocoder_config=config["model_params"]["vocoder"],
        model_params=config["model_params"],
        # Learning rate
        base_lr=config["loss_params"]["base_lr"],
    )

    # Start training
    trainer.train()


if __name__ == "__main__":
    if sys.platform == "win32":
        mp.freeze_support()
        mp.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser(
        description="Train Seed-VC voice conversion model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="./configs/presets/config_dit_mel_seed_uvit_xlsr_tiny.yml",
        help="Path to the configuration YAML file",
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        required=True,
        help="Path to the dataset directory containing training data",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=time.strftime("%Y%m%d-%H%M%S") + "_my_run",
        help="Name for this training run (used for organizing logs and checkpoints)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=1000,
        help=(
            "Maximum number of training steps before stopping "
            "(Shorter one of max_steps and max_epochs will be preferred)"
        ),
    )
    parser.add_argument(
        "--pretrained-ckpt",
        type=str,
        default=None,
        help="Path to a pretrained checkpoint to load before training (optional)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size for training (overrides config value if specified)",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=10000,
        help="Maximum number of epochs to train for",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=500,
        help="Save checkpoint every N steps",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of worker processes for data loading (0 for main process only)",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=None,
        help="Interval for logging training progress (overrides config value if specified)",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU device ID to use for training (ignored if MPS is available)",
    )
    args = parser.parse_args()

    if torch.backends.mps.is_available():
        args.device = "mps"
    else:
        args.device = f"cuda:{args.gpu}" if args.gpu else "cuda:0"

    main(args)

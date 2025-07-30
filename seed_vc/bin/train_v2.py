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
import glob
import shutil
import time
from typing import Any, Dict, List, Optional, Tuple

import hydra
import torch
import torchaudio
import yaml
from accelerate import Accelerator, DistributedDataParallelKwargs
from omegaconf import DictConfig
from tqdm import tqdm

from seed_vc.data.ft_dataset import build_ft_dataloader


class Trainer:
    """Training class for voice conversion models using V2 architecture.

    This trainer supports training both CFM (Continuous Flow Matching) and AR
    (Auto-Regressive) models with distributed training through Accelerate.

    Attributes:
        config_path: Path to configuration file.
        mixed_precision: Mixed precision training mode.
        config: Loaded configuration dictionary.
        log_dir: Directory for saving logs and checkpoints.
        accelerator: Accelerate instance for distributed training.
        device: Training device (cuda/cpu).
        train_dataloader: DataLoader for training data.
        spect_params: Spectrogram parameters.
        sr: Sample rate.
        train_cfm: Whether to train CFM model.
        train_ar: Whether to train AR model.
        model: Main model instance.
        optimizer: Optimizer instance.
        scheduler: Learning rate scheduler.
        iters: Current iteration count.
        start_epoch: Starting epoch number.
        log_interval: Interval for logging.
        max_steps: Maximum training steps.
        save_interval: Interval for saving checkpoints.
        max_epochs: Maximum training epochs.
    """

    def __init__(
        self,
        config_path: str,
        pretrained_cfm_ckpt_path: Optional[str],
        pretrained_ar_ckpt_path: Optional[str],
        data_dir: str,
        run_name: str,
        batch_size: int = 0,
        num_workers: int = 0,
        steps: int = 1000,
        save_interval: int = 500,
        max_epochs: int = 1000,
        train_cfm: bool = True,
        train_ar: bool = False,
        mixed_precision: Optional[str] = None,
    ) -> None:
        """Initialize the trainer.

        Args:
            config_path: Path to YAML configuration file.
            pretrained_cfm_ckpt_path: Path to pretrained CFM checkpoint.
            pretrained_ar_ckpt_path: Path to pretrained AR checkpoint.
            data_dir: Directory containing training data.
            run_name: Name for this training run.
            batch_size: Batch size for training.
            num_workers: Number of data loader workers.
            steps: Maximum number of training steps.
            save_interval: Save checkpoint every N steps.
            max_epochs: Maximum number of epochs.
            train_cfm: Whether to train the CFM model.
            train_ar: Whether to train the AR model.
            mixed_precision: Mixed precision mode ('no', 'fp16', 'bf16').
        """
        self.config_path = config_path
        self.mixed_precision = mixed_precision

        # Load configuration
        self.config = yaml.safe_load(open(config_path))

        # Setup logging directory
        self.log_dir = os.path.join("runs", run_name)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir, exist_ok=True)
        shutil.copy(config_path, os.path.join(self.log_dir, os.path.basename(config_path)))

        # Setup accelerator
        ddp_kwargs = DistributedDataParallelKwargs(
            find_unused_parameters=True,
            broadcast_buffers=False,
        )
        self.accelerator = Accelerator(
            project_dir=self.log_dir,
            split_batches=True,
            kwargs_handlers=[ddp_kwargs],
            mixed_precision=mixed_precision,
        )
        self.device = self.accelerator.device

        # Initialize training parameters
        self._init_dataloader(
            data_dir=data_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            spect_params=self.config["mel_fn"],
            sr=self.config["sr"],
        )

        # Initialize models and optimizers
        self._init_models(train_cfm=train_cfm, train_ar=train_ar)

        # Load checkpoint if available
        self._load_checkpoint(pretrained_cfm_ckpt_path, pretrained_ar_ckpt_path)

        # Initialize training parameters
        self.iters = 0
        self.start_epoch = 0
        self.log_interval = 10
        self.max_steps = steps
        self.save_interval = save_interval
        self.max_epochs = max_epochs

    def _init_dataloader(
        self,
        data_dir: str,
        batch_size: int,
        num_workers: int,
        spect_params: Dict[str, Any],
        sr: int,
    ) -> None:
        """Initialize the training dataloader.

        Args:
            data_dir: Directory containing training data.
            batch_size: Batch size for training.
            num_workers: Number of data loader workers.
            spect_params: Spectrogram configuration parameters.
            sr: Sample rate.
        """
        self.spect_params = spect_params
        self.sr = sr
        # Initialize dataloader
        self.train_dataloader = build_ft_dataloader(
            data_dir,
            spect_params,
            self.sr,
            batch_size=batch_size,
            num_workers=num_workers,
        )

    def _init_models(self, train_cfm: bool = True, train_ar: bool = False) -> None:
        """Initialize models and optimizers.

        Args:
            train_cfm: Whether to train CFM model.
            train_ar: Whether to train AR model.

        Raises:
            AssertionError: If neither model is set to train.
        """
        assert train_cfm or train_ar, "At least one model should be trained"
        self.train_cfm = train_cfm
        self.train_ar = train_ar
        # Initialize main model
        self._init_main_model(train_cfm=train_cfm, train_ar=train_ar)

        # Initialize optimizers
        self._init_optimizers()

    def _init_main_model(self, train_cfm: bool = True, train_ar: bool = False) -> None:
        """Initialize the main model.

        Args:
            train_cfm: Whether to train CFM model.
            train_ar: Whether to train AR model.
        """
        with self.accelerator.main_process_first():
            cfg = DictConfig(self.config)
            self.model = hydra.utils.instantiate(cfg).to(self.device)
            for p in self.model.parameters():
                p.requires_grad = False
            if train_cfm:
                for p in self.model.cfm.parameters():
                    p.requires_grad = True
                for p in self.model.cfm_length_regulator.parameters():
                    p.requires_grad = True
            if train_ar:
                for p in self.model.ar.parameters():
                    p.requires_grad = True
                for p in self.model.ar_length_regulator.parameters():
                    p.requires_grad = True

    def _init_optimizers(self) -> None:
        """Initialize optimizers and schedulers."""
        from seed_vc.optimizers import build_single_optimizer

        self.optimizer, self.scheduler = build_single_optimizer(
            self.model,
            lr=2e-5,
        )
        self.optimizer = self.accelerator.prepare(self.optimizer)
        self.scheduler = self.accelerator.prepare(self.scheduler)

    def _find_checkpoint(self, name_pattern: str, max_keep: int = 1) -> Optional[str]:
        """Find checkpoint files in the specified directory.

        Args:
            name_pattern: Glob pattern for checkpoint files.
            max_keep: Maximum number of checkpoints to keep.

        Returns:
            Path to the latest checkpoint if found, None otherwise.
        """
        available_checkpoints = glob.glob(os.path.join(self.log_dir, name_pattern))
        if len(available_checkpoints) > max_keep - 1:
            # find the checkpoint that has the highest step number
            latest_checkpoint = max(
                available_checkpoints,
                key=lambda x: int(x.split("_")[-1].split(".")[0]),
            )
            earliest_checkpoint = min(
                available_checkpoints,
                key=lambda x: int(x.split("_")[-1].split(".")[0]),
            )
            # delete the earliest checkpoint
            if (
                earliest_checkpoint != latest_checkpoint
                and self.accelerator.is_main_process
                and len(available_checkpoints) > max_keep
            ):
                os.remove(earliest_checkpoint)
                print(f"Removed {earliest_checkpoint}")
            return latest_checkpoint
        else:
            return None

    def _load_checkpoint(
        self,
        pretrained_cfm_ckpt_path: Optional[str],
        pretrained_ar_ckpt_path: Optional[str],
    ) -> None:
        """Load checkpoint if available.

        Args:
            pretrained_cfm_ckpt_path: Path to pretrained CFM checkpoint.
            pretrained_ar_ckpt_path: Path to pretrained AR checkpoint.
        """
        cfm_checkpoint_path = pretrained_cfm_ckpt_path or self._find_checkpoint(
            "CFM_epoch_*_step_*.pth",
            max_keep=1,
        )
        ar_checkpoint_path = pretrained_ar_ckpt_path or self._find_checkpoint(
            "AR_epoch_*_step_*.pth",
            max_keep=1,
        )

        with self.accelerator.main_process_first():
            if cfm_checkpoint_path:
                print(f"Loading CFM checkpoint from {cfm_checkpoint_path}")
            if ar_checkpoint_path:
                print(f"Loading AR checkpoint from {ar_checkpoint_path}")
            self.model.load_checkpoints(
                cfm_checkpoint_path=cfm_checkpoint_path,
                ar_checkpoint_path=ar_checkpoint_path,
            )
        self.model = self.accelerator.prepare(self.model)

    def filter_state_dict_shapes(
        self,
        params: Dict[str, torch.Tensor],
        model: torch.nn.Module,
    ) -> Tuple[Dict[str, torch.Tensor], set]:
        """Filter state dict to match model shapes.

        Args:
            params: State dict to filter.
            model: Target model.

        Returns:
            Tuple of (filtered state dict, skipped keys).
        """
        model_state_dict = model.state_dict()
        filtered_state_dict = {
            k: v
            for k, v in params.items()
            if k in model_state_dict and v.shape == model_state_dict[k].shape
        }
        skipped_keys = set(params.keys()) - set(filtered_state_dict.keys())
        if skipped_keys:
            print(f"Warning: Skipped loading some keys due to shape mismatch: {skipped_keys}")
        return filtered_state_dict, skipped_keys

    def train(self) -> None:
        """Main training loop."""
        for epoch in range(self.start_epoch, self.start_epoch + 1000):
            epoch_start_time = time.time()

            try:
                self.train_dataloader.sampler.set_epoch(epoch)
            except AttributeError:
                pass

            self.model.train()

            for i, batch in enumerate(tqdm(self.train_dataloader)):
                # Process batch
                self._process_batch(epoch, i, batch)
                if self.iters >= self.max_steps and self.accelerator.is_main_process:
                    print("Reached max steps, stopping training")
                    self._save_checkpoint(epoch)
                    exit()

            # Log epoch completion
            if self.accelerator.is_main_process:
                print(f"Epoch {epoch} completed in {time.time() - epoch_start_time:.2f} seconds")

            if epoch + 1 >= self.max_epochs and self.accelerator.is_main_process:
                print("Reached max epochs, stopping training")
                self._save_checkpoint(epoch)
                exit()

    def _process_batch(self, epoch: int, i: int, batch: List[torch.Tensor]) -> None:
        """Process a single batch.

        Args:
            epoch: Current epoch number.
            i: Current batch index.
            batch: Batch data containing waves, mels, wave_lens, mel_lens.
        """
        # Move batch to device
        waves, mels, wave_lens, mel_lens = batch
        # Resample to 16kHz for ASR models
        waves_16k = torchaudio.functional.resample(waves, self.sr, 16000)
        wave_lengths_16k = (wave_lens.float() * 16000 / self.sr).long()

        # Forward pass and loss calculation
        with self.accelerator.autocast():
            loss_ar, loss_cfm = self.model(
                waves_16k.to(self.device),
                mels.to(self.device),
                wave_lengths_16k.to(self.device),
                mel_lens.to(self.device),
                forward_ar=self.train_ar,
                forward_cfm=self.train_cfm,
            )

            loss = loss_ar + loss_cfm

            self.accelerator.backward(loss)

            grad_norm_g = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1000.0)
            self.optimizer.step()
            self.scheduler.step(self.iters)
            self.optimizer.zero_grad()

        # Log training progress
        self._log_training_progress(epoch, i, loss, loss_ar, loss_cfm, grad_norm_g)

        # Save checkpoint
        if (
            self.iters != 0
            and self.iters % self.save_interval == 0
            and self.accelerator.is_main_process
        ):
            self._save_checkpoint(epoch)

        # Increment iteration counter
        self.iters += 1

    def _log_training_progress(
        self,
        epoch: int,
        i: int,
        loss: torch.Tensor,
        loss_ar: torch.Tensor,
        loss_cfm: torch.Tensor,
        grad_norm_g: float,
    ) -> None:
        """Log training progress to console.

        Args:
            epoch: Current epoch number.
            i: Current batch index.
            loss: Total loss value.
            loss_ar: AR model loss.
            loss_cfm: CFM model loss.
            grad_norm_g: Gradient norm.
        """
        if self.iters % self.log_interval == 0 and self.accelerator.is_main_process:
            with torch.no_grad():
                cur_lr = self.scheduler.get_last_lr()[0] if i != 0 else 0

                # Log to console
                print(
                    "Epoch %d, Iteration %d, Loss: %.4f, Loss AR: %.4f, "
                    "Loss CFM: %.4f, Grad Norm: %.4f, LR: %.6f"
                    % (epoch, i, loss.item(), loss_ar.item(), loss_cfm.item(), grad_norm_g, cur_lr),
                )

    def _save_checkpoint(self, epoch: int) -> None:
        """Save model checkpoint.

        Args:
            epoch: Current epoch number.
        """
        print("Saving checkpoint...")
        if self.train_ar:
            state = {
                "net": {
                    "ar": self.accelerator.unwrap_model(self.model).ar.state_dict(),
                    "length_regulator": self.accelerator.unwrap_model(
                        self.model,
                    ).ar_length_regulator.state_dict(),
                },
                "iters": self.iters,
                "epoch": epoch,
            }
            save_path = os.path.join(
                self.log_dir,
                "AR_epoch_%05d_step_%05d.pth" % (epoch, self.iters),
            )
            torch.save(state, save_path)
            print(f"Saved AR checkpoint to {save_path}")

            # Find all checkpoints and remove old ones
            self._remove_old_checkpoints("AR_epoch_*_step_*.pth", max_keep=1)
        if self.train_cfm:
            state = {
                "net": {
                    "cfm": self.accelerator.unwrap_model(self.model).cfm.state_dict(),
                    "length_regulator": self.accelerator.unwrap_model(
                        self.model,
                    ).cfm_length_regulator.state_dict(),
                },
                "iters": self.iters,
                "epoch": epoch,
            }
            save_path = os.path.join(
                self.log_dir,
                "CFM_epoch_%05d_step_%05d.pth" % (epoch, self.iters),
            )
            torch.save(state, save_path)
            print(f"Saved CFM checkpoint to {save_path}")

            # Find all checkpoints and remove old ones
            self._remove_old_checkpoints("CFM_epoch_*_step_*.pth", max_keep=1)

    def _remove_old_checkpoints(self, name_pattern: str, max_keep: int = 1) -> None:
        """Remove old checkpoints.

        Args:
            name_pattern: Glob pattern for checkpoint files.
            max_keep: Maximum number of checkpoints to keep.
        """
        checkpoints = glob.glob(os.path.join(self.log_dir, name_pattern))
        if len(checkpoints) > max_keep:
            # Sort by step
            checkpoints.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
            # Remove all except last 1
            for cp in checkpoints[:-max_keep]:
                os.remove(cp)


def main(args: argparse.Namespace) -> None:
    """Main training function.

    Args:
        args: Command line arguments.
    """
    trainer = Trainer(
        config_path=args.config,
        pretrained_cfm_ckpt_path=args.pretrained_cfm_ckpt,
        pretrained_ar_ckpt_path=args.pretrained_ar_ckpt,
        data_dir=args.dataset_dir,
        run_name=args.run_name,
        batch_size=args.batch_size,
        steps=args.max_steps,
        max_epochs=args.max_epochs,
        save_interval=args.save_every,
        num_workers=args.num_workers,
        train_cfm=args.train_cfm,
        train_ar=args.train_ar,
    )
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/v2/vc_wrapper.yaml")
    parser.add_argument("--pretrained-cfm-ckpt", type=str, default=None)
    parser.add_argument("--pretrained-ar-ckpt", type=str, default=None)
    parser.add_argument("--dataset-dir", type=str, default="/path/to/dataset")
    parser.add_argument("--run-name", type=str, default="my_run")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--max-epochs", type=int, default=1000)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--train-cfm", action="store_true", help="Train CFM model")
    parser.add_argument("--train-ar", action="store_true", help="Train AR model")
    args = parser.parse_args()
    main(args)

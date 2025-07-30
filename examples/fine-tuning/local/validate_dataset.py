# Copyright (C) 2025 Human Dataware Lab.
# Created by HDL members
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

"""Dataset validation script for Seed-VC fine-tuning."""

import argparse
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import soundfile as sf
import yaml
from tqdm import tqdm


def validate_audio_file(
    file_path: str,
    target_sr: int,
    min_duration: float = 0.5,
    max_duration: float = 30.0,
) -> Tuple[bool, Dict[str, any], str]:
    """Validate a single audio file using header information only.

    Args:
        file_path: Path to the audio file.
        target_sr: Target sampling rate from config.
        min_duration: Minimum allowed duration in seconds.
        max_duration: Maximum allowed duration in seconds.

    Returns:
        Tuple of (is_valid, stats, error_message).
    """
    stats = {
        "duration": 0.0,
        "sample_rate": 0,
        "num_samples": 0,
        "file_size": 0,
    }

    try:
        # Check file exists
        if not os.path.exists(file_path):
            return False, stats, f"File not found: {file_path}"

        stats["file_size"] = os.path.getsize(file_path)

        # Get audio info from header
        info = sf.info(file_path)
        stats["sample_rate"] = info.samplerate
        stats["num_samples"] = info.frames
        stats["duration"] = info.duration

        # Check sampling rate
        if info.samplerate < target_sr:
            return (
                False,
                stats,
                f"Sampling rate {info.samplerate} Hz is lower than target {target_sr} Hz",
            )

        # Check duration
        if info.duration < min_duration:
            return (
                False,
                stats,
                f"Duration {info.duration:.2f}s is too short (min: {min_duration}s)",
            )

        if info.duration > max_duration:
            return False, stats, f"Duration {info.duration:.2f}s is too long (max: {max_duration}s)"

        return True, stats, ""

    except Exception as e:
        return False, stats, f"Error loading audio: {str(e)}"


def find_audio_files(directory: str) -> List[str]:
    """Find all WAV files in a directory.

    Args:
        directory: Directory to search.

    Returns:
        List of audio file paths.
    """
    audio_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith((".wav", ".wave")):
                audio_files.append(os.path.join(root, file))
    return sorted(audio_files)


def print_statistics(
    valid_files: List[Tuple[str, Dict]],
    invalid_files: List[Tuple[str, Dict, str]],
    target_sr: int,
) -> None:
    """Print dataset statistics."""
    total_files = len(valid_files) + len(invalid_files)

    print("\n" + "=" * 60)
    print("DATASET VALIDATION SUMMARY")
    print("=" * 60)

    print(f"\nTotal files found: {total_files}")
    print(f"Valid files: {len(valid_files)} ({len(valid_files) / total_files * 100:.1f}%)")
    print(f"Invalid files: {len(invalid_files)} ({len(invalid_files) / total_files * 100:.1f}%)")

    if valid_files:
        print("\n" + "-" * 40)
        print("VALID FILES STATISTICS")
        print("-" * 40)

        durations = [stats["duration"] for _, stats in valid_files]
        sample_rates = [stats["sample_rate"] for _, stats in valid_files]
        file_sizes = [stats["file_size"] for _, stats in valid_files]

        print("\nDuration statistics:")
        print(f"  Total duration: {sum(durations):.1f} seconds ({sum(durations) / 60:.1f} minutes)")
        print(f"  Average duration: {np.mean(durations):.2f} seconds")
        print(f"  Min duration: {min(durations):.2f} seconds")
        print(f"  Max duration: {max(durations):.2f} seconds")
        print(f"  Std deviation: {np.std(durations):.2f} seconds")

        print("\nSampling rate distribution:")
        unique_srs = list(set(sample_rates))
        for sr in sorted(unique_srs):
            count = sample_rates.count(sr)
            print(f"  {sr} Hz: {count} files ({count / len(valid_files) * 100:.1f}%)")

        print("\nFile size statistics:")
        print(f"  Total size: {sum(file_sizes) / 1024 / 1024:.1f} MB")
        print(f"  Average size: {np.mean(file_sizes) / 1024 / 1024:.2f} MB")

    if invalid_files:
        print("\n" + "-" * 40)
        print("INVALID FILES")
        print("-" * 40)

        # Group by error type
        error_types = {}
        for file_path, _, error in invalid_files:
            error_type = error.split(":")[0]
            if error_type not in error_types:
                error_types[error_type] = []
            error_types[error_type].append((file_path, error))

        for error_type, files in error_types.items():
            print(f"\n{error_type}: {len(files)} files")
            for file_path, error in files:
                print(f"  - {os.path.basename(file_path)}: {error}")

    print("\n" + "=" * 60)

    if invalid_files:
        print("\n⚠️  WARNING: Dataset contains invalid files!")
        print("Please fix or remove invalid files before training.")
    else:
        print("\n✅ All files passed validation!")

    if valid_files and sum(stats["duration"] for _, stats in valid_files) < 300:  # 5 minutes
        print("\n⚠️  WARNING: Total dataset duration is less than 5 minutes.")
        print("Consider adding more data for better fine-tuning results.")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Validate audio dataset for Seed-VC fine-tuning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "dataset_dir",
        type=str,
        help="Path to the dataset directory containing audio files",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the training configuration file",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=0.5,
        help="Minimum allowed audio duration in seconds",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=30.0,
        help="Maximum allowed audio duration in seconds",
    )

    args = parser.parse_args()

    # Check dataset directory exists
    if not os.path.isdir(args.dataset_dir):
        print(f"Error: Dataset directory '{args.dataset_dir}' does not exist.")
        sys.exit(1)

    # Load config to get target sampling rate
    try:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
        target_sr = config["preprocess_params"]["sr"]
        print(f"Target sampling rate from config: {target_sr} Hz")
    except Exception as e:
        print(f"Error loading config file: {e}")
        sys.exit(1)

    # Find audio files
    print(f"\nSearching for audio files in: {args.dataset_dir}")
    audio_files = find_audio_files(args.dataset_dir)

    if not audio_files:
        print("No WAV files found in the dataset directory.")
        sys.exit(1)

    print(f"Found {len(audio_files)} audio files")

    # Validate each file
    valid_files = []
    invalid_files = []

    print("\nValidating audio files...")
    for file_path in tqdm(audio_files, desc="Validating"):
        is_valid, stats, error = validate_audio_file(
            file_path,
            target_sr,
            args.min_duration,
            args.max_duration,
        )

        if is_valid:
            valid_files.append((file_path, stats))
        else:
            invalid_files.append((file_path, stats, error))

    # Print statistics
    print_statistics(valid_files, invalid_files, target_sr)

    # Exit with non-zero status if invalid files found
    if invalid_files:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()

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

"""Evaluate speaker similarity and audio quality for voice conversion models."""

import argparse
import glob
import json
import os
from typing import Dict, List

import librosa
import numpy as np
import torch
from resemblyzer import VoiceEncoder, preprocess_wav
from tqdm import tqdm

from seed_vc.modules.dnsmos_computor import DNSMOSComputer


def calculate_similarity(
    converted_path: str,
    reference_path: str,
    encoder: VoiceEncoder,
) -> float:
    """Calculate speaker similarity between two audio files.

    Args:
        converted_path: Path to converted audio file.
        reference_path: Path to reference audio file.
        encoder: Pre-trained voice encoder.

    Returns:
        Cosine similarity score between 0 and 1.
    """
    # Preprocess audio files
    converted_wav = preprocess_wav(converted_path)
    reference_wav = preprocess_wav(reference_path)

    # Extract embeddings
    converted_embed = encoder.embed_utterance(converted_wav)
    reference_embed = encoder.embed_utterance(reference_wav)

    # Calculate cosine similarity
    similarity = np.inner(converted_embed, reference_embed)

    return float(similarity)


def calculate_dnsmos(
    audio_path: str,
    mos_computer: DNSMOSComputer,
) -> Dict[str, float]:
    """Calculate DNS MOS scores for audio quality assessment.

    Args:
        audio_path: Path to audio file.
        mos_computer: DNS MOS computer instance.

    Returns:
        Dictionary with SIG, BAK, and OVRL scores.
    """
    # Load audio at 16kHz (required by DNS MOS)
    audio, sr = librosa.load(audio_path, sr=16000)

    # Compute MOS scores
    result = mos_computer.compute(audio, 16000, False)

    return {
        "sig": float(result["SIG"]),
        "bak": float(result["BAK"]),
        "ovr": float(result["OVRL"]),
    }


def find_converted_files(converted_dir: str) -> List[str]:
    """Find all converted audio files in directory.

    Args:
        converted_dir: Directory containing converted audio files.

    Returns:
        List of audio file paths.
    """
    audio_files = []
    for ext in [".wav", ".mp3", ".flac"]:
        pattern = os.path.join(converted_dir, f"*{ext}")
        audio_files.extend(glob.glob(pattern))
        pattern = os.path.join(converted_dir, f"*{ext.upper()}")
        audio_files.extend(glob.glob(pattern))

    return sorted(list(set(audio_files)))


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Evaluate speaker similarity for voice conversion",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--converted-dir",
        type=str,
        required=True,
        help="Directory containing converted audio files",
    )
    parser.add_argument(
        "--reference",
        type=str,
        required=True,
        help="Path to reference audio file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save evaluation results",
    )

    args = parser.parse_args()

    # Check inputs exist
    if not os.path.isdir(args.converted_dir):
        print(f"Error: Converted directory '{args.converted_dir}' does not exist.")
        return

    if not os.path.isfile(args.reference):
        print(f"Error: Reference file '{args.reference}' does not exist.")
        return

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize models
    print("Loading evaluation models...")
    encoder = VoiceEncoder()

    # Determine device
    if torch.cuda.is_available():
        device = "cuda"
        device_id = 0
    else:
        device = "cpu"
        device_id = None

    mos_computer = DNSMOSComputer(
        primary_model_path="assets/models/dnsmos/sig_bak_ovr.onnx",
        p808_model_path="assets/models/dnsmos/model_v8.onnx",
        device=device,
        device_id=device_id,
    )

    # Find converted files
    converted_files = find_converted_files(args.converted_dir)
    if not converted_files:
        print(f"No audio files found in {args.converted_dir}")
        return

    print(f"Found {len(converted_files)} converted files")
    print(f"Reference: {args.reference}")

    # Calculate metrics
    similarities = []
    mos_scores = []
    results = []

    print("\nEvaluating converted audio files...")
    for converted_file in tqdm(converted_files):
        try:
            # Calculate speaker similarity
            similarity = calculate_similarity(converted_file, args.reference, encoder)

            # Calculate DNS MOS
            mos = calculate_dnsmos(converted_file, mos_computer)

            similarities.append(similarity)
            mos_scores.append(mos)
            results.append(
                {
                    "filename": os.path.basename(converted_file),
                    "similarity": similarity,
                    "mos_sig": mos["sig"],
                    "mos_bak": mos["bak"],
                    "mos_ovr": mos["ovr"],
                }
            )
        except Exception as e:
            print(f"Error processing {converted_file}: {e}")
            continue

    if not similarities:
        print("No files evaluated successfully.")
        return

    # Calculate statistics
    mean_similarity = np.mean(similarities)
    std_similarity = np.std(similarities)
    min_similarity = np.min(similarities)
    max_similarity = np.max(similarities)

    # Calculate MOS statistics
    mos_sig_scores = [m["sig"] for m in mos_scores]
    mos_bak_scores = [m["bak"] for m in mos_scores]
    mos_ovr_scores = [m["ovr"] for m in mos_scores]

    mean_mos_sig = np.mean(mos_sig_scores)
    mean_mos_bak = np.mean(mos_bak_scores)
    mean_mos_ovr = np.mean(mos_ovr_scores)

    # Print results
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"\nTotal files evaluated: {len(similarities)}")

    print("\nSPEAKER SIMILARITY:")
    print(f"  Mean: {mean_similarity:.4f}")
    print(f"  Std:  {std_similarity:.4f}")
    print(f"  Min:  {min_similarity:.4f}")
    print(f"  Max:  {max_similarity:.4f}")

    print("\nDNS MOS SCORES:")
    print(f"  Signal (SIG):     {mean_mos_sig:.3f}")
    print(f"  Background (BAK): {mean_mos_bak:.3f}")
    print(f"  Overall (OVRL):   {mean_mos_ovr:.3f}")

    # Save results as JSON
    result_file = os.path.join(args.output_dir, "evaluation_results.json")

    # Sort results by similarity score
    results.sort(key=lambda x: x["similarity"], reverse=True)

    output_data = {
        "summary": {
            "reference": args.reference,
            "converted_dir": args.converted_dir,
            "num_files": len(similarities),
            "speaker_similarity": {
                "mean": float(mean_similarity),
                "std": float(std_similarity),
                "min": float(min_similarity),
                "max": float(max_similarity),
            },
            "dns_mos": {
                "signal_mean": float(mean_mos_sig),
                "background_mean": float(mean_mos_bak),
                "overall_mean": float(mean_mos_ovr),
            },
        },
        "detailed_results": results,
    }

    with open(result_file, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to: {result_file}")


if __name__ == "__main__":
    main()

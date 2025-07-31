# Fine-tuning the Seed-VC Model

This document explains how to fine-tune the Seed-VC model.

## Recipe Structure

The initial state of the recipe has the following structure:

```bash
$ tree -L 2
.
├── configs/      # Directory containing base configuration files for model training
├── checkpoints/  # Cache directory for pretrained models
├── assets/       # Directory containing evaluation model audio files and models
├── local/        # Directory with recipe-specific scripts
├── README.md     # This file
├── run.sh        # Main script
└── utils/        # Shared utility scripts for the recipe
```

When running experiments, you'll generally only need to modify the main script (`run.sh`) and the base config file (`configs/*.yml`).

The main script is structured into the following four stages:

* **Stage 1:** Data validation
* **Stage 2:** Training the voice conversion model
* **Stage 3:** Inference using the voice conversion model
* **Stage 4:** Evaluation of the converted audio

Running the main script will execute all necessary steps for the experiment in sequence.

## Usage

### Preparing the Data

The following data is required for fine-tuning:

* **Training audio data**: Audio files of the speaker (in WAV format)
* **Validation audio data**: Audio files for evaluation (in WAV format)
* **Reference audio**: A single file of the target speaker’s voice (optional)

Place the data in the following directory structure:

```
data/
├── train/          # Training data (WAV files)
│   ├── 001.wav
│   ├── 002.wav
│   └── ...
└── test/           # Validation data (WAV files)
    ├── 001.wav
    ├── 002.wav
    └── ...
```

### Running the Training

```bash
# Move to the recipe directory
$ cd examples/fine-tuning/

# Start training
$ ./run.sh
```

This will run all stages, and training plus evaluation will complete in a few hours.

### Monitoring Training Progress

You can monitor the training process in real time using TensorBoard.

```bash
# Launch TensorBoard
$ tensorboard --logdir ./runs
```

Access `http://localhost:6006` in your browser to view the following metrics:

* **train/total\_loss**: Total loss (sum of all losses)
* **train/cfm\_loss**: CFM (Conditional Flow Matching) loss
* **train/commitment\_loss**: Commitment loss
* **train/codebook\_loss**: Codebook loss
* **train/ema\_loss**: Exponential Moving Average loss
* **train/learning\_rate\_cfm**: Learning rate of the CFM model
* **train/learning\_rate\_length\_regulator**: Learning rate of the Length Regulator

Running the training will generate a directory structure like this:

```
runs/
└── <run_name>/
    ├── *.pth              # Trained model files
    ├── *.yml              # Configuration files used
    ├── tensorboard/       # TensorBoard logs
    └── inferences/        # Inference results
        └── <checkpoint_name>/
            ├── vc_*.wav        # Converted audio files
            └── eval/           # Evaluation results
                ├── converted/  # Evaluations of converted audio
                └── original/   # Evaluations of original audio
```

### Running Specific Stages Only

To run only specific stages, use the `--stage` and `--stop_stage` options:

```bash
# Run only Stage 2 (training)
$ bash run.sh --stage 2 --stop_stage 2

# Run Stages 3-4 (inference and evaluation)
$ bash run.sh --stage 3 --stop_stage 4 \
    --checkpoint runs/my_experiment/best_model.pth \
    --reference_audio_path ./data/reference.wav
```

You can view other options with:

```bash
$ ./run.sh --help
2025-07-23T14:48:12 (run.sh:59:main) ./run.sh --help
Usage: ./run.sh [options]

Options:
    --stage <stage>                Stage to start from (default: "1")
    --stop_stage <stop_stage>      Stage to stop at (default: "4")
    --train_dataset_dir <dir>      Directory containing the training dataset (default: "./data/train")
    --test_dataset_dir <dir>       Directory containing the test dataset (default: "./data/test")
    --train_config <file>          Path to the training configuration file (default: "./configs/presets/config_dit_mel_seed_uvit_xlsr_tiny.yml")
    --run_name <name>              Name for the run (default: "")
    --checkpoint <file>            Path to the checkpoint file for evaluation (default: "")
    --reference_audio_path <path>  Path to the reference wav file for evaluation (default: "")
    --inference_opts <opts>        Additional options for inference (default: "")
```

### Evaluation Metrics

In **Stage 4**, the following metrics are computed for both the original and converted audio:

* **Speaker Similarity**: Measures how close the converted voice is to the reference speaker (closer to 1 is better)
* **DNS MOS**: Mean Opinion Score for audio quality (SIG, BAK, OVRL)

  * **SIG**: Signal clarity, higher (1 to 5) means clearer speech
  * **BAK**: Background noise level, higher means less noise
  * **OVRL**: Overall quality, higher is better

Instead of focusing on absolute values, compare the converted audio’s scores against the original’s for meaningful insights.

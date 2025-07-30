# Seed-VCモデルのファインチューニング

本ドキュメントでは、Seed-VCモデルのファインチューニング方法について説明します。

## レシピの構造

初期状態のレシピは以下のような構造になっています。

```bash
$ tree -L 2
.
├── configs/      # モデル学習のベースとなるコンフィグが入ったディレクトリ
├── checkpoints/  # Pretrainedモデルのキャッシュディレクトリ
├── assets/       # 評価用モデルの音声ファイルやモデルが入ったディレクトリ
├── local/        # レシピ固有のスクリプトが入ったディレクトリ
├── README.md     # 本ファイル
├── run.sh        # メインスクリプト
└── utils/        # レシピ共通のユーティリティスクリプトの入ったディレクトリ
```

実験を回す際は、基本的にメインスクリプト(`run.sh`)とベースとなるコンフィグファイル(`configs/*.yml`)のみを変更します。

メインスクリプトは以下の4つのステージで構成されています。

- Stage 1: データのバリデーション
- Stage 2: 音声変換モデルの学習
- Stage 3: 音声変換モデルを用いた推論
- Stage 4: 変換音声の評価

メインスクリプトを実行することで実験に必要なすべての手順を一通り実行することができます。

## 使い方

### データの準備

ファインチューニングには以下のデータが必要です。

- **学習用音声データ**: 話者の音声ファイル（wav形式）
- **検証用音声データ**: 評価用の音声ファイル（wav形式）
- **リファレンス音声**: 変換先話者の音声ファイル（1ファイル, Optional）

データは以下のようなディレクトリ構造で配置します。

```
data/
├── train/          # 学習用データ（wavファイル）
│   ├── 001.wav
│   ├── 002.wav
│   └── ...
└── test/           # 検証用データ（wavファイル）
    ├── 001.wav
    ├── 002.wav
    └── ...
```


### 実行

```bash
# レシピのディレクトリに移動
$ cd examples/fine-tuning/

# 学習を開始
$ ./run.sh
```

以上ですべてのステージが実行され、数時間で学習及び評価まで完了します。

### 学習の観察

学習中の進捗をリアルタイムで観察するには、TensorBoardを使用します。

```bash
# TensorBoardを起動
$ tensorboard --logdir ./runs
```

ブラウザで http://localhost:6006 にアクセスすることで、以下のメトリクスを確認できます。

- **train/total_loss**: 総損失（すべての損失の合計）
- **train/cfm_loss**: CFM（Conditional Flow Matching）損失
- **train/commitment_loss**: コミットメント損失
- **train/codebook_loss**: コードブック損失
- **train/ema_loss**: 指数移動平均損失
- **train/learning_rate_cfm**: CFMモデルの学習率
- **train/learning_rate_length_regulator**: Length Regulatorの学習率

学習を実行すると、以下のようなディレクトリ構造が生成されます。

```
runs/
└── <run_name>/
    ├── *.pth              # 学習済みモデル
    ├── *.yml              # 利用したコンフィグファイル
    ├── tensorboard/       # TensorBoardログ
    └── inferences/             # 推論結果
        └── <checkpoint_name>/
            ├── vc_*.wav        # 変換された音声
            └── eval/           # 評価結果
                ├── converted/  # 変換音声の評価
                └── original/   # 元音声の評価
```

### ステージごとの実行

特定のステージのみを実行する場合は、`--stage`と`--stop_stage`オプションを使用します。

```bash
# Stage 2（学習）のみを実行
$ bash run.sh --stage 2 --stop_stage 2

# Stage 3-4（推論と評価）を実行
$ bash run.sh --stage 3 --stop_stage 4 \
    --checkpoint runs/my_experiment/best_model.pth \
    --reference_audio_path ./data/reference.wav
```

その他のオプションについては、`run.sh --help`で確認できます。

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


### 評価指標について

Stage 4では以下の評価指標が元音声と変換音声に対して計算されます。

- **Speaker Similarity**: 話者の類似度、1に近いほど参照話者に似ている
- **DNS MOS**: 音声品質スコア（SIG, BAK, OVRL）
    - SIG: 音声の明瞭さ、1から5までのスケールで大きいほど明瞭
    - BAK: 背景ノイズの影響、1から5までのスケールで大きいほどノイズが少ない
    - OVRL: 全体的な音声品質、1から5までのスケールで大きいほど品質が良い

値の絶対値自体よりも、元音声のMOSを基準にして、変換音声のMOSを比較することが重要です。

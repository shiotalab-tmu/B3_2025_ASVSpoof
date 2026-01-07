# ASVSpoof2021 Baseline Inference System

音声Anti-Spoofing（なりすまし検出）の推論システムです。ASVspoof2021の4つのベースラインモデルに対応しています。

## モデル

- **CQCC-GMM**: CQCC特徴量 + Gaussian Mixture Model
- **LFCC-GMM**: LFCC特徴量 + GMM
- **LFCC-LCNN**: LFCC特徴量 + Light Convolutional Neural Network
- **RawNet2**: End-to-end raw waveform CNN

## セットアップ

### 1. 環境構築

```bash
bash setup.sh
```

## 使い方

音声ファイルのリストを作成し，各モデルのプログラムに渡してください．

### 1. ファイルリストの作成

```bash
# file_list.txt を作成（1行に1ファイルパス）
/path/to/audio1.wav
/path/to/audio2.wav
/path/to/audio3.wav
...
```

### 2. LAタスクでの推論

```bash
# RawNet2
uv run python LA/rawnet2.py file_list.txt LA_rawnet2.txt

# LFCC-LCNN
uv run python LA/lfcc_lcnn.py file_list.txt LA_lfcc_lcnn.txt

# LFCC-GMM
uv run python LA/lfcc_gmm.py file_list.txt LA_lfcc_gmm.txt

# CQCC-GMM
uv run python LA/cqcc_gmm.py file_list.txt LA_cqcc_gmm.txt
```

### 3. PAタスクでの推論

```bash
# RawNet2
uv run python PA/rawnet2.py file_list.txt PA_rawnet2.txt

# LFCC-LCNN
uv run python PA/lfcc_lcnn.py file_list.txt PA_lfcc_lcnn.txt

# LFCC-GMM
uv run python PA/lfcc_gmm.py file_list.txt PA_lfcc_gmm.txt

# CQCC-GMM
uv run python PA/cqcc_gmm.py file_list.txt PA_cqcc_gmm.txt
```

### 4. 出力ファイル形式

スコアファイル（例: `PA_rawnet2.txt`）の内容：

```
/path/to/audio1.wav 2.345678
/path/to/audio2.wav -1.234567
/path/to/audio3.wav 0.987654
```

各行は `<ファイルパス> <スコア>` の形式です。

## Pretrained Models

学習済みモデルは `setup.sh` 実行時に自動的にダウンロードされます。

```bash
# LAトラック
LA/pretrained/
├── LFCC_GMM.pkl      # LFCC-GMM
├── CQCC_GMM.pkl      # CQCC-GMM
├── LFCC_LCNN.pt      # LFCC-LCNN
└── rawnet2.pth       # RawNet2

# PAトラック
PA/pretrained/
├── LFCC_GMM.pkl      # LFCC-GMM
├── CQCC_GMM.pkl      # CQCC-GMM
├── LFCC_LCNN.pt      # LFCC-LCNN
└── rawnet2.pth       # RawNet2
```

## ディレクトリ構造

```
B3_ASV/
├── LA/                     # LAトラック用モデル
│   ├── cqcc_gmm.py
│   ├── lfcc_gmm.py
│   ├── lfcc_lcnn.py
│   ├── rawnet2.py
│   └── pretrained/        # Pretrainedモデル
├── PA/                     # PAトラック用モデル
│   └── （LA/と同様）
├── GMM_training/          # GMM学習スクリプト
│   ├── train_cqcc_gmm.py
│   ├── train_lfcc_gmm.py
│   ├── trained_models/   # 学習済みモデル出力先
│   └── README.md
├── common/                # 共通モジュール
│   ├── base_model.py     # 基底クラス
│   ├── audio_utils.py    # 音声読み込み
│   └── feature_extraction/
├── logs/                  # 会話ログ・調査結果
├── setup.sh              # セットアップスクリプト
├── pyproject.toml        # 依存関係
├── PLAN.md               # 実装計画
└── README.md             # このファイル
```



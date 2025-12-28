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

このスクリプトは以下を実行します：
- ASVSpoof2021ベースラインリポジトリをclone
- UVで依存関係をインストール（`uv sync`）
- Pretrained modelを自動ダウンロード（LFCC-LCNN, RawNet2）
- 特徴抽出コードをセットアップ

**注意**: GMM系モデル（CQCC-GMM, LFCC-GMM）の学習済みモデル（.pklファイル）は別途提供されます。

## 使い方

### LA（Logical Access）トラック

```python
# example_la.py
from LA.rawnet2 import RawNet2

# RawNet2で推論
model = RawNet2("LA/pretrained/rawnet2_model.pth", track="LA")
score = model.predict("/path/to/audio.flac")
print(f"Score: {score:.4f}")  # 正ならbonafide、負ならspoof
```

実行：
```bash
uv run python example_la.py
```

### PA（Physical Access）トラック

```python
from PA.rawnet2 import RawNet2

# RawNet2で推論
model = RawNet2("PA/pretrained/rawnet2_model.pth", track="PA")
score = model.predict("/path/to/audio.wav")
print(f"Score: {score:.4f}")
```

### 全モデルの使用例

```python
# LFCC-LCNN
from LA.lfcc_lcnn import LFCC_LCNN
model = LFCC_LCNN("LA/pretrained/trained_network.pt", track="LA")
score = model.predict("/path/to/audio.flac")

# CQCC-GMM
from LA.cqcc_gmm import CQCC_GMM
model = CQCC_GMM("LA/pretrained/gmm_cqcc_la.pkl", track="LA")
score = model.predict("/path/to/audio.flac")

# LFCC-GMM
from LA.lfcc_gmm import LFCC_GMM
model = LFCC_GMM("LA/pretrained/gmm_lfcc_la.pkl", track="LA")
score = model.predict("/path/to/audio.flac")
```

## スコアの解釈

- **正の値**: bonafide（本人音声）の可能性が高い
- **負の値**: spoof（偽音声）の可能性が高い
- スコアの範囲はモデルによって異なります

## GMM Pretrained Model

CQCC-GMMとLFCC-GMMの学習済みモデルは配布されます。

### 学習済みモデルの配置

配布されたGMMモデル（.pklファイル）を以下のディレクトリに配置してください：

```bash
# LAトラック用
LA/pretrained/gmm_cqcc_la.pkl
LA/pretrained/gmm_lfcc_la.pkl

# PAトラック用
PA/pretrained/gmm_cqcc_pa.pkl
PA/pretrained/gmm_lfcc_pa.pkl
```

### GMM学習（開発者向け）

GMM学習を行いたい場合は、[GMM_training/README.md](GMM_training/README.md)を参照してください。

**注意**: 通常のユーザーは学習済みモデルを使用するため、GMM学習は不要です。

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

## Pretrained Model

### 自動ダウンロード（setup.sh実行時）

- LFCC-LCNN (LA): https://www.asvspoof.org/asvspoof2021/pre_trained_LA_LFCC-LCNN.zip
- LFCC-LCNN (PA): https://www.asvspoof.org/asvspoof2021/pre_trained_PA_LFCC-LCNN.zip
- RawNet2 (DF): https://www.asvspoof.org/asvspoof2021/pre_trained_DF_RawNet2.zip
  - **注意**: DFモデルをLA/PAで使い回します（モデル構造は同一）

### 学習が必要

- CQCC-GMM (LA/PA): 公式に公開されていない
- LFCC-GMM (LA/PA): 公式に公開されていない

## 音声フォーマット

- **サンプリングレート**: 16kHz（自動リサンプリング対応）
- **チャンネル**: モノラル（ステレオは自動変換）
- **フォーマット**: .flac, .wav等（soundfile対応形式）

## トラブルシューティング

### Pretrainedモデルが見つからない

```bash
# setup.shを再実行
bash setup.sh
```

### GMM学習でメモリ不足

学習スクリプト内のGMMコンポーネント数を減らしてください。

### CQCC特徴抽出が遅い

CQCC抽出は処理が遅いです（1音声あたり約3秒）。これは正常な動作です。

## 参考資料

- [ASVspoof 2021 公式サイト](https://www.asvspoof.org/index2021.html)
- [ASVspoof 2021 GitHub](https://github.com/asvspoof-challenge/2021)
- [実装計画](PLAN.md)
- [調査レポート](logs/investigation_report.md)

## 引用

このシステムを使用する場合は、ASVspoof2021を引用してください：

```bibtex
@misc{liu2022asvspoof,
  author = {Liu, Xuechen and Wang, Xin and Sahidullah, Md and Patino, Jose and Delgado, Héctor and Kinnunen, Tomi and Todisco, Massimiliano and Yamagishi, Junichi and Evans, Nicholas and Nautsch, Andreas and Lee, Kong Aik},
  title = {{ASVspoof 2021: Towards Spoofed and Deepfake Speech Detection in the Wild}},
  year = {2022},
  url = {https://arxiv.org/abs/2210.02437}
}
```

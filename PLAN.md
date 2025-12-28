# ASVSpoof2021 ベースライン音声ASV推論システム実装計画

## 概要

ASVSpoof2021の4つのベースラインモデル（CQCC-GMM、LFCC-GMM、LFCC-LCNN、RawNet2）を使った音声Anti-Spoofing推論システムを構築します。音声ファイルのパスを指定するとbonafide/spoofの判定スコアを返すシンプルなインターフェースを提供します。

## ディレクトリ構造

```
/home/ymgt/ShiotaLab/B3_ASV/
├── ASVSpoof2021_baseline_system/  # 既存（変更なし）
├── LA/                             # 新規作成
│   ├── cqcc_gmm.py
│   ├── lfcc_gmm.py
│   ├── lfcc_lcnn.py
│   ├── rawnet2.py
│   └── pretrained/
│       ├── gmm_cqcc_la.pkl        # GMM学習後に配置
│       ├── gmm_lfcc_la.pkl        # GMM学習後に配置
│       ├── trained_network.pt      # LFCC-LCNN
│       └── rawnet2_model.pth       # RawNet2（DF用を使い回し）
├── PA/                             # 新規作成
│   ├── cqcc_gmm.py
│   ├── lfcc_gmm.py
│   ├── lfcc_lcnn.py
│   ├── rawnet2.py
│   └── pretrained/
│       ├── gmm_cqcc_pa.pkl        # GMM学習後に配置
│       ├── gmm_lfcc_pa.pkl        # GMM学習後に配置
│       ├── trained_network.pt
│       └── rawnet2_model.pth       # RawNet2（DF用を使い回し）
├── GMM_training/                   # GMM学習用（新規作成）
│   ├── train_cqcc_gmm.py          # CQCC-GMM学習スクリプト
│   ├── train_lfcc_gmm.py          # LFCC-GMM学習スクリプト
│   ├── README.md                  # 学習方法の説明
│   └── trained_models/             # 学習済みモデル出力先
│       ├── gmm_cqcc_la.pkl
│       ├── gmm_cqcc_pa.pkl
│       ├── gmm_lfcc_la.pkl
│       └── gmm_lfcc_pa.pkl
├── common/                         # 共通ユーティリティ
│   ├── __init__.py
│   ├── base_model.py              # 基底クラス
│   ├── audio_utils.py             # 音声読み込み等
│   └── feature_extraction/         # 特徴抽出モジュール
│       ├── __init__.py
│       ├── cqcc.py                # CQCC実装（既存コードから流用）
│       └── lfcc.py                # LFCC実装
├── logs/                           # ログ・調査結果保存
│   ├── conversation_log.json      # 会話ログ（JSON形式）
│   ├── conversation_log.md        # 会話ログ（Markdown形式）
│   ├── investigation_report.md    # 調査レポート
│   └── agent_history/             # エージェント履歴
│       ├── exploration_results.json
│       └── plan_results.json
├── setup.sh                        # セットアップスクリプト
├── pyproject.toml                  # UV用設定
└── README.md                       # 使用方法説明
```

## 実装の詳細

### 1. 共通モジュール（common/）

#### common/base_model.py
全モデルの基底クラスを定義：
```python
from abc import ABC, abstractmethod

class BaseASVModel(ABC):
    def __init__(self, model_path: str, track: str):
        self.model_path = model_path
        self.track = track
        self.load_model()

    @abstractmethod
    def load_model(self):
        """モデルロード処理"""
        pass

    @abstractmethod
    def predict(self, audio_path: str) -> float:
        """
        推論処理
        Returns:
            score: 正の値ならbonafide、負の値ならspoof
        """
        pass
```

#### common/audio_utils.py
音声ファイル読み込みとリサンプリング：
```python
import soundfile as sf
import librosa

def load_audio(audio_path: str, target_sr: int = 16000) -> np.ndarray:
    """音声ファイルを読み込み、16kHzにリサンプリング"""
    audio, sr = sf.read(audio_path)
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    return audio
```

#### common/feature_extraction/
- `cqcc.py`: 既存の`/home/ymgt/ShiotaLab/B3_ASV/ASVSpoof2021_baseline_system/LA/Baseline-CQCC-GMM/python/CQCC/`をコピーまたはインポート
- `lfcc.py`: LFCC特徴抽出（`spafe`ライブラリまたは自前実装）

### 2. 各モデルの実装

#### LA/rawnet2.py, PA/rawnet2.py
**参照ファイル**: `/home/ymgt/ShiotaLab/B3_ASV/ASVSpoof2021_baseline_system/LA/Baseline-RawNet2/model.py`

**実装方針**:
- 既存の`model.py`をインポートパスに追加して利用
- または、モデル定義部分のみコピー
- YAMLファイルの設定を辞書として埋め込む

**インターフェース**:
```python
class RawNet2(BaseASVModel):
    def __init__(self, model_path: str, track: str):
        super().__init__(model_path, track)

    def load_model(self):
        # YAMLから設定を読み込み
        # モデルを初期化
        # state_dictをロード

    def predict(self, audio_path: str) -> float:
        # 音声を読み込み（16kHz, 64600サンプルにパディング）
        # モデルで推論
        # batch_out[:, 1] のスコアを返す
```

#### LA/lfcc_lcnn.py, PA/lfcc_lcnn.py
**参照ファイル**: `/home/ymgt/ShiotaLab/B3_ASV/ASVSpoof2021_baseline_system/LA/Baseline-LFCC-LCNN/project/baseline_LA/model.py`

**実装方針**:
- `sys.path.append()`で既存のベースラインディレクトリをパスに追加
- `project.baseline_LA.model`または`project.baseline_PA.model`をインポート
- または、必要な部分のみコピー

**インターフェース**:
```python
class LFCC_LCNN(BaseASVModel):
    def __init__(self, model_path: str, track: str):
        super().__init__(model_path, track)

    def load_model(self):
        # モデルを初期化
        # checkpointをロード（辞書形式 or state_dict形式の両方に対応）

    def predict(self, audio_path: str) -> float:
        # 音声を読み込み
        # モデルのforward/inferenceメソッドで推論
        # スコアを返す
```

#### LA/cqcc_gmm.py, PA/cqcc_gmm.py
**参照ファイル**: `/home/ymgt/ShiotaLab/B3_ASV/ASVSpoof2021_baseline_system/LA/Baseline-CQCC-GMM/python/gmm.py`

**実装方針**:
- `gmm.py`の`scoring()`関数を単一ファイル処理用にラップ
- CQCC特徴抽出は`common/feature_extraction/cqcc.py`を利用

**インターフェース**:
```python
class CQCC_GMM(BaseASVModel):
    def __init__(self, model_path: str, track: str):
        super().__init__(model_path, track)

    def load_model(self):
        # pickleファイルから2つのGMM（bonafide, spoof）をロード

    def predict(self, audio_path: str) -> float:
        # CQCC特徴抽出
        # bonafide_gmm.score() - spoof_gmm.score()
        # スコアを返す
```

#### LA/lfcc_gmm.py, PA/lfcc_gmm.py
CQCC-GMMと同様だが、LFCC特徴抽出を使用

### 2.5. GMM学習スクリプト（GMM_training/）

#### GMM_training/train_cqcc_gmm.py, train_lfcc_gmm.py

既存のベースラインスクリプトをベースに作成：
- 元ファイル: `/home/ymgt/ShiotaLab/B3_ASV/ASVSpoof2021_baseline_system/LA/Baseline-CQCC-GMM/python/asvspoof2021_baseline.py`
- 元ファイル: `/home/ymgt/ShiotaLab/B3_ASV/ASVSpoof2021_baseline_system/LA/Baseline-LFCC-GMM/python/asvspoof2021_baseline.py`

**実装方針**:
- setup.shで既存スクリプトをコピー
- 学習データパスを設定可能にする
- 学習済みモデルを`GMM_training/trained_models/`に出力
- LA/PA両方の学習に対応

**使用例**:
```bash
cd GMM_training
# CQCC-GMM LA学習
python train_cqcc_gmm.py --track LA --data_path /path/to/ASVspoof2019_LA_train --output trained_models/gmm_cqcc_la.pkl

# LFCC-GMM PA学習
python train_lfcc_gmm.py --track PA --data_path /path/to/ASVspoof2019_PA_train --output trained_models/gmm_lfcc_pa.pkl
```

#### GMM_training/README.md

学習手順の説明：
- ASVspoof2019データセットのダウンロード方法
- 学習の実行方法
- 学習済みモデルの配置方法（`GMM_training/trained_models/` → `LA/pretrained/`, `PA/pretrained/`）
- /home/ayuに既存モデルがある場合の利用方法

### 3. setup.sh

```bash
#!/bin/bash

echo "=== ASVSpoof2021 Baseline Inference System Setup ==="

# UVで環境作成
echo "Creating Python environment with UV..."
uv venv
source .venv/bin/activate
uv pip install -e .

# ディレクトリ作成
echo "Creating directory structure..."
mkdir -p LA/pretrained PA/pretrained logs/agent_history
mkdir -p common/feature_extraction
mkdir -p GMM_training/trained_models

# Pretrainedモデルダウンロード
echo "Downloading pretrained models..."

# LFCC-LCNN LA
echo "  - Downloading LFCC-LCNN LA model..."
wget -q https://www.asvspoof.org/asvspoof2021/pre_trained_LA_LFCC-LCNN.zip -O temp_la_lcnn.zip
unzip -q temp_la_lcnn.zip -d LA/pretrained/
rm temp_la_lcnn.zip

# LFCC-LCNN PA
echo "  - Downloading LFCC-LCNN PA model..."
wget -q https://www.asvspoof.org/asvspoof2021/pre_trained_PA_LFCC-LCNN.zip -O temp_pa_lcnn.zip
unzip -q temp_pa_lcnn.zip -d PA/pretrained/
rm temp_pa_lcnn.zip

# RawNet2 DF（LA/PAで使い回す）
echo "  - Downloading RawNet2 DF model (will be used for LA/PA)..."
wget -q https://www.asvspoof.org/asvspoof2021/pre_trained_DF_RawNet2.zip -O temp_df_rawnet.zip
if [ -f temp_df_rawnet.zip ]; then
    unzip -q temp_df_rawnet.zip
    # LA/PA両方にコピー
    cp pre_trained_DF_model.pth LA/pretrained/rawnet2_model.pth
    cp pre_trained_DF_model.pth PA/pretrained/rawnet2_model.pth
    rm temp_df_rawnet.zip pre_trained_DF_model.pth
    echo "    RawNet2 DF model downloaded and copied to LA/PA directories"
else
    echo "    Error: Failed to download RawNet2 DF model"
fi

# /home/ayu内の既存GMMモデルを検索してコピー
echo "Searching for existing GMM models in /home/ayu..."
find /home/ayu -name "*gmm*asvspoof*.pkl" 2>/dev/null | while read -r pkl_file; do
    echo "  Found: $pkl_file"
    filename=$(basename "$pkl_file")
    cp "$pkl_file" GMM_training/trained_models/ 2>/dev/null && echo "  Copied to GMM_training/trained_models/$filename"
done

# CQCC/LFCCフィーチャー抽出コードをコピー
echo "Copying feature extraction code..."
cp -r ASVSpoof2021_baseline_system/LA/Baseline-CQCC-GMM/python/CQCC common/feature_extraction/ 2>/dev/null || echo "  Note: CQCC code not found"
cp ASVSpoof2021_baseline_system/LA/Baseline-LFCC-GMM/python/LFCC_pipeline.py common/feature_extraction/ 2>/dev/null || echo "  Note: LFCC code not found"

# GMM学習スクリプトをコピー
echo "Setting up GMM training scripts..."
cp ASVSpoof2021_baseline_system/LA/Baseline-CQCC-GMM/python/asvspoof2021_baseline.py GMM_training/train_cqcc_gmm.py 2>/dev/null
cp ASVSpoof2021_baseline_system/LA/Baseline-LFCC-GMM/python/asvspoof2021_baseline.py GMM_training/train_lfcc_gmm.py 2>/dev/null

# 会話ログと調査結果を保存
echo "Saving conversation logs and investigation reports..."
# この処理は実装時に自動的に実行される

echo ""
echo "=== Setup Notes ==="
echo "1. GMM models (CQCC-GMM, LFCC-GMM):"
echo "   - If not found in /home/ayu, you need to train them"
echo "   - Training scripts are in GMM_training/ directory"
echo "   - Requires ASVspoof2019 LA/PA training datasets"
echo "2. RawNet2: Using DF model for both LA and PA"
echo "   - Model structure is identical across tracks"
echo "   - Performance may be suboptimal but functional"
echo "3. See README.md for detailed usage instructions"
echo ""
echo "Setup complete! Run 'source .venv/bin/activate' to activate the environment."
```

### 4. pyproject.toml

```toml
[project]
name = "b3-asv-inference"
version = "0.1.0"
description = "ASVSpoof2021 Baseline Inference System"
requires-python = ">=3.8,<3.11"
dependencies = [
    "torch>=1.6.0,<2.0.0",
    "torchaudio>=0.6.0",
    "numpy>=1.18.0,<1.24.0",
    "scipy>=1.4.0",
    "scikit-learn>=0.24.0",
    "soundfile>=0.10.0",
    "librosa>=0.8.0",
    "pandas>=1.2.0",
    "h5py>=2.10.0",
    "pyyaml>=5.4.0",
    "numba==0.48.0",
]

[build-system]
requires = ["setuptools>=45", "wheel"]
build-backend = "setuptools.build_meta"
```

### 5. ログ・調査結果の保存

#### logs/conversation_log.json
会話履歴をJSON形式で保存：
```json
{
  "session_id": "abundant-fluttering-pie",
  "timestamp": "2025-12-28T...",
  "conversation": [
    {
      "role": "user",
      "content": "...",
      "timestamp": "..."
    },
    {
      "role": "assistant",
      "content": "...",
      "timestamp": "..."
    }
  ]
}
```

#### logs/investigation_report.md
調査結果をMarkdown形式で保存（既にagent a8fa3af, ad95838で取得した内容を整形）

#### logs/agent_history/
各エージェントの出力を個別に保存

**実装方法**:
- 実装開始時に、会話履歴とエージェント出力をこれらのファイルに書き込む
- 各ステップで更新し、他のエージェントや将来のセッションで参照可能にする

### 6. README.md

使用方法、セットアップ手順、各モデルの説明、ディレクトリ構造を記載

## 実装順序

1. **プロジェクト初期化**
   - この計画ファイル（abundant-fluttering-pie.md）を`PLAN.md`としてワークスペース（/home/ymgt/ShiotaLab/B3_ASV/）にコピー
   - 会話ログと調査結果をlogs/に保存

2. **環境構築** (setup.sh, pyproject.toml)

3. **共通モジュール** (common/base_model.py, common/audio_utils.py, common/feature_extraction/)

4. **GMM学習スクリプト準備** (GMM_training/)
   - 既存のベースラインコードを活用
   - README.mdで学習手順を説明
   - /home/ayu内の既存モデルがあればそれを利用
   - /home/audio/内のASVspoof2019データセットを参照

5. **最もシンプルなモデルから実装**:
   - RawNet2 (LA, PA) - DFモデルを使い回し
   - LFCC-LCNN (LA, PA)
   - LFCC-GMM (LA, PA) - GMM学習が完了していれば
   - CQCC-GMM (LA, PA) - GMM学習が完了していれば

6. **ログ・ドキュメント整備** (logs/, README.md)
   - 会話ログと調査結果の保存
   - 使用方法の説明
   - GMMモデルの学習手順

7. **テスト** - /home/audio/内のサンプル音声で各モデルの動作確認

## 重要な参照ファイル

- `/home/ymgt/ShiotaLab/B3_ASV/ASVSpoof2021_baseline_system/LA/Baseline-RawNet2/model.py` - RawNet2モデル定義
- `/home/ymgt/ShiotaLab/B3_ASV/ASVSpoof2021_baseline_system/LA/Baseline-RawNet2/main.py` - 推論処理（特にproduce_evaluation_file関数）
- `/home/ymgt/ShiotaLab/B3_ASV/ASVSpoof2021_baseline_system/LA/Baseline-LFCC-LCNN/project/baseline_LA/model.py` - LFCC-LCNNモデル定義
- `/home/ymgt/ShiotaLab/B3_ASV/ASVSpoof2021_baseline_system/LA/Baseline-CQCC-GMM/python/gmm.py` - GMMの学習とスコアリング
- `/home/ymgt/ShiotaLab/B3_ASV/ASVSpoof2021_baseline_system/LA/Baseline-LFCC-LCNN/project/00_download.sh` - Pretrainedモデルダウンロード方法

## 既存コードの活用

- **RawNet2**: モデル定義をそのままインポート、設定ファイルのみ調整
- **LFCC-LCNN**: `sys.path.append()`で既存のcore_scriptsを参照、またはモデル定義のみコピー
- **GMM系**: `gmm.py`の関数をラップして単一ファイル処理に対応
- **特徴抽出**: CQCC/LFCCのコードを`common/feature_extraction/`にコピーまたはインポート

## Pretrained Modelの入手方法

### 確実にダウンロード可能なモデル

1. **LFCC-LCNN**:
   - LA: `https://www.asvspoof.org/asvspoof2021/pre_trained_LA_LFCC-LCNN.zip`
   - PA: `https://www.asvspoof.org/asvspoof2021/pre_trained_PA_LFCC-LCNN.zip`

2. **RawNet2**:
   - DF: `https://www.asvspoof.org/asvspoof2021/pre_trained_DF_RawNet2.zip`
   - **LA/PA**: 公式には公開されていないが、**DFモデルを使い回す**
     - モデル構造はLA/PA/DF間で完全に同一
     - `--track`引数はデータパスの切り替えのみに使用
     - README.mdに「DFモデルで代用」と注記

### GMMモデル（学習が必要）

3. **CQCC-GMM & LFCC-GMM**:
   - Pretrained weightsは公式に公開されていない
   - 対策：`GMM_training/`ディレクトリを作成し、学習スクリプトを配置
   - `/home/ayu`内に既存の重みがあれば利用

## データセットの場所

- **音声データ**: `/home/audio/`内に配置
- **ASVspoof2019データセット**: GMM学習用
  - `/home/audio/ASVspoof2019_LA_train/`
  - `/home/audio/ASVspoof2019_PA_train/`
  - 等（実際のパスはsetup時に確認）

## 注意事項

1. **GMMモデルのpretrained weights**:
   - 公式に公開されていない
   - `GMM_training/`で学習スクリプトを提供
   - 学習にはASVspoof2019 LA/PAトレーニングデータが必要（`/home/audio/`内）
   - `/home/ayu`内に既存の重みがあればそれを利用

2. **依存関係のバージョン**:
   - PyTorch 1.6-1.9推奨（2.0以降は互換性問題の可能性）
   - numba==0.48.0固定（librosa 0.8.0との互換性）

3. **音声フォーマット**:
   - 入力は16kHz, monoを前提
   - `audio_utils.py`で自動リサンプリング・変換

4. **会話ログの同期**:
   - 実装時に`~/.claude/`内のログを`logs/`にも保存
   - JSON + Markdown両形式で保存し、可読性と機械可読性を両立

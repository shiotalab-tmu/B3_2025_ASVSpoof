# ASVSpoof2021ベースラインシステム調査レポート

**調査日時**: 2025-12-28
**調査エージェントID**: a8fa3af, ad95838

## プロジェクト概要

ASVSpoof2021チャレンジのベースラインシステムを使った音声Anti-Spoofing推論システムの構築

## ディレクトリ構造調査結果

### 既存のベースラインシステム

```
/home/ymgt/ShiotaLab/B3_ASV/ASVSpoof2021_baseline_system/
├── LA/                          # Logical Access トラック
│   ├── Baseline-CQCC-GMM
│   ├── Baseline-LFCC-GMM
│   ├── Baseline-LFCC-LCNN
│   └── Baseline-RawNet2
├── PA/                          # Physical Access トラック
│   ├── Baseline-CQCC-GMM
│   ├── Baseline-LFCC-GMM
│   ├── Baseline-LFCC-LCNN
│   └── Baseline-RawNet2
└── DF/                          # Deepfake トラック
    └── （同様のベースライン）
```

## ベースラインモデル詳細

### 1. RawNet2 (End-to-End DNN)

**特徴**:
- Raw waveformを直接入力
- Sinc Convolution + Residual Blocks + GRU
- モデル構造はLA/PA/DF間で完全に同一

**モデル設定** (`model_config_RawNet.yaml`):
- サンプリングレート: 16kHz
- 入力長: 64,600サンプル（約4秒）
- フィルタチャネル数: [20, [20, 20], [20, 128], [128, 128]]
- 出力クラス: 2 (bonafide/spoof)

**推論処理**:
```python
# produce_evaluation_file()関数（main.py:36-56）
batch_out = model(batch_x)
batch_score = batch_out[:, 1].data.cpu().numpy().ravel()  # bonafideクラスのスコア
```

**Pretrained Model**:
- DF: `https://www.asvspoof.org/asvspoof2021/pre_trained_DF_RawNet2.zip`
- LA/PA: 公式には公開されていない→**DFモデルを使い回す**

### 2. LFCC-LCNN (Feature-based CNN)

**特徴**:
- LFCC特徴抽出 + CNN + LSTM
- PyTorchベース
- 入力: 音声ファイルパス

**モデル設定**:
- LFCC次元: 20次元（delta/delta-deltaを含めて60次元）
- フロントエンド: LFCC抽出（`util_frontend.py`）
- バックエンド: CNN → LSTM → FC

**チェックポイントロード** (2種類対応):
```python
# state_dictが辞書内にある場合
pt_model.load_state_dict(checkpoint['state_dict'])
# state_dictが直接の場合
pt_model.load_state_dict(checkpoint)
```

**Pretrained Model**:
- LA: `https://www.asvspoof.org/asvspoof2021/pre_trained_LA_LFCC-LCNN.zip`
- PA: `https://www.asvspoof.org/asvspoof2021/pre_trained_PA_LFCC-LCNN.zip`

### 3. CQCC-GMM

**特徴**:
- CQCC (Constant Q Cepstral Coefficients) 特徴抽出
- GMM (Gaussian Mixture Model) 分類器
- Matlab & Python実装

**実装** (`gmm.py`):
```python
def scoring(scores_file, dict_file, features, eval_ndx, eval_folder, audio_ext):
    gmm_bona = GaussianMixture(covariance_type='diag')
    gmm_spoof = GaussianMixture(covariance_type='diag')
    # pickleファイルからGMMパラメータをロード
    with open(dict_file, "rb") as tf:
        gmm_dict = pickle.load(tf)
        gmm_bona._set_parameters(gmm_dict['bona'])
        gmm_spoof._set_parameters(gmm_dict['spoof'])

    # スコアリング
    scr[i] = gmm_bona.score(Tx.T) - gmm_spoof.score(Tx.T)
```

**Pretrained Model**:
- **公式には公開されていない**
- 学習スクリプト: `asvspoof2021_baseline.py`
- 出力: `gmm_cqcc_asvspoof21_{la|pa}.pkl`

### 4. LFCC-GMM

CQCC-GMMと同様だが、LFCC特徴抽出を使用

**Pretrained Model**:
- **公式には公開されていない**
- 学習スクリプト: `asvspoof2021_baseline.py`
- 出力: `gmm_lfcc_asvspoof21_{la|pa}.pkl`

## LA vs PA の違い

### RawNet2
- **実装**: ほぼ同一（シンボリックリンク）
- **差分**: `--track`パラメータのデフォルト値のみ
- **データパス**: `ASVspoof_{track}` で自動生成

### LFCC-LCNN
- **実装**: LA/PA個別に実装（config.pyが異なる）
- **学習データ**: LA用はASVspoof2019 LA、PA用はASVspoof2019 PA

### GMM系
- **実装**: LA/PAで独立
- **学習データ**: 各トラックごとに学習が必要

## スコア形式

全モデル共通:
- **フォーマット**: `filename score`（各行）
- **意味**: 高いほどbonafide（本人音声）の可能性が高い
- **範囲**: モデルに依存
  - RawNet2: logit値（-∞ ～ +∞）
  - LFCC-LCNN: 生スコア（0 ～ 数十）
  - GMM: log likelihood差分

## 依存関係

### RawNet2
```
pytorch==1.4.0
numba==0.48
numpy==1.17.0
librosa==0.7.2
pyyaml==5.3.1
```

### LFCC-LCNN
```
python=3.8
pytorch=1.6
scipy=1.4.1
numpy=1.18.1
librosa=0.8.0
soundfile
numba=0.48.0
```

### GMM系
```
spafe
librosa
pandas
matplotlib
samplerate
h5py
scikit-learn
```

## データセットの場所

- `/home/audio/`: 音声データの配置先
- `/home/audio/ASVspoof2019_LA_train/`: LA学習データ
- `/home/audio/ASVspoof2019_PA_train/`: PA学習データ

## Webソース

- [ASVspoof 2021 GitHub Repository](https://github.com/asvspoof-challenge/2021)
- [MattyB95/pre_trained_DF_RawNet2 on Hugging Face](https://huggingface.co/MattyB95/pre_trained_DF_RawNet2)

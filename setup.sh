#!/bin/bash

echo "=== ASVSpoof2021 Baseline Inference System Setup ==="

# ASVSpoof2021ベースラインシステムをclone
if [ ! -d "ASVSpoof2021_baseline_system" ]; then
    echo "Cloning ASVspoof2021 baseline repository..."
    git clone https://github.com/asvspoof-challenge/2021.git ASVSpoof2021_baseline_system
    echo "Repository cloned successfully."
else
    echo "ASVSpoof2021_baseline_system already exists. Skipping clone."
fi

# UVで依存関係をインストール
echo "Installing dependencies with UV..."
uv sync

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

# CQCC/LFCC特徴量抽出コードをコピー
echo "Copying feature extraction code..."
cp -r ASVSpoof2021_baseline_system/LA/Baseline-CQCC-GMM/python/CQCC common/feature_extraction/ 2>/dev/null || echo "  Note: CQCC code not found"
cp ASVSpoof2021_baseline_system/LA/Baseline-LFCC-GMM/python/LFCC_pipeline.py common/feature_extraction/ 2>/dev/null || echo "  Note: LFCC code not found"
cp ASVSpoof2021_baseline_system/LA/Baseline-CQCC-GMM/python/gmm.py common/ 2>/dev/null || echo "  Note: gmm.py not found"
cp ASVSpoof2021_baseline_system/LA/Baseline-LFCC-GMM/python/gmm.py common/ 2>/dev/null || echo "  Note: gmm.py not found (LFCC)"

# GMM学習スクリプトをコピー
echo "Setting up GMM training scripts..."
cp ASVSpoof2021_baseline_system/LA/Baseline-CQCC-GMM/python/asvspoof2021_baseline.py GMM_training/train_cqcc_gmm.py 2>/dev/null
cp ASVSpoof2021_baseline_system/LA/Baseline-LFCC-GMM/python/asvspoof2021_baseline.py GMM_training/train_lfcc_gmm.py 2>/dev/null

echo ""
echo "=== Setup Notes ==="
echo "1. GMM models (CQCC-GMM, LFCC-GMM):"
echo "   - Pretrained GMM models (.pkl files) should be provided separately"
echo "   - Place them in the appropriate pretrained directories (LA/pretrained/, PA/pretrained/)"
echo "2. RawNet2: Using DF model for both LA and PA"
echo "   - Model structure is identical across tracks"
echo "   - Performance may be suboptimal but functional"
echo "3. See README.md for detailed usage instructions"
echo ""
echo "Setup complete! Use 'uv run python your_script.py' to run scripts."

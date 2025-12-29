#!/bin/bash
set -e  # エラーが発生したら即座に終了

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
mkdir -p common/feature_extraction

# CQCC/LFCC特徴量抽出コードをコピー
echo "Copying feature extraction code..."
cp -r ASVSpoof2021_baseline_system/LA/Baseline-CQCC-GMM/python/CQCC common/feature_extraction/ 2>/dev/null || echo "  Warning: CQCC code not found"
cp ASVSpoof2021_baseline_system/LA/Baseline-LFCC-GMM/python/LFCC_pipeline.py common/feature_extraction/ 2>/dev/null || echo "  Warning: LFCC code not found"
cp ASVSpoof2021_baseline_system/LA/Baseline-CQCC-GMM/python/gmm.py common/ 2>/dev/null || echo "  Warning: gmm.py not found"
cp ASVSpoof2021_baseline_system/LA/Baseline-LFCC-GMM/python/gmm.py common/ 2>/dev/null || echo "  Warning: gmm.py not found (LFCC)"

echo "Setup complete! Follow README.md to run scripts."

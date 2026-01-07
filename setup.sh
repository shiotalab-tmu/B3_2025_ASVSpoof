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

# ================================================================================
# 事前学習モデルのダウンロード関数
# ================================================================================
download_pretrained_models() {
    echo ""
    echo "=== Downloading Pretrained Models ==="

    SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
    TMP_DIR="${SCRIPT_DIR}/.tmp_download"

    # URLs (全8モデル: PA/LA × CQCC-GMM/LFCC-GMM/LFCC-LCNN/RawNet2)
    declare -A URLS=(
        # PA
        ["PA/CQCC-GMM"]="http://www.asvspoof.org/asvspoof2021/pre_trained_PA_CQCC-GMM.zip"
        ["PA/LFCC-GMM"]="http://www.asvspoof.org/asvspoof2021/pre_trained_PA_LFCC-GMM.zip"
        ["PA/LFCC-LCNN"]="https://www.asvspoof.org/asvspoof2021/pre_trained_PA_LFCC-LCNN.zip"
        ["PA/RawNet2"]="https://www.asvspoof.org/asvspoof2021/pre_trained_PA_RawNet2.zip"
        # LA
        ["LA/CQCC-GMM"]="http://www.asvspoof.org/asvspoof2021/pre_trained_LA_CQCC-GMM.zip"
        ["LA/LFCC-GMM"]="http://www.asvspoof.org/asvspoof2021/pre_trained_LA_LFCC-GMM.zip"
        ["LA/LFCC-LCNN"]="https://www.asvspoof.org/asvspoof2021/pre_trained_LA_LFCC-LCNN.zip"
        ["LA/RawNet2"]="https://www.asvspoof.org/asvspoof2021/pre_trained_LA_RawNet2.zip"
    )

    # 出力ファイル名のマッピング
    declare -A OUTPUT_NAMES=(
        ["CQCC-GMM"]="CQCC_GMM.pkl"
        ["LFCC-GMM"]="LFCC_GMM.pkl"
        ["LFCC-LCNN"]="LFCC_LCNN.pt"
        ["RawNet2"]="rawnet2.pth"
    )

    mkdir -p "$TMP_DIR"

    for key in "${!URLS[@]}"; do
        url="${URLS[$key]}"
        track=$(dirname "$key")   # PA or LA
        model=$(basename "$key")  # CQCC-GMM, LFCC-GMM, LFCC-LCNN, RawNet2

        # 出力先: ./PA/pretrained/ or ./LA/pretrained/
        output_dir="${SCRIPT_DIR}/${track}/pretrained"
        mkdir -p "$output_dir"

        zip_name=$(basename "$url")
        zip_path="${TMP_DIR}/${zip_name}"

        echo ""
        echo "[${track}/${model}]"

        # Download
        echo "  Downloading: $url"
        curl -sL -o "$zip_path" "$url"

        # Extract to temp
        echo "  Extracting..."
        unzip -o -q -d "$TMP_DIR" "$zip_path"

        # Move and rename files
        if [[ "$model" == *"GMM"* ]]; then
            # GMM: .mat -> .pkl (convert_mat_to_pkl.py を使用)
            mat_file=$(find "$TMP_DIR" -name "*.mat" -newer "$zip_path" 2>/dev/null | head -1)
            if [ -z "$mat_file" ]; then
                mat_file=$(find "$TMP_DIR" -name "*.mat" | head -1)
            fi
            if [ -f "$mat_file" ]; then
                pkl_path="${output_dir}/${OUTPUT_NAMES[$model]}"
                echo "  Converting to: ${pkl_path}"
                uv run python "${SCRIPT_DIR}/convert_mat_to_pkl.py" "$mat_file" "$pkl_path"
                rm -f "$mat_file"
            fi
        elif [[ "$model" == "LFCC-LCNN" ]]; then
            # LFCC-LCNN: .pt -> LFCC_LCNN.pt
            pt_file=$(find "$TMP_DIR" -name "*.pt" | head -1)
            if [ -f "$pt_file" ]; then
                dest="${output_dir}/${OUTPUT_NAMES[$model]}"
                echo "  Moving to: ${dest}"
                mv "$pt_file" "$dest"
            fi
        elif [[ "$model" == "RawNet2" ]]; then
            # RawNet2: .pth -> rawnet2.pth
            pth_file=$(find "$TMP_DIR" -name "*.pth" | head -1)
            if [ -f "$pth_file" ]; then
                dest="${output_dir}/${OUTPUT_NAMES[$model]}"
                echo "  Moving to: ${dest}"
                mv "$pth_file" "$dest"
            fi
        fi

        # Cleanup zip
        rm -f "$zip_path"
    done

    # Cleanup temp dir
    rm -rf "$TMP_DIR"

    echo ""
    echo "=== Pretrained Models Download Complete ==="
    echo ""
    echo "Models saved to:"
    ls -la "${SCRIPT_DIR}/PA/pretrained/" 2>/dev/null || echo "  PA/pretrained/ not found"
    ls -la "${SCRIPT_DIR}/LA/pretrained/" 2>/dev/null || echo "  LA/pretrained/ not found"
}

# 事前学習モデルをダウンロード
download_pretrained_models

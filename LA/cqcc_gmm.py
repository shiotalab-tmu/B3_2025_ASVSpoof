"""CQCC-GMM model for LA track"""

import sys
from pathlib import Path
from typing import Union
import numpy as np
import pickle
from sklearn.mixture import GaussianMixture

# 既存のCQCC-GMMベースラインをインポートパスに追加
baseline_path = Path(__file__).parent.parent / "ASVSpoof2021_baseline_system" / "LA" / "Baseline-CQCC-GMM" / "python"
sys.path.insert(0, str(baseline_path))

from gmm import extract_cqcc  # noqa: E402
from common.base_model import BaseASVModel  # noqa: E402


class CQCC_GMM(BaseASVModel):
    """CQCC-GMM ASV model"""

    def __init__(self, model_path: Union[str, Path], track: str = "LA"):
        """
        Args:
            model_path: pretrained GMM model path (.pkl)
            track: 'LA' or 'PA'
        """
        self.gmm_bona = None
        self.gmm_spoof = None

        super().__init__(model_path, track)

    def load_model(self):
        """モデルをロード"""
        # pickleからGMM辞書をロード
        with open(str(self.model_path), "rb") as f:
            gmm_dict = pickle.load(f)

        # bonafideとspoofのGMMを初期化
        self.gmm_bona = GaussianMixture(covariance_type='diag')
        self.gmm_spoof = GaussianMixture(covariance_type='diag')

        # パラメータをセット
        self.gmm_bona._set_parameters(gmm_dict['bona'])
        self.gmm_spoof._set_parameters(gmm_dict['spoof'])

        print(f"CQCC-GMM model loaded for {self.track} track")
        print(f"  Bonafide GMM components: {self.gmm_bona.n_components}")
        print(f"  Spoof GMM components: {self.gmm_spoof.n_components}")

    def predict(self, audio_path: Union[str, Path]) -> float:
        """
        音声ファイルから推論

        Args:
            audio_path: 音声ファイルパス

        Returns:
            score: スコア（正の値ならbonafide、負の値ならspoof）
        """
        import soundfile as sf

        # 音声を読み込み
        sig, fs = sf.read(str(audio_path))

        # CQCC特徴抽出
        # extract_cqcc(sig, fs, fmin, fmax, B, cf, d)
        # デフォルト: fmin=96, fmax=fs/2, B=12, cf=19, d=16
        features = extract_cqcc(
            sig=sig,
            fs=fs,
            fmin=96,
            fmax=fs/2,
            B=12,
            cf=19,
            d=16
        )

        # スコア計算: log P(X|bonafide) - log P(X|spoof)
        score_bona = self.gmm_bona.score(features)
        score_spoof = self.gmm_spoof.score(features)
        score = score_bona - score_spoof

        return float(score)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: uv run python LA/cqcc_gmm.py <file_list.txt> <output_scores.txt>")
        print("  file_list.txt: Text file containing audio file paths (one per line)")
        print("  output_scores.txt: Output file for scores")
        sys.exit(1)

    file_list_path = sys.argv[1]
    output_path = sys.argv[2]
    model_path = "LA/pretrained/CQCC_GMM.pkl"

    try:
        # Load model
        print(f"Loading CQCC-GMM model from {model_path}...")
        model = CQCC_GMM(model_path, track="LA")

        # Read file list
        with open(file_list_path, 'r') as f:
            audio_files = [line.strip() for line in f if line.strip()]

        print(f"Processing {len(audio_files)} files...")

        # Process each file and write scores
        with open(output_path, 'w') as out_f:
            for i, audio_path in enumerate(audio_files, 1):
                try:
                    score = model.predict(audio_path)
                    out_f.write(f"{audio_path} {score:.6f}\n")
                    print(f"[{i}/{len(audio_files)}] {audio_path}: {score:.6f}")
                except Exception as e:
                    print(f"[{i}/{len(audio_files)}] Error processing {audio_path}: {e}", file=sys.stderr)

        print(f"\nScores written to {output_path}")

    except FileNotFoundError as e:
        print(f"Error: File not found - {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

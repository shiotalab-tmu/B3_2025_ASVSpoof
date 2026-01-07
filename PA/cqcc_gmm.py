"""CQCC-GMM model for PA track"""

import sys
from pathlib import Path
from typing import Union
import numpy as np
import pickle
from sklearn.mixture import GaussianMixture

# 既存のCQCC-GMMベースラインをインポートパスに追加
baseline_path = Path(__file__).parent.parent / "ASVSpoof2021_baseline_system" / "PA" / "Baseline-CQCC-GMM" / "python"
sys.path.insert(0, str(baseline_path))

from gmm import extract_cqcc  # noqa: E402
from common.base_model import BaseASVModel  # noqa: E402


def cqcc_deltas(x, hlen=3):
    from numpy import tile, concatenate, arange
    from scipy.signal import lfilter
    win = list(range(hlen, -hlen - 1, -1))
    norm = 2 * sum([i ** 2 for i in range(1, hlen + 1)])
    xx_1 = tile(x[:, 0], (1, hlen)).reshape(hlen, -1).T
    xx_2 = tile(x[:, -1], (1, hlen)).reshape(hlen, -1).T
    xx = concatenate([xx_1, x, xx_2], axis=-1)
    D = lfilter(win, 1, xx) / norm
    return D[:, hlen*2:]


class CQCC_GMM(BaseASVModel):
    """CQCC-GMM ASV model"""

    def __init__(self, model_path: Union[str, Path], track: str = "PA"):
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

        # n_componentsを手動で設定（_set_parametersでは更新されない）
        self.gmm_bona.n_components = len(self.gmm_bona.weights_)
        self.gmm_spoof.n_components = len(self.gmm_spoof.weights_)

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
        # MATLABに合わせたパラメータ: fmin=62.50, fmax=8000
        features = extract_cqcc(
            sig=sig,
            fs=fs,
            fmin=62.50,
            fmax=8000,
            B=12,
            cf=19,
            d=16
        )

        # スコア計算: log P(X|bonafide) - log P(X|spoof)
        score_bona = self.gmm_bona.score(features.T)
        score_spoof = self.gmm_spoof.score(features.T)
        score = score_bona - score_spoof

        return float(score)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: uv run python PA/cqcc_gmm.py <file_list.txt> <output_scores.txt>")
        print("  file_list.txt: Text file containing audio file paths (one per line)")
        print("  output_scores.txt: Output file for scores")
        sys.exit(1)

    file_list_path = sys.argv[1]
    output_path = sys.argv[2]
    model_path = "PA/pretrained/CQCC_GMM.pkl"

    try:
        # Load model
        print(f"Loading CQCC-GMM model from {model_path}...")
        model = CQCC_GMM(model_path, track="PA")

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

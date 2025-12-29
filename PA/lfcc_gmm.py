"""LFCC-GMM model for PA track"""

import sys
from pathlib import Path
from typing import Union
import numpy as np
import pickle
from sklearn.mixture import GaussianMixture
import soundfile as sf

# 既存のLFCC-GMMベースラインをインポートパスに追加
baseline_path = Path(__file__).parent.parent / "ASVSpoof2021_baseline_system" / "PA" / "Baseline-LFCC-GMM" / "python"
sys.path.insert(0, str(baseline_path))

from LFCC_pipeline import lfcc  # noqa: E402
from common.base_model import BaseASVModel  # noqa: E402


def extract_lfcc_with_deltas(file_path: Union[str, Path], num_ceps: int = 20, order_deltas: int = 2,
                              low_freq: int = 0, high_freq: int = 4000) -> np.ndarray:
    """
    LFCC特徴量をデルタ係数付きで抽出

    Args:
        file_path: 音声ファイルパス
        num_ceps: ケプストラム次数
        order_deltas: デルタ次数（0, 1, 2）
        low_freq: 低域周波数
        high_freq: 高域周波数

    Returns:
        features: LFCC特徴量 (n_features, n_frames)
    """
    from numpy import tile, concatenate, floor
    from scipy.signal import lfilter

    def Deltas(x, width=3):
        """デルタ係数計算"""
        hlen = int(floor(width/2))
        win = list(range(hlen, -hlen-1, -1))
        xx_1 = tile(x[:, 0], (1, hlen)).reshape(hlen, -1).T
        xx_2 = tile(x[:, -1], (1, hlen)).reshape(hlen, -1).T
        xx = concatenate([xx_1, x, xx_2], axis=-1)
        D = lfilter(win, 1, xx)
        return D[:, hlen*2:]

    # 音声読み込み
    sig, fs = sf.read(str(file_path))

    # LFCC抽出
    lfccs = lfcc(sig=sig,
                 fs=fs,
                 num_ceps=num_ceps,
                 low_freq=low_freq,
                 high_freq=high_freq).T

    # デルタ追加
    if order_deltas > 0:
        feats = list()
        feats.append(lfccs)
        for d in range(order_deltas):
            feats.append(Deltas(feats[-1]))
        lfccs = np.vstack(feats)

    return lfccs


class LFCC_GMM(BaseASVModel):
    """LFCC-GMM ASV model"""

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

        print(f"LFCC-GMM model loaded for {self.track} track")
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
        # LFCC特徴抽出（デルタ付き）
        features = extract_lfcc_with_deltas(
            audio_path,
            num_ceps=20,
            order_deltas=2,
            low_freq=0,
            high_freq=4000
        )

        # スコア計算: log P(X|bonafide) - log P(X|spoof)
        score_bona = self.gmm_bona.score(features.T)
        score_spoof = self.gmm_spoof.score(features.T)
        score = score_bona - score_spoof

        return float(score)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: uv run python PA/lfcc_gmm.py <file_list.txt> <output_scores.txt>")
        print("  file_list.txt: Text file containing audio file paths (one per line)")
        print("  output_scores.txt: Output file for scores")
        sys.exit(1)

    file_list_path = sys.argv[1]
    output_path = sys.argv[2]
    model_path = "PA/pretrained/LFCC_GMM.pkl"

    try:
        # Load model
        print(f"Loading LFCC-GMM model from {model_path}...")
        model = LFCC_GMM(model_path, track="PA")

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

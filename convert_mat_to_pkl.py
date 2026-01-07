"""
.mat GMM → .pkl 変換スクリプト

Usage:
    uv run python GMM_training/convert_mat_to_pkl.py <input.mat> [output.pkl]
"""

import sys
import scipy.io as sio
import numpy as np
import pickle
from pathlib import Path


def convert_mat_to_pkl(mat_path: str, pkl_path: str = None):
    """
    .matファイルをscikit-learn GMM互換の.pklに変換

    .matの構造:
    - genuineGMM, spoofGMM: struct with fields m, s, w
    - m: means (n_features, n_components)
    - s: covariances (n_features, n_components)
    - w: weights (n_components, 1)

    .pklの構造:
    - dict with keys 'bona', 'spoof'
    - each value is tuple: (weights, means, covariances, precisions_cholesky)
    """
    mat_path = Path(mat_path)
    if pkl_path is None:
        pkl_path = mat_path.with_suffix('.pkl')
    else:
        pkl_path = Path(pkl_path)

    print(f"Converting: {mat_path} -> {pkl_path}")

    mat = sio.loadmat(str(mat_path))

    def extract_gmm_params(gmm_struct):
        """MATLAB GMM structからパラメータを抽出"""
        gmm_data = gmm_struct[0, 0]

        # MATLABから読み込み (n_features, n_components)
        means = gmm_data['m']
        covars = gmm_data['s']
        weights = gmm_data['w'].flatten()

        # scikit-learn形式に転置 (n_components, n_features)
        means = means.T
        covars = covars.T
        precisions_cholesky = 1.0 / np.sqrt(covars)

        # scikit-learn GaussianMixture._get_parameters()形式
        return (weights, means, covars, precisions_cholesky)

    gmm_dict = {
        'bona': extract_gmm_params(mat['genuineGMM']),
        'spoof': extract_gmm_params(mat['spoofGMM']),
    }

    with open(pkl_path, 'wb') as f:
        pickle.dump(gmm_dict, f)

    print(f"  Bonafide: {gmm_dict['bona'][1].shape[0]} components, {gmm_dict['bona'][1].shape[1]} features")
    print(f"  Spoof: {gmm_dict['spoof'][1].shape[0]} components, {gmm_dict['spoof'][1].shape[1]} features")
    print(f"  Saved to: {pkl_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: uv run python GMM_training/convert_mat_to_pkl.py <input.mat> [output.pkl]")
        sys.exit(1)

    mat_path = sys.argv[1]
    pkl_path = sys.argv[2] if len(sys.argv) > 2 else None
    convert_mat_to_pkl(mat_path, pkl_path)

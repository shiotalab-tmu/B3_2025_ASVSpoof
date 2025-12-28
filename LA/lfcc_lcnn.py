"""LFCC-LCNN model for LA track"""

import sys
from pathlib import Path
from typing import Union
import torch
import numpy as np

# 既存のLFCC-LCNNベースラインをインポートパスに追加
baseline_path = Path(__file__).parent.parent / "ASVSpoof2021_baseline_system" / "LA" / "Baseline-LFCC-LCNN"
sys.path.insert(0, str(baseline_path))
sys.path.insert(0, str(baseline_path / "project" / "baseline_LA"))

from model import Model  # noqa: E402
from common.base_model import BaseASVModel  # noqa: E402
from common.audio_utils import load_audio  # noqa: E402


class LFCC_LCNN(BaseASVModel):
    """LFCC-LCNN ASV model"""

    def __init__(self, model_path: Union[str, Path], track: str = "LA"):
        """
        Args:
            model_path: pretrained model path (.pt)
            track: 'LA' or 'PA'
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = None

        super().__init__(model_path, track)

    def load_model(self):
        """モデルをロード"""
        # モデルを初期化（入力次元と出力次元は推論時には不要だが、ダミーで設定）
        # 実際にはモデルは音声ファイルパスから自動的に特徴抽出を行う
        self.model = Model(
            in_dim=1,  # ダミー（実際には使用されない）
            out_dim=1,  # ダミー
            args=None,
            prj_conf=type('obj', (object,), {'optional_argument': ['']})(),
            mean_std=None
        )

        # Checkpointをロード
        checkpoint = torch.load(str(self.model_path), map_location=self.device)

        # 2種類のチェックポイント形式に対応
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['state_dict'])
        else:
            self.model.load_state_dict(checkpoint)

        self.model = self.model.to(self.device)
        self.model.eval()

        print(f"LFCC-LCNN model loaded for {self.track} track")
        print(f"Device: {self.device}")

    def predict(self, audio_path: Union[str, Path]) -> float:
        """
        音声ファイルから推論

        Args:
            audio_path: 音声ファイルパス

        Returns:
            score: スコア（正の値ならbonafide、負の値ならspoof）
        """
        # 音声を読み込み（16kHz）
        audio = load_audio(audio_path, target_sr=16000)

        # Tensorに変換 (1, n_samples)
        audio_tensor = torch.FloatTensor(audio).unsqueeze(0).to(self.device)

        # 推論
        with torch.no_grad():
            # fileinfoは "filepath,label" 形式を期待
            # ダミーラベルとしてbonafideを使用
            fileinfo = [f"{str(audio_path)},bonafide"]
            output = self.model(audio_tensor, fileinfo)

            # スコアを取得
            if isinstance(output, torch.Tensor):
                score = output.item() if output.numel() == 1 else output[0].item()
            else:
                score = float(output)

        return score


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        audio_path = "ymgt.wav"
        print(f"No audio path specified, using default: {audio_path}")
    else:
        audio_path = sys.argv[1]
    model_path = "LA/pretrained/trained_network.pt"

    try:
        print(f"Loading LFCC-LCNN model from {model_path}...")
        model = LFCC_LCNN(model_path, track="LA")

        print(f"Processing audio: {audio_path}")
        score = model.predict(audio_path)

        print(f"\n=== Results ===")
        print(f"Audio: {audio_path}")
        print(f"Score: {score:.6f}")
        print(f"Result: {'Bonafide' if score > 0 else 'Spoof'}")
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

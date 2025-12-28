"""RawNet2 model for PA track"""

import sys
from pathlib import Path
from typing import Union
import torch
import numpy as np
import yaml

# 既存のRawNet2モデルをインポートパスに追加
baseline_path = Path(__file__).parent.parent / "ASVSpoof2021_baseline_system" / "PA" / "Baseline-RawNet2"
sys.path.insert(0, str(baseline_path))

from model import RawNet  # noqa: E402
from common.base_model import BaseASVModel  # noqa: E402
from common.audio_utils import load_audio, pad_audio  # noqa: E402


class RawNet2(BaseASVModel):
    """
    RawNet2 end-to-end ASV model

    Note: LA/PAトラックでは、DFモデルの重みを使用します。
          モデル構造はLA/PA/DF間で完全に同一です。
    """

    def __init__(self, model_path: Union[str, Path], track: str = "PA"):
        """
        Args:
            model_path: pretrained model path (.pth)
            track: 'LA' or 'PA'
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = None
        self.target_length = 64600  # 約4秒（16kHz）

        super().__init__(model_path, track)

    def load_model(self):
        """モデルをロード"""
        # YAML設定を読み込み（ハードコード）
        model_config = {
            'nb_samp': 64600,
            'first_conv': 1024,
            'in_channels': 1,
            'filts': [20, [20, 20], [20, 128], [128, 128]],
            'blocks': [2, 4],
            'nb_fc_node': 1024,
            'gru_node': 1024,
            'nb_gru_layer': 3,
            'nb_classes': 2
        }

        # モデルを初期化
        self.model = RawNet(model_config, self.device)

        # Pretrainedモデルをロード
        self.model.load_state_dict(
            torch.load(str(self.model_path), map_location=self.device)
        )

        self.model = self.model.to(self.device)
        self.model.eval()

        print(f"RawNet2 model loaded for {self.track} track (using DF weights)")
        print(f"Device: {self.device}")

    def predict(self, audio_path: Union[str, Path]) -> float:
        """
        音声ファイルから推論

        Args:
            audio_path: 音声ファイルパス

        Returns:
            score: bonafide スコア（正の値ならbonafide、負の値ならspoof）
        """
        # 音声を読み込み（16kHz）
        audio = load_audio(audio_path, target_sr=16000)

        # 64600サンプルにパディング
        audio = pad_audio(audio, self.target_length)

        # Tensorに変換
        audio_tensor = torch.FloatTensor(audio).unsqueeze(0).to(self.device)

        # 推論
        with torch.no_grad():
            output = self.model(audio_tensor)
            # batch_out[:, 1] でbonafideクラスのスコアを取得
            score = output[0, 1].item()

        return score


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: uv run python PA/rawnet2.py <file_list.txt> <output_scores.txt>")
        print("  file_list.txt: Text file containing audio file paths (one per line)")
        print("  output_scores.txt: Output file for scores")
        sys.exit(1)

    file_list_path = sys.argv[1]
    output_path = sys.argv[2]
    model_path = "PA/pretrained/rawnet2_model.pth"

    try:
        # Load model
        print(f"Loading RawNet2 model from {model_path}...")
        model = RawNet2(model_path, track="PA")

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

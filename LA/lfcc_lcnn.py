"""LFCC-LCNN model for LA track"""

import sys
from pathlib import Path
from typing import Union
import torch

# 既存のLFCC-LCNNベースラインをインポートパスに追加
baseline_path = Path(__file__).parent.parent / "ASVSpoof2021_baseline_system" / "LA" / "Baseline-LFCC-LCNN"
sys.path.insert(0, str(baseline_path))

from common.base_model import BaseASVModel  # noqa: E402
from common.audio_utils import load_audio  # noqa: E402


class LFCC_LCNN(BaseASVModel):
    """LFCC-LCNN ASV model (direct model loading)"""

    def __init__(self, model_path: Union[str, Path], track: str = "LA"):
        """
        Args:
            model_path: pretrained model path (.pt)
            track: 'LA' or 'PA'
        """
        self.device = 'cpu'  # Force CPU
        self.model = None
        super().__init__(model_path, track)

    def load_model(self):
        """モデルをロード"""
        # baseline model.pyをインポート
        import importlib.util
        model_py_path = Path(__file__).parent.parent / "ASVSpoof2021_baseline_system" / self.track / "Baseline-LFCC-LCNN" / "project" / f"baseline_{self.track}" / "model.py"

        spec = importlib.util.spec_from_file_location("baseline_model", model_py_path)
        baseline_model = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(baseline_model)

        # ダミーのargsとprj_confを作成
        class DummyArgs:
            def __init__(self):
                pass

        class DummyConf:
            def __init__(self):
                self.optional_argument = ['']

        args = DummyArgs()
        prj_conf = DummyConf()

        # モデルを初期化（in_dim=1, out_dim=1）
        self.model = baseline_model.Model(in_dim=1, out_dim=1, args=args, prj_conf=prj_conf)

        # Pretrainedモデルをロード（CPU強制）
        checkpoint = torch.load(str(self.model_path), map_location='cpu')
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

        # Tensorに変換 (batch, length, 1)
        audio_tensor = torch.FloatTensor(audio).unsqueeze(0).unsqueeze(-1).to(self.device)

        # 推論
        with torch.no_grad():
            # Model.forward returns None during inference, but prints the score
            # We need to capture the embedding instead
            feature_vec = self.model._compute_embedding(audio_tensor, [len(audio)])
            score = self.model._compute_score(feature_vec, inference=True)

        # スコアの平均を返す（submodelsがある場合）
        return float(score.mean().item())


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: uv run python LA/lfcc_lcnn.py <file_list.txt> <output_scores.txt>")
        print("  file_list.txt: Text file containing audio file paths (one per line)")
        print("  output_scores.txt: Output file for scores")
        sys.exit(1)

    file_list_path = sys.argv[1]
    output_path = sys.argv[2]
    model_path = "LA/pretrained/LFCC_LCNN.pt"

    try:
        # Load model
        print(f"Loading LFCC-LCNN model from {model_path}...")
        model = LFCC_LCNN(model_path, track="LA")

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

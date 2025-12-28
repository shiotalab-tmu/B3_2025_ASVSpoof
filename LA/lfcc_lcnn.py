"""LFCC-LCNN model for LA track"""

import sys
import subprocess
import tempfile
from pathlib import Path
from typing import Union

from common.base_model import BaseASVModel  # noqa: E402


class LFCC_LCNN(BaseASVModel):
    """LFCC-LCNN ASV model (uses baseline main.py via subprocess)"""

    def __init__(self, model_path: Union[str, Path], track: str = "LA"):
        """
        Args:
            model_path: pretrained model path (.pt)
            track: 'LA' or 'PA'
        """
        self.baseline_main = Path(__file__).parent.parent / "ASVSpoof2021_baseline_system" / track / "Baseline-LFCC-LCNN" / "project" / "baseline_LA" / "main.py"
        super().__init__(model_path, track)

    def load_model(self):
        """モデルをロード（サブプロセス方式では不要）"""
        if not self.baseline_main.exists():
            raise FileNotFoundError(f"Baseline main.py not found: {self.baseline_main}")
        print(f"LFCC-LCNN model ready for {self.track} track")
        print(f"Using baseline script: {self.baseline_main}")

    def predict(self, audio_path: Union[str, Path]) -> float:
        """
        音声ファイルから推論

        Args:
            audio_path: 音声ファイルパス

        Returns:
            score: スコア（正の値ならbonafide、負の値ならspoof）
        """
        # 一時ファイルリストを作成（絶対パスに変換）
        absolute_audio_path = Path(audio_path).absolute()
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(str(absolute_audio_path) + '\n')
            temp_list = f.name

        try:
            # ベースラインのmain.pyを実行（cwdをベースラインディレクトリに設定）
            baseline_dir = self.baseline_main.parent.parent.parent.absolute()
            main_py_relative = self.baseline_main.relative_to(baseline_dir)

            # PYTHONPATHにベースラインディレクトリを追加
            import os
            env = os.environ.copy()
            pythonpath = str(baseline_dir)
            if 'PYTHONPATH' in env:
                pythonpath = f"{pythonpath}:{env['PYTHONPATH']}"
            env['PYTHONPATH'] = pythonpath

            # デバッグ出力
            print(f"DEBUG: baseline_dir = {baseline_dir}")
            print(f"DEBUG: PYTHONPATH = {pythonpath}")
            print(f"DEBUG: cwd = {baseline_dir}")
            print(f"DEBUG: Command: {sys.executable} {main_py_relative}")

            result = subprocess.run(
                [
                    sys.executable,
                    str(main_py_relative),
                    '--inference',
                    '--trained-model', str(Path(self.model_path).absolute()),
                    '--test-list', str(Path(temp_list).absolute())
                ],
                capture_output=True,
                text=True,
                check=False,
                cwd=str(baseline_dir),
                env=env
            )

            # エラーチェック
            if result.returncode != 0:
                error_msg = f"Baseline script failed with exit code {result.returncode}\n"
                error_msg += f"STDOUT:\n{result.stdout}\n"
                error_msg += f"STDERR:\n{result.stderr}"
                raise RuntimeError(error_msg)

            # 標準出力から "Output, filename, label, score" をパース
            for line in result.stdout.splitlines():
                if line.startswith('Output,'):
                    parts = line.split(',')
                    score = float(parts[3].strip())
                    return score

            raise ValueError("No output line found in baseline script output")

        finally:
            # 一時ファイルを削除
            Path(temp_list).unlink(missing_ok=True)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: uv run python LA/lfcc_lcnn.py <file_list.txt> <output_scores.txt>")
        print("  file_list.txt: Text file containing audio file paths (one per line)")
        print("  output_scores.txt: Output file for scores")
        sys.exit(1)

    file_list_path = sys.argv[1]
    output_path = sys.argv[2]
    model_path = "LA/pretrained/trained_network.pt"

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

"""Audio utility functions"""

import numpy as np
import soundfile as sf
import librosa
from typing import Union
from pathlib import Path


def load_audio(audio_path: Union[str, Path], target_sr: int = 16000) -> np.ndarray:
    """
    音声ファイルを読み込み、指定のサンプリングレートにリサンプリング

    Args:
        audio_path: 音声ファイルパス
        target_sr: 目標サンプリングレート（デフォルト: 16000Hz）

    Returns:
        audio: 音声データ（numpy array, shape=(n_samples,)）
    """
    audio_path = Path(audio_path)

    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # soundfileで読み込み
    audio, sr = sf.read(str(audio_path))

    # ステレオの場合はモノラルに変換
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)

    # リサンプリング
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)

    return audio.astype(np.float32)


def pad_audio(audio: np.ndarray, target_length: int) -> np.ndarray:
    """
    音声を指定長にパディング（またはトリミング）

    Args:
        audio: 音声データ
        target_length: 目標長（サンプル数）

    Returns:
        padded_audio: パディングされた音声データ
    """
    current_length = len(audio)

    if current_length >= target_length:
        # 長い場合はトリミング
        return audio[:target_length]
    else:
        # 短い場合は繰り返しパディング
        num_repeats = int(np.ceil(target_length / current_length))
        padded = np.tile(audio, num_repeats)
        return padded[:target_length]

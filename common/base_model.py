"""Base class for all ASV models"""

from abc import ABC, abstractmethod
from typing import Union
from pathlib import Path


class BaseASVModel(ABC):
    """基底クラス：全モデルの共通インターフェース"""

    def __init__(self, model_path: Union[str, Path], track: str):
        """
        Args:
            model_path: pretrainedモデルのパス
            track: 'LA' or 'PA'
        """
        self.model_path = Path(model_path)
        self.track = track.upper()

        if self.track not in ['LA', 'PA']:
            raise ValueError(f"Invalid track: {self.track}. Must be 'LA' or 'PA'.")

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        self.load_model()

    @abstractmethod
    def load_model(self):
        """モデルロード処理（サブクラスで実装）"""
        pass

    @abstractmethod
    def predict(self, audio_path: Union[str, Path]) -> float:
        """
        推論処理（サブクラスで実装）

        Args:
            audio_path: 音声ファイルパス

        Returns:
            score: スコア値
                   正の値ならbonafide（本人音声）、負の値ならspoof（偽音声）の可能性が高い
        """
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}(track={self.track}, model={self.model_path.name})"

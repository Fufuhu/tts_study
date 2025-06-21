from pathlib import Path
from typing import Tuple, Union
from style_bert_vits2.tts_model import TTSModel
from huggingface_hub import hf_hub_download
import logging

class ModelManager:
    """
    モデルファイルのダウンロードとTTSModelインスタンスの作成を管理するクラス。
    """
    def __init__(self, model_file: str, config_file: str, style_file: str, assets_root: Union[Path, str]):
        self.model_file = model_file
        self.config_file = config_file
        self.style_file = style_file
        self.assets_root = Path(assets_root)
        self.model_path = None
        self.config_path = None
        self.style_vec_path = None

    def download_model_files(self) -> Tuple[Path, Path, Path]:
        file_descriptions = [
            (self.model_file, "モデル重みファイル"),
            (self.config_file, "設定ファイル"),
            (self.style_file, "スタイルベクトルファイル")
        ]
        for file, desc in file_descriptions:
            logging.info(f"{desc}をダウンロードします: {file} (保存先: {self.assets_root})")
            hf_hub_download("litagin/sbv2_amitaro", file, local_dir=self.assets_root)
        self.model_path = self.assets_root / self.model_file
        self.config_path = self.assets_root / self.config_file
        self.style_vec_path = self.assets_root / self.style_file
        return self.model_path, self.config_path, self.style_vec_path

    def create_tts_model(self, device: str = "cpu") -> TTSModel:
        if not all([self.model_path, self.config_path, self.style_vec_path]):
            raise RuntimeError("モデルファイルがダウンロードされていません。先にdownload_model_files()を呼んでください。")
        return TTSModel(
            model_path=self.model_path,
            config_path=self.config_path,
            style_vec_path=self.style_vec_path,
            device=device,
        )


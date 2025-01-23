import os
from typing import (Optional, Union, Dict, List)
import numpy as np
from numpy.typing import NDArray
from kokoro_onnx import Kokoro

from .model_downloader import (download_model_if_no_exists, KOKORO_MODELS_URL)


class KokoroPipeline:
    def __init__(self,
                 model_dir: str):
        self.model: Kokoro = None
        self.default_model_name = "kokoro-v0_19.onnx"
        self.model_dir: str = model_dir
        self.available_models: list = self.get_available_models()
        self.available_voices: Optional[list] = None
        self.available_langs: Optional[list] = None
        os.makedirs(self.model_dir, exist_ok=True)

    def load_model(self,
                   model_name: str,
                   voice_pack: str):
        """
        Set model instance
        """
        model_path = os.path.join(self.model_dir, model_name)
        voice_pack_path = os.path.join(self.model_dir, voice_pack)

        if not os.path.exists(model_path):
            print(f"Model {model_name} is not detected. Downloading models..")
            download_model_if_no_exists(
                model_name=model_name,
                download_dir=self.model_dir
            )

        # Device ( GPU or CPU is automatically set by kokoro-onnx )
        self.model = Kokoro(model_path, voice_pack_path)
        self.available_voices = self.model.get_voices()
        self.available_langs = self.model.get_languages()

    def predict(self,
                text: str,
                voice: str,
                lang: str = "en-us",
                speed: float = 1.0,
                phonemes: Optional[str] = None,
                trim: bool = True) -> tuple[NDArray[np.float32], int]:
        """Predict"""
        if self.model is None:
            raise ValueError("Load model first with `load_model()`")

        samples, sample_rate = self.model.create(
            text=text,
            voice=voice,
            lang=lang,
            speed=speed,
            phonemes=phonemes,
            trim=trim
        )
        return samples, sample_rate

    def get_available_models(self) -> List:
        """
        Get available models
        """
        allowed_model_extensions = ["onnx"]
        files = os.listdir(os.path.join(self.model_dir))
        models = [f for f in files if f.split(".")[-1] in allowed_model_extensions]
        models = set(models) | set(KOKORO_MODELS_URL.keys())
        return list(models)

    def get_available_voice_packs(self) -> List:
        """
        Get available voice packs
        """
        allowed_voice_packs_extensions = ["bin"]
        files = os.listdir(os.path.join(self.model_dir))
        voice_packs = [f for f in files if f.split(".")[-1] in allowed_voice_packs_extensions]
        default_voice_pack_name = KOKORO_MODELS_URL[self.default_model_name]["voice"].split("/")[-1]
        voice_packs = {default_voice_pack_name, } | set(voice_packs)
        return list(voice_packs)


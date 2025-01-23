import numpy as np
import os
import torch
from torch import Tensor
from typing import Union, BinaryIO, Dict, List, Tuple, Optional
import time

#  ComfyUI Modules
import folder_paths
from comfy.utils import ProgressBar

#  Your Modules
from .modules.kokoro.kokoro_inferencer import KokoroPipeline
from .modules.kokoro.model_downloader import KOKORO_MODELS_URL


custom_nodes_script_dir = os.path.dirname(os.path.abspath(__file__))
custom_nodes_model_dir = os.path.join(folder_paths.models_dir, "kokoro-onnx")
custom_nodes_output_dir = os.path.join(folder_paths.get_output_directory())


def get_category_name():
    return "ComfyUI jhj Kokoro Onnx"


def numpy_to_tensor(audio: np.ndarray) -> Tensor:
    return torch.from_numpy(audio).unsqueeze(0).unsqueeze(0).float()


class KokoroModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        kokoro_models = KokoroPipeline.get_available_models(custom_nodes_model_dir)
        kokoro_voice_packs = KokoroPipeline.get_available_voice_packs(custom_nodes_model_dir)

        return {
            "required": {
                "model": (kokoro_models,),
                "voice_pack": (kokoro_voice_packs,),
            }
        }

    RETURN_TYPES = ("KOKORO_ONNX",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = get_category_name()

    def load_model(self,
                   model: str,
                   voice_pack: str,
                   ) -> Tuple[KokoroPipeline]:
        pipeline = KokoroPipeline(model_dir=custom_nodes_model_dir)
        pipeline.load_model(model, voice_pack)

        return (pipeline, )


class KokoroAudioGenerator:
    @classmethod
    def INPUT_TYPES(s):
        default_model = list(KOKORO_MODELS_URL.keys())[0]

        return {
            "required": {
                "model": ("KOKORO_ONNX", ),
                "text": ("STR", ),
                "voice": (default_model["default_available_voices"], ),
                "lang": (default_model["default_available_langs"], ),
                "speed": ("FLOAT", {"default": 1.0}),
                "phonemes": ("STR", {"default": None}),
                "trim": ("BOOL", {"default": True}),
            },
            "optional": {
                "a": ("INT", {"default": 5}),
                "b": ("INT", {"default": 10}),
            }
        }

    RETURN_TYPES = ("AUDIO", "INT")
    RETURN_NAMES = ("audio", "sample_rate")
    FUNCTION = "predict"
    CATEGORY = get_category_name()

    def predict(self,
                model: KokoroPipeline,
                text: str,
                voice: str,
                lang: str = "en-us",
                speed: float = 1.0,
                phonemes: Optional[str] = None,
                trim: bool = True) -> Tuple[np.ndarray]:
        samples, sample_rate = model.predict(
            text=text,
            voice=voice,
            lang=lang,
            speed=speed,
            phonemes=phonemes,
            trim=trim
        )
        samples = numpy_to_tensor(samples)

        return (samples, sample_rate)

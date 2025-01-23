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

os.makedirs(custom_nodes_model_dir, exist_ok=True)


def get_category_name():
    return "ComfyUI jhj Kokoro Onnx"


def numpy_to_tensor(audio: np.ndarray) -> Tensor:
    tensor = torch.from_numpy(audio).unsqueeze(0).unsqueeze(0).float()
    return tensor


class KokoroModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        _instance = KokoroPipeline(custom_nodes_model_dir)
        kokoro_models = _instance.get_available_models()
        kokoro_voice_packs = _instance.get_available_voice_packs()

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
        # Hardcode defaults for not knowing how to set them dynamically with change listeners in ComfyUI
        default_model = list(KOKORO_MODELS_URL.keys())[0]
        default_voices = KOKORO_MODELS_URL[default_model]["default_available_voices"]
        default_langs = KOKORO_MODELS_URL[default_model]["default_available_langs"]

        return {
            "required": {
                "model": ("KOKORO_ONNX", ),
                "text": ("STRING", ),
                "voice": (default_voices, {"default": default_voices[0]}),
                "lang": (default_langs, {"default": default_langs[0]}),
                "speed": ("FLOAT", {"default": 1.0, "step": 0.1}),
            },
            "optional": {
                "phonemes": ("STRING", {"default": None}),
                "trim": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
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

        return ({"waveform": samples, "sample_rate": sample_rate},)

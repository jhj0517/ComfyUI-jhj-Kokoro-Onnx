import os
import torch
from typing import (Optional, Union, Dict, List)
import json

from models import (build_model, forward)
from model_downloader import (download_model_if_no_exists, KOKORO_MODELS_REPO_ID_MAP)
from ..tokenizer.tokenizer import tokenize
from ..tokenizer.normalizer import (phonemize, get_vocab)


class KokoroInferencer:
    def __init__(self,
                 model_dir: str):
        self.model = None
        self.voice_pack = None
        self.model_dir = model_dir
        self.available_models = list(self.get_models().keys()) + list(KOKORO_MODELS_REPO_ID_MAP.keys())
        self.available_voices = None
        os.makedirs(self.model_dir, exist_ok=True)

    def load_model(self,
                   model_name: str,
                   device: str):
        """
        Load models.
        Following the structure of https://huggingface.co/hexgrad/Kokoro-82M/tree/main/voices,
        It is assumed that the `voice` subdirectory is there along with it.
        """
        if model_name not in list(self.get_models().values()):
            download_model_if_no_exists(
                repo_id=KOKORO_MODELS_REPO_ID_MAP[model_name],
                download_dir=self.model_dir
            )

        model_dir_path = self.get_models()[model_name]
        config_path = os.path.join(model_dir_path, "config.json")
        if not os.path.exists(config_path):
            raise ValueError(f"The config file for the model does not exist at \"{config_path}\".")

        config = json.load(config_path)
        self.model = build_model(config, device=device)
        self.available_voices = os.listdir(os.path.join(model_dir_path, "voices"))

    def generate(self, text, voicepack, lang='a', speed=1, ps=None):
        if self.model is None:
            raise ValueError("Load model first")

        ps = ps or phonemize(text, lang)
        tokens = tokenize(ps)
        if not tokens:
            return None
        elif len(tokens) > 510:
            tokens = tokens[:510]
            print('Truncated to 510 tokens')
        ref_s = voicepack[len(tokens)]
        out_audio = forward(self.model, tokens, ref_s, speed)
        out_ps = ''.join(next(k for k, v in get_vocab().items() if i == v) for i in tokens)
        return out_audio

    def get_models(self) -> Dict:
        """
        Get model directory dictionary that mapped as {"model_name": "path/to/dir"}.
        When downloaded from huggingface, the directory name usually has the structure of `model--repo_id--repo_name`,
        which is confusing. So map the `repo_name` only at the end to display it cleanly in the UI.
        """
        model_names = os.listdir(self.model_dir)
        wrong_dirs = [".locks"]
        model_names = list(set(model_names) - set(wrong_dirs))

        model_map = dict()
        for name in model_names:
            model_path = os.path.join(self.model_dir, name)
            if "--" in name:
                name = name.split("--")[-1]
            model_map[name] = model_path
        return model_map







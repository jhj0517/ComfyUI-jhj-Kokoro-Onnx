from huggingface_hub import snapshot_download
import requests
from tqdm import tqdm
import os

KOKORO_MODELS_URL = {
    "kokoro-v0_19.onnx": {
        "model": "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/kokoro-v0_19.onnx",
        "voice": "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/voices.bin",
        "default_available_voices": ['af', 'af_bella', 'af_nicole', 'af_sarah', 'af_sky', 'am_adam', 'am_michael', 'bf_emma', 'bf_isabella', 'bm_george', 'bm_lewis'],
        "default_available_langs": ['en-us', 'en-gb', 'fr-fr', 'ja', 'ko', 'cmn']
    }
}


def download_model_if_no_exists(
    model_name: str,
    download_dir: str
):
    model_url = KOKORO_MODELS_URL.get(model_name, None).get("model", None)
    if model_url is None:
        raise ValueError(f"Model {model_name} not found in the list of available models.")
    voice_url = KOKORO_MODELS_URL.get(model_name, None).get("voice", None)
    download_file(model_url, os.path.join(download_dir, model_name))

    if voice_url is not None:
        download_file(voice_url, os.path.join(download_dir, "voices.bin"))


def download_file(url: str, save_path: str):
    """Download a file from a URL"""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))

        with open(save_path, "wb") as f, tqdm(
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
            desc=f"Downloading {save_path}"
        ) as progress_bar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                progress_bar.update(len(chunk))
        print(f"Downloaded: {save_path}")

    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Failed to download {url} to \"{save_path}\": {e}")
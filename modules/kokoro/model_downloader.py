from huggingface_hub import snapshot_download
import requests
import os

KOKORO_MODELS_URL = {
    "kokoro-v0_19": {
        "model": "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/kokoro-v0_19.onnx",
        "voice": "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/voices.bin"
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
    download_file(model_url, os.path.join(download_dir, f"{model_name}.onnx"))

    if voice_url is not None:
        download_file(voice_url, os.path.join(download_dir, "voices.bin"))


def download_file(url: str, save_path: str):
    """Download a file from a URL to a specified path."""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(save_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Downloaded: {save_path}")
    except requests.exceptions.RequestException as e:
        print(f"Failed to download {url} to \"{save_path}\": {e}")
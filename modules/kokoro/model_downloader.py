from huggingface_hub import snapshot_download
import os

KOKORO_MODELS_REPO_ID_MAP = {
    "Kokoro-82M": "hexgrad/Kokoro-82M"
}


def download_model_if_no_exists(
    repo_id: str,
    download_dir: str
):
    try:
        simplified_name = repo_id
        _id, _name = repo_id.split("/")
        structured_name = f"models--{_id}--{_name}"
        simplified_path, structured_path = os.path.join(download_dir, simplified_name), os.path.join(download_dir, structured_name)
        if os.path.exists(download_dir) and os.listdir(simplified_path):
            return True

        if os.path.exists(download_dir) and os.listdir(structured_path):
            return True

        return snapshot_download(repo_id=repo_id,
                                 allow_patterns=['*.pt', '*.pth', '*.yml', "*.json"],
                                 cache_dir=download_dir,
                                 resume_download=True)
    except Exception as e:
        raise ValueError(f"Failed to download model. Check if repo_id is valid. Error : {e}")


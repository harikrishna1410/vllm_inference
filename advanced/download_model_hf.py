import os

from huggingface_hub import hf_hub_download, snapshot_download


def download_model(repo_id, hf_home=None, token=None):
    """
    Download a model from Hugging Face Hub

    Args:
        repo_id (str): Model repository ID (e.g., 'bert-base-uncased')
        hf_home (str): Local directory to save the model
        token (str): Hugging Face API token (optional, will use HF_TOKEN env var if not provided)
    """
    try:
        # Use environment variable if token not provided
        if token is None:
            token = os.getenv("HUGGINGFACE_HUB_TOKEN")

        if hf_home is None:
            # Use HF_HOME or default location
            hf_home = os.getenv("HF_HOME", f"{os.getcwd()}/.cache")

        os.environ["HF_HOME"] = hf_home
        os.environ["HF_HUB_CACHE"] = os.path.join(hf_home, "hub")

        cache_dir = os.environ["HF_HUB_CACHE"]
        print(f"Downloading model '{repo_id}' to '{cache_dir}'...")

        # Download entire model repository
        snapshot_download(repo_id, token=token, cache_dir=cache_dir)

        print(f"Model downloaded successfully to: {hf_home}")
        return hf_home

    except Exception as e:
        print(f"Error downloading model: {e}")
        return None


# Example usage
if __name__ == "__main__":
    # Replace with desired model
    model_name = "meta-llama/Llama-3.2-3B-Instruct"

    # Download the model
    download_path = download_model(model_name)

    if download_path:
        print(f"Model ready at: {download_path}")

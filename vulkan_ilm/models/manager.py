"""
Model management utilities for VulkanIlm.
"""
import json
import logging
import os
import shutil
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlparse

try:
    from huggingface_hub import HfApi, hf_hub_download, list_repo_files, snapshot_download
    from huggingface_hub.utils import RepositoryNotFoundError, RevisionNotFoundError

    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


class ModelManager:
    """
    Manages LLM models for VulkanIlm.

    This class handles model discovery, downloading, and validation.
    """

    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize a new ModelManager.

        Args:
            cache_dir: Directory to store downloaded models
                       (defaults to ~/.vulkan_ilm/models)
        """
        self.cache_dir = cache_dir or os.path.expanduser("~/.vulkan_ilm/models")
        os.makedirs(self.cache_dir, exist_ok=True)

        self.logger = logging.getLogger("vulkan_ilm.models")

        if HF_AVAILABLE:
            self.api = HfApi()

    def list_local_models(self) -> List[str]:
        """
        List all locally available models.

        Returns:
            List of model file paths
        """
        models = []
        for root, _, files in os.walk(self.cache_dir):
            for file in files:
                if file.endswith((".gguf", ".bin")):
                    models.append(os.path.join(root, file))
        return models

    def search_models(self, query: str = "", limit: int = 20) -> List[Dict[str, Any]]:
        """
        Search for compatible models on HuggingFace.

        Args:
            query: Search query string
            limit: Maximum number of results to return

        Returns:
            List of model information dictionaries
        """
        if not HF_AVAILABLE:
            raise ImportError(
                "huggingface_hub is required for model search. "
                "Install with: pip install huggingface_hub"
            )

        try:
            # First try with "gguf" tag
            results = self.api.list_models(filter="gguf", search=query, limit=limit)

            # If no results, try with a broader search
            if not results:
                results = self.api.list_models(search=f"{query} gguf", limit=limit)

            # Convert to a more useful format
            models = []
            for model in results:
                # Try to get more details from the model
                try:
                    # Get model details
                    details = {
                        "id": model.id,
                        "downloads": model.downloads,
                        "last_modified": model.last_modified,
                        "tags": model.tags,
                        "files": [],
                    }

                    # Try to list files to find GGUF models
                    try:
                        files = list_repo_files(model.id)
                        gguf_files = [f for f in files if f.endswith(".gguf")]
                        if gguf_files:
                            details["files"] = gguf_files
                    except:
                        # If we can't list files, just skip this part
                        pass

                    models.append(details)
                except:
                    # If we can't get details, just add the basic info
                    models.append(
                        {
                            "id": model.id,
                            "downloads": model.downloads,
                        }
                    )

            return models

        except Exception as e:
            self.logger.error(f"Error searching for models: {e}")
            return []

    def download_model(self, model_id: str, filename: Optional[str] = None) -> str:
        """
        Download a model from HuggingFace.

        Args:
            model_id: HuggingFace model ID (e.g., 'TheBloke/Llama-2-7B-Chat-GGUF')
            filename: Specific filename to download (if None, will download all GGUF files)

        Returns:
            Path to the downloaded model file
        """
        if not HF_AVAILABLE:
            raise ImportError(
                "huggingface_hub is required for model download. "
                "Install with: pip install huggingface_hub"
            )

        try:
            # Check if it's a direct URL
            if model_id.startswith(("http://", "https://")):
                return self._download_from_url(model_id)

            # Create a directory for this model
            model_dir = os.path.join(self.cache_dir, model_id.replace("/", "_"))
            os.makedirs(model_dir, exist_ok=True)

            # If a specific filename is provided, download just that file
            if filename:
                try:
                    path = hf_hub_download(
                        repo_id=model_id,
                        filename=filename,
                        cache_dir=model_dir,
                        resume_download=True,
                    )
                    return path
                except Exception as e:
                    raise ValueError(f"Error downloading {filename} from {model_id}: {e}")

            # Otherwise, try to download all GGUF files
            try:
                # List files in the repo
                files = list_repo_files(model_id)

                # Filter for GGUF files
                gguf_files = [f for f in files if f.endswith(".gguf")]

                if not gguf_files:
                    raise ValueError(f"No GGUF files found in {model_id}")

                # Download each file
                paths = []
                for file in gguf_files:
                    path = hf_hub_download(
                        repo_id=model_id, filename=file, cache_dir=model_dir, resume_download=True
                    )
                    paths.append(path)

                # Return the path to the first file
                return paths[0]

            except Exception as e:
                # Fallback: try snapshot_download with pattern
                try:
                    path = snapshot_download(
                        repo_id=model_id,
                        cache_dir=model_dir,
                        allow_patterns="*.gguf",
                        resume_download=True,
                    )

                    # Find GGUF files in the downloaded directory
                    gguf_files = []
                    for root, _, files in os.walk(path):
                        for file in files:
                            if file.endswith(".gguf"):
                                gguf_files.append(os.path.join(root, file))

                    if not gguf_files:
                        raise ValueError(f"No GGUF files found in {model_id}")

                    # Return the first GGUF file
                    return gguf_files[0]

                except Exception as nested_e:
                    raise ValueError(f"Error downloading from {model_id}: {e} -> {nested_e}")

        except Exception as e:
            raise ValueError(f"Failed to download model: {e}")

    def _download_from_url(self, url: str) -> str:
        """
        Download a model from a direct URL.

        Args:
            url: URL to the model file

        Returns:
            Path to the downloaded model file
        """
        import requests
        from tqdm import tqdm

        # Parse the URL to get the filename
        parsed_url = urlparse(url)
        filename = os.path.basename(parsed_url.path)

        if not filename.endswith((".gguf", ".bin")):
            raise ValueError(f"URL does not point to a supported model file: {url}")

        # Create the destination path
        dest_path = os.path.join(self.cache_dir, "downloads", filename)
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)

        # Check if the file already exists
        if os.path.exists(dest_path):
            return dest_path

        # Download the file with progress bar
        self.logger.info(f"Downloading {url} to {dest_path}")

        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total_size = int(r.headers.get("content-length", 0))

            with open(dest_path, "wb") as f, tqdm(
                desc=filename,
                total=total_size,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
            ) as progress:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        progress.update(len(chunk))

        return dest_path

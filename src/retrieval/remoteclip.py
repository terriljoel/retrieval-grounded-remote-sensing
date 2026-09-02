from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image


class RemoteClipEncoder:
    """Lazy runtime wrapper matching notebook 04's RemoteCLIP contract."""

    def __init__(
        self,
        *,
        model_name: str,
        repository: str,
        filename: str,
        cache_dir: Path,
        device: str | None = None,
        expected_dimension: int = 512,
    ):
        try:
            import open_clip
            import torch
            from huggingface_hub import hf_hub_download
        except ImportError as error:
            raise RuntimeError(
                "RemoteCLIP runtime dependencies are missing. Install .[annotation]."
            ) from error

        self._torch = torch
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        cache_dir = Path(cache_dir).resolve()
        cache_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = hf_hub_download(
            repo_id=repository,
            filename=filename,
            cache_dir=str(cache_dir),
        )

        model, _, preprocess = open_clip.create_model_and_transforms(model_name)
        try:
            checkpoint = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=True,
            )
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        model.load_state_dict(checkpoint)
        self.model = model.to(self.device).eval().requires_grad_(False)
        self.preprocess = preprocess
        self.expected_dimension = expected_dimension

    def encode(self, images: Iterable[Image.Image]) -> np.ndarray:
        torch = self._torch
        image_list = list(images)
        if not image_list:
            raise ValueError("At least one image is required for embedding")
        batch = torch.stack([
            self.preprocess(image.convert("RGB")) for image in image_list
        ]).to(self.device)
        with torch.inference_mode():
            with torch.autocast(
                device_type=self.device.type,
                dtype=torch.float16,
                enabled=self.device.type == "cuda",
            ):
                embeddings = self.model.encode_image(batch)
            embeddings = torch.nn.functional.normalize(
                embeddings.float(), p=2, dim=-1
            )
        vectors = embeddings.cpu().numpy().astype(np.float32)
        if vectors.shape != (len(image_list), self.expected_dimension):
            raise ValueError(
                f"Unexpected RemoteCLIP shape {vectors.shape}; expected "
                f"({len(image_list)}, {self.expected_dimension})"
            )
        if not np.isfinite(vectors).all():
            raise ValueError("RemoteCLIP returned non-finite values")
        return vectors


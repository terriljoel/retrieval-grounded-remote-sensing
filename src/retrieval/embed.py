"""Batch embedding with plain .npy caching.

Row order always matches the crop_index order it was given, so an embedding
matrix and the index can be zipped positionally without a join.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from src.retrieval.encoders import l2_normalise, load_encoder

BATCH_SIZE = 64


def embedding_path(cache_root: Path, encoder_key: str, stream: str, rows: int) -> Path:
    # Row count is in the filename so a --limit smoke run cannot overwrite the
    # full-corpus cache and force a re-embed of everything.
    return cache_root / f"{encoder_key}_{stream}_{rows}.npy"


def embed_stream(
    crop_ids: list[str],
    crop_root: Path,
    cache_root: Path,
    encoder_key: str,
    stream: str,
    device: str | None = None,
    batch_size: int = BATCH_SIZE,
    force: bool = False,
) -> np.ndarray:
    """Return L2-normalised float32 embeddings, one row per crop_id."""
    import torch

    path = embedding_path(cache_root, encoder_key, stream, len(crop_ids))
    if path.is_file() and not force:
        cached = np.load(path)
        print(f"  cached {cache_root.name}/{encoder_key}/{stream}: {cached.shape}")
        return cached

    encode, preprocess = load_encoder(encoder_key, device)
    stream_dir = crop_root / stream

    rows: list[np.ndarray] = []
    for start in range(0, len(crop_ids), batch_size):
        chunk = crop_ids[start : start + batch_size]
        tensors = []
        for crop_id in chunk:
            with Image.open(stream_dir / f"{crop_id}.jpg") as image:
                tensors.append(preprocess(image.convert("RGB")))
        rows.append(encode(torch.stack(tensors)))
        done = min(start + batch_size, len(crop_ids))
        print(f"\r  {encoder_key}/{stream}: {done}/{len(crop_ids)}", end="", flush=True)
    print()

    embeddings = l2_normalise(np.concatenate(rows, axis=0))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, embeddings)
    return embeddings

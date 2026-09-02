from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.annotation.models import Box


@dataclass(frozen=True)
class EvidenceRecord:
    embedding_id: str
    image_id: str
    image_key: str
    source_path: Path
    split: str
    class_id: int
    class_name: str
    crop_type: str
    box: Box
    context_margin: float
    cosine_similarity: float


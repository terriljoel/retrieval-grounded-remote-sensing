from __future__ import annotations

from dataclasses import dataclass

from PIL import Image

from src.annotation.models import DetectionSuggestion
from src.retrieval.crops import crop_image
from src.retrieval.lancedb_store import LanceDbEvidenceStore
from src.retrieval.models import EvidenceRecord
from src.retrieval.remoteclip import RemoteClipEncoder


@dataclass(frozen=True)
class DetectionEvidence:
    object_crop: Image.Image
    context_crop: Image.Image
    object_cases: list[EvidenceRecord]
    context_cases: list[EvidenceRecord]


def retrieve_detection_evidence(
    *,
    image: Image.Image,
    detection: DetectionSuggestion,
    encoder: RemoteClipEncoder,
    store: LanceDbEvidenceStore,
    context_margin: float,
    evidence_splits: tuple[str, ...],
    top_k: int,
    candidate_limit: int,
    query_image_id: str | None = None,
) -> DetectionEvidence:
    object_crop = crop_image(image, detection.box, margin_fraction=0.0)
    context_crop = crop_image(
        image, detection.box, margin_fraction=context_margin
    )
    object_vector, context_vector = encoder.encode([object_crop, context_crop])
    return DetectionEvidence(
        object_crop=object_crop,
        context_crop=context_crop,
        object_cases=store.search(
            object_vector,
            crop_type="object",
            evidence_splits=evidence_splits,
            top_k=top_k,
            candidate_limit=candidate_limit,
            exclude_image_id=query_image_id,
        ),
        context_cases=store.search(
            context_vector,
            crop_type="context",
            evidence_splits=evidence_splits,
            top_k=top_k,
            candidate_limit=candidate_limit,
            exclude_image_id=query_image_id,
        ),
    )


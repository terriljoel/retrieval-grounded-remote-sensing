from pathlib import Path

from src.annotation.models import Box, DetectionSuggestion
from src.annotation.policy import evaluate_acceptance_policy
from src.retrieval.models import EvidenceRecord
from src.vlm.nim import VlmAssessment


def _assessment(decision: str = "accept", confidence: float = 0.95) -> VlmAssessment:
    return VlmAssessment(decision, "airplane", confidence, "", "", (), "")


def _settings() -> dict:
    return {
        "detector_min_confidence": 0.80,
        "retrieval_min_cosine_similarity": 0.90,
        "retrieval_min_supporting_neighbors": 3,
        "vlm_min_confidence": 0.85,
        "low_confidence_rescue": {
            "enabled": True,
            "detector_min_confidence": 0.20,
            "detector_max_confidence": 0.50,
            "retrieval_min_cosine_similarity": 0.92,
            "retrieval_min_supporting_neighbors": 3,
            "query_vlm_min_confidence": 0.90,
            "grounded_vlm_min_confidence": 0.90,
        },
    }


def _evidence(index: int) -> EvidenceRecord:
    return EvidenceRecord(
        f"e{index}", f"i{index}", f"{index}.jpg", Path(f"{index}.jpg"),
        "train", 0, "airplane", "object", Box(0, 0, 10, 10), 0.25, 0.95,
    )


def test_low_confidence_detection_can_be_rescued():
    result = evaluate_acceptance_policy(
        detection=DetectionSuggestion("d1", 0, "airplane", 0.35, Box(0, 0, 10, 10)),
        evidence=[_evidence(index) for index in range(3)],
        assessment=_assessment(),
        query_assessment=_assessment(),
        settings=_settings(),
    )

    assert result.recommendation == "accept"
    assert result.rescue_passed
    assert result.acceptance_path == "low_confidence_rescue"
    assert result.reasons == ()


def test_rescue_requires_query_only_vlm_agreement():
    result = evaluate_acceptance_policy(
        detection=DetectionSuggestion("d1", 0, "airplane", 0.35, Box(0, 0, 10, 10)),
        evidence=[_evidence(index) for index in range(3)],
        assessment=_assessment(),
        query_assessment=_assessment("human_review"),
        settings=_settings(),
    )

    assert result.recommendation == "human_review"
    assert not result.rescue_passed

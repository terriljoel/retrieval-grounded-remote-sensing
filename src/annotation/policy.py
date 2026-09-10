from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from src.annotation.models import DetectionSuggestion
from src.retrieval.models import EvidenceRecord
from src.vlm.nim import VlmAssessment


@dataclass(frozen=True)
class AcceptancePolicyResult:
    recommendation: str
    detector_passed: bool
    retrieval_passed: bool
    vlm_passed: bool
    supporting_neighbors: int
    required_supporting_neighbors: int
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["reasons"] = list(self.reasons)
        return result


def evaluate_acceptance_policy(
    *,
    detection: DetectionSuggestion,
    evidence: list[EvidenceRecord],
    assessment: VlmAssessment,
    settings: dict[str, Any],
) -> AcceptancePolicyResult:
    detector_minimum = float(settings["detector_min_confidence"])
    similarity_minimum = float(settings["retrieval_min_cosine_similarity"])
    required_neighbors = int(settings["retrieval_min_supporting_neighbors"])
    vlm_minimum = float(settings["vlm_min_confidence"])

    supporting = [
        item
        for item in evidence
        if item.class_name == detection.class_name
        and item.cosine_similarity >= similarity_minimum
    ]
    detector_passed = detection.confidence >= detector_minimum
    retrieval_passed = len(supporting) >= required_neighbors
    vlm_passed = (
        assessment.decision == "accept"
        and assessment.confidence is not None
        and assessment.confidence >= vlm_minimum
        and assessment.suggested_class in {None, detection.class_name}
    )

    reasons: list[str] = []
    if not detector_passed:
        reasons.append(
            f"detector confidence {detection.confidence:.3f} is below "
            f"{detector_minimum:.3f}"
        )
    if not retrieval_passed:
        reasons.append(
            f"only {len(supporting)} same-class neighbours have cosine "
            f"similarity >= {similarity_minimum:.3f}; {required_neighbors} required"
        )
    if not vlm_passed:
        confidence = (
            "missing" if assessment.confidence is None
            else f"{assessment.confidence:.3f}"
        )
        reasons.append(
            f"VLM decision={assessment.decision!r}, confidence={confidence}, "
            f"suggested_class={assessment.suggested_class!r}; accept with confidence "
            f">= {vlm_minimum:.3f} and no conflicting class required"
        )

    accepted = detector_passed and retrieval_passed and vlm_passed
    return AcceptancePolicyResult(
        recommendation="accept" if accepted else "human_review",
        detector_passed=detector_passed,
        retrieval_passed=retrieval_passed,
        vlm_passed=vlm_passed,
        supporting_neighbors=len(supporting),
        required_supporting_neighbors=required_neighbors,
        reasons=tuple(reasons),
    )

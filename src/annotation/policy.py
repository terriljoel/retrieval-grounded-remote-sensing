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
    query_vlm_passed: bool
    rescue_passed: bool
    acceptance_path: str
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
    query_assessment: VlmAssessment | None = None,
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

    rescue = settings.get("low_confidence_rescue") or {}
    rescue_enabled = bool(rescue.get("enabled", False))
    rescue_supporting = [
        item for item in evidence
        if item.class_name == detection.class_name
        and item.cosine_similarity
        >= float(rescue.get("retrieval_min_cosine_similarity", 1.0))
    ]
    query_vlm_passed = bool(
        query_assessment
        and query_assessment.decision == "accept"
        and query_assessment.confidence is not None
        and query_assessment.confidence
        >= float(rescue.get("query_vlm_min_confidence", 1.0))
        and query_assessment.suggested_class in {None, detection.class_name}
    )
    rescue_grounded_passed = (
        assessment.decision == "accept"
        and assessment.confidence is not None
        and assessment.confidence
        >= float(rescue.get("grounded_vlm_min_confidence", 1.0))
        and assessment.suggested_class in {None, detection.class_name}
    )
    rescue_passed = (
        rescue_enabled
        and float(rescue["detector_min_confidence"]) <= detection.confidence
        < float(rescue["detector_max_confidence"])
        and len(rescue_supporting)
        >= int(rescue["retrieval_min_supporting_neighbors"])
        and query_vlm_passed
        and rescue_grounded_passed
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

    standard_passed = detector_passed and retrieval_passed and vlm_passed
    accepted = standard_passed or rescue_passed
    if accepted:
        reasons = []
    return AcceptancePolicyResult(
        recommendation="accept" if accepted else "human_review",
        detector_passed=detector_passed,
        retrieval_passed=retrieval_passed,
        vlm_passed=vlm_passed,
        query_vlm_passed=query_vlm_passed,
        rescue_passed=rescue_passed,
        acceptance_path=(
            "standard" if standard_passed
            else "low_confidence_rescue" if rescue_passed
            else "none"
        ),
        supporting_neighbors=len(supporting),
        required_supporting_neighbors=required_neighbors,
        reasons=tuple(reasons),
    )

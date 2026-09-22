from __future__ import annotations

from pathlib import Path
from typing import Any

from src.annotation.models import NWPU_CLASSES
from src.detection.config import load_yaml_config, resolve_path


REQUIRED_SECTIONS = {
    "experiment",
    "cases",
    "classes",
    "embedding",
    "retrieval",
    "vlm",
    "decision_policy",
    "execution",
}
CASE_STATUSES = {"true_positive", "false_positive", "class_error"}


def validate_assistance_experiment_config(config: dict[str, Any]) -> None:
    missing = REQUIRED_SECTIONS - set(config)
    if missing:
        raise ValueError(f"Missing experiment configuration sections: {sorted(missing)}")
    if tuple(config["classes"]) != NWPU_CLASSES:
        raise ValueError("The comparison requires the fixed NWPU VHR-10 class order")

    cases = config["cases"]
    if not cases.get("inference_directory"):
        raise ValueError("cases.inference_directory must not be empty")
    if cases["inference_directory"] == "latest" and not cases.get("inference_root"):
        raise ValueError(
            "cases.inference_root is required when inference_directory=latest"
        )
    if bool(cases.get("manifest")) != bool(cases.get("manifest_dataset_root")):
        raise ValueError(
            "cases.manifest and cases.manifest_dataset_root must be set together"
        )
    statuses = set(cases.get("statuses") or [])
    if not statuses or not statuses <= CASE_STATUSES:
        raise ValueError(f"cases.statuses must be drawn from {sorted(CASE_STATUSES)}")
    maximum = cases.get("maximum_per_status")
    if maximum is not None and int(maximum) <= 0:
        raise ValueError("cases.maximum_per_status must be null or positive")
    minimum_confidence = float(cases.get("minimum_detector_confidence", 0.0))
    if not 0.0 <= minimum_confidence <= 1.0:
        raise ValueError("cases.minimum_detector_confidence must be between 0 and 1")
    maximum_confidence = cases.get("maximum_detector_confidence")
    if maximum_confidence is not None:
        maximum_confidence = float(maximum_confidence)
        if not minimum_confidence < maximum_confidence <= 1.0:
            raise ValueError(
                "cases.maximum_detector_confidence must be greater than the "
                "minimum and at most 1"
            )

    retrieval = config["retrieval"]
    if not retrieval.get("database_path") and not retrieval.get("artifact_root"):
        raise ValueError("Set retrieval.database_path or retrieval.artifact_root")
    if not retrieval.get("evidence_splits"):
        raise ValueError("retrieval.evidence_splits must not be empty")
    top_k = int(retrieval.get("top_k", 0))
    if top_k <= 0:
        raise ValueError("retrieval.top_k must be positive")

    vlm = config["vlm"]
    if float(vlm.get("requests_per_minute", 30)) <= 0:
        raise ValueError("vlm.requests_per_minute must be positive")
    if int(vlm.get("max_retries", 3)) < 0:
        raise ValueError("vlm.max_retries must not be negative")
    if float(vlm.get("retry_initial_delay_seconds", 1.0)) < 0:
        raise ValueError("vlm.retry_initial_delay_seconds must not be negative")
    if float(vlm.get("retry_delay_increment_seconds", 1.0)) < 0:
        raise ValueError("vlm.retry_delay_increment_seconds must not be negative")
    if int(vlm.get("query_image_max_size", 1024)) <= 0:
        raise ValueError("vlm.query_image_max_size must be positive")
    max_evidence = int(vlm.get("max_evidence", 0))
    if not 1 <= max_evidence <= top_k:
        raise ValueError("vlm.max_evidence must be between 1 and retrieval.top_k")

    execution = config["execution"]
    variants = set(execution.get("variants") or [])
    allowed_variants = {
        "detector_only",
        "detector_retrieval",
        "query_only_vlm",
        "retrieval_grounded_vlm",
        "configured_policy",
    }
    if not variants or not variants <= allowed_variants:
        raise ValueError(
            f"execution.variants must be drawn from {sorted(allowed_variants)}"
        )
    if "configured_policy" in variants and "retrieval_grounded_vlm" not in variants:
        raise ValueError("configured_policy requires retrieval_grounded_vlm")
    rescue = config["decision_policy"].get("low_confidence_rescue") or {}
    if rescue.get("enabled", False):
        if "configured_policy" not in variants or "query_only_vlm" not in variants:
            raise ValueError(
                "low_confidence_rescue requires configured_policy and query_only_vlm"
            )
        lower = float(rescue["detector_min_confidence"])
        upper = float(rescue["detector_max_confidence"])
        if not 0.0 <= lower < upper <= 1.0:
            raise ValueError("low-confidence rescue detector range is invalid")
        for key in (
            "retrieval_min_cosine_similarity",
            "query_vlm_min_confidence",
            "grounded_vlm_min_confidence",
        ):
            if not 0.0 <= float(rescue[key]) <= 1.0:
                raise ValueError(f"decision_policy.low_confidence_rescue.{key} is invalid")
        if int(rescue["retrieval_min_supporting_neighbors"]) <= 0:
            raise ValueError("low-confidence rescue supporting neighbours must be positive")
    if any("vlm" in variant for variant in variants) and not vlm.get("enabled", True):
        raise ValueError("VLM variants require vlm.enabled=true")


def load_assistance_experiment_config(path: Path) -> dict[str, Any]:
    config = load_yaml_config(path)
    validate_assistance_experiment_config(config)
    return config


def resolve_experiment_path(value: str | Path, project_root: Path) -> Path:
    return resolve_path(value, project_root)

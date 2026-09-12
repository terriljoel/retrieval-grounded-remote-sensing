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

    retrieval = config["retrieval"]
    if not retrieval.get("database_path") and not retrieval.get("artifact_root"):
        raise ValueError("Set retrieval.database_path or retrieval.artifact_root")
    if not retrieval.get("evidence_splits"):
        raise ValueError("retrieval.evidence_splits must not be empty")
    top_k = int(retrieval.get("top_k", 0))
    if top_k <= 0:
        raise ValueError("retrieval.top_k must be positive")

    vlm = config["vlm"]
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
    if any("vlm" in variant for variant in variants) and not vlm.get("enabled", True):
        raise ValueError("VLM variants require vlm.enabled=true")
def load_assistance_experiment_config(path: Path) -> dict[str, Any]:
    config = load_yaml_config(path)
    validate_assistance_experiment_config(config)
    return config


def resolve_experiment_path(value: str | Path, project_root: Path) -> Path:
    return resolve_path(value, project_root)

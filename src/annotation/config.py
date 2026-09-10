from __future__ import annotations

from pathlib import Path
from typing import Any

from src.annotation.models import NWPU_CLASSES
from src.detection.config import load_yaml_config, resolve_path


REQUIRED_SECTIONS = {
    "app",
    "classes",
    "detector",
    "embedding",
    "retrieval",
    "vlm",
    "decision_policy",
    "annotations",
}


def validate_annotation_config(config: dict[str, Any]) -> None:
    missing = REQUIRED_SECTIONS - set(config)
    if missing:
        raise ValueError(f"Missing annotation configuration sections: {sorted(missing)}")

    classes = tuple(config["classes"])
    if classes != NWPU_CLASSES:
        raise ValueError(
            "The initial annotation prototype requires the fixed NWPU VHR-10 "
            "class order"
        )

    detector = config["detector"]
    if detector.get("backend") != "ultralytics":
        raise ValueError("The annotation prototype currently supports ultralytics")
    if not detector.get("checkpoint") and not detector.get("models"):
        raise ValueError("Set detector.checkpoint or detector.models")
    if detector.get("models") and not isinstance(detector["models"], dict):
        raise ValueError("detector.models must map model names to checkpoints")

    retrieval = config["retrieval"]
    if not retrieval.get("table"):
        raise ValueError("retrieval.table must not be empty")
    if not retrieval.get("artifact_root") and not retrieval.get("database_path"):
        raise ValueError(
            "Set retrieval.database_path or retrieval.artifact_root"
        )
    if not retrieval.get("evidence_splits"):
        raise ValueError("retrieval.evidence_splits must not be empty")
    if int(retrieval.get("top_k", 0)) <= 0:
        raise ValueError("retrieval.top_k must be positive")

    vlm = config["vlm"]
    allowed_modes = {"query_only", "retrieval_grounded", "compare"}
    if vlm.get("default_mode", "retrieval_grounded") not in allowed_modes:
        raise ValueError(f"vlm.default_mode must be one of {sorted(allowed_modes)}")
    max_evidence = int(vlm.get("max_evidence", retrieval["top_k"]))
    if not 1 <= max_evidence <= int(retrieval["top_k"]):
        raise ValueError("vlm.max_evidence must be between 1 and retrieval.top_k")

    policy = config["decision_policy"]
    for key in (
        "detector_min_confidence",
        "retrieval_min_cosine_similarity",
        "vlm_min_confidence",
    ):
        value = float(policy.get(key, -1.0))
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"decision_policy.{key} must be between 0 and 1")
    required_neighbors = int(policy.get("retrieval_min_supporting_neighbors", 0))
    if not 1 <= required_neighbors <= int(retrieval["top_k"]):
        raise ValueError(
            "decision_policy.retrieval_min_supporting_neighbors must be between "
            "1 and retrieval.top_k"
        )

    if float(config["embedding"].get("context_margin", -1.0)) < 0.0:
        raise ValueError("embedding.context_margin must be non-negative")
    if not config["annotations"].get("output_root"):
        raise ValueError("annotations.output_root must not be empty")


def load_annotation_config(path: Path) -> dict[str, Any]:
    config = load_yaml_config(path)
    validate_annotation_config(config)
    return config


def resolve_runtime_path(value: str | Path, project_root: Path) -> Path:
    return resolve_path(value, project_root)

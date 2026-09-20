from __future__ import annotations

from pathlib import Path
from typing import Any

from src.detection.config import load_yaml_config


REQUIRED_SECTIONS = {"run", "detector", "source", "inference", "outputs"}
SUPPORTED_GROUND_TRUTH_FORMATS = {"nwpu", "yolo"}


def validate_inference_config(config: dict[str, Any]) -> None:
    missing = REQUIRED_SECTIONS - set(config)
    if missing:
        raise ValueError(
            f"Missing inference configuration sections: {sorted(missing)}"
        )
    if not config["run"].get("name"):
        raise ValueError("run.name must not be empty")
    if config["detector"].get("backend") != "ultralytics":
        raise ValueError(
            "Unsupported inference backend. Currently supported: ultralytics"
        )
    if not config["detector"].get("checkpoint"):
        raise ValueError("detector.checkpoint must not be empty")

    source = config["source"]
    if not source.get("path"):
        raise ValueError("source.path must not be empty")
    if not source.get("dataset_id"):
        raise ValueError("source.dataset_id must not be empty")
    if source.get("manifest") and not source.get("dataset_root"):
        raise ValueError("source.dataset_root is required when source.manifest is set")
    if not isinstance(source.get("allow_test", False), bool):
        raise ValueError("source.allow_test must be true or false")

    for name in ("confidence", "iou"):
        value = config["inference"].get(name)
        if not isinstance(value, (int, float)) or not 0.0 <= value <= 1.0:
            raise ValueError(f"inference.{name} must be between 0 and 1")
    batch = config["inference"].get("batch")
    if not isinstance(batch, int) or isinstance(batch, bool) or batch <= 0:
        raise ValueError("inference.batch must be a positive integer")

    ground_truth = config.get("ground_truth")
    if ground_truth is not None:
        if not isinstance(ground_truth, dict):
            raise ValueError("ground_truth must be a mapping or null")
        if ground_truth.get("format") not in SUPPORTED_GROUND_TRUTH_FORMATS:
            raise ValueError(
                "ground_truth.format must be one of: "
                + ", ".join(sorted(SUPPORTED_GROUND_TRUTH_FORMATS))
            )
        if not ground_truth.get("path"):
            raise ValueError("ground_truth.path must not be empty")
        iou_threshold = ground_truth.get("iou_threshold", 0.5)
        if (
            not isinstance(iou_threshold, (int, float))
            or not 0.0 < iou_threshold <= 1.0
        ):
            raise ValueError("ground_truth.iou_threshold must be in (0, 1]")
    if not config["outputs"].get("root"):
        raise ValueError("outputs.root must not be empty")


def load_inference_config(path: Path) -> dict[str, Any]:
    config = load_yaml_config(path)
    validate_inference_config(config)
    return config

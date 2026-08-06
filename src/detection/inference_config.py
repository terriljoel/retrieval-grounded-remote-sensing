from __future__ import annotations

from pathlib import Path
from typing import Any

from src.detection.config import load_yaml_config


REQUIRED_SECTIONS = {"run", "detector", "inference", "outputs"}


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
    if not config["inference"].get("default_source_split"):
        raise ValueError("inference.default_source_split must not be empty")
    for name in ("confidence", "iou"):
        value = config["inference"].get(name)
        if not isinstance(value, (int, float)) or not 0.0 <= value <= 1.0:
            raise ValueError(f"inference.{name} must be between 0 and 1")
    if not config["outputs"].get("root"):
        raise ValueError("outputs.root must not be empty")


def load_inference_config(path: Path) -> dict[str, Any]:
    config = load_yaml_config(path)
    validate_inference_config(config)
    return config

from __future__ import annotations

import os
from pathlib import Path
import re
from typing import Any

import yaml


ENVIRONMENT_VARIABLE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
REQUIRED_SECTIONS = {
    "experiment", "data", "split", "export", "classes", "detector",
    "model", "training", "evaluation", "prediction", "outputs",
}


def _expand_environment(value: Any) -> Any:
    if isinstance(value, str):
        missing = sorted({
            name for name in ENVIRONMENT_VARIABLE.findall(value)
            if name not in os.environ
        })
        if missing:
            raise ValueError(
                "Missing environment variables in configuration: "
                + ", ".join(missing)
            )
        return ENVIRONMENT_VARIABLE.sub(
            lambda match: os.environ[match.group(1)], value
        )
    if isinstance(value, list):
        return [_expand_environment(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_environment(item) for key, item in value.items()}
    return value


def validate_detection_config(config: dict[str, Any]) -> None:
    missing = REQUIRED_SECTIONS - set(config)
    if missing:
        raise ValueError(f"Missing configuration sections: {sorted(missing)}")
    if config["detector"].get("backend") != "ultralytics":
        raise ValueError(
            "Unsupported detector backend. Currently supported: ultralytics"
        )
    if config["export"].get("format") != "yolo":
        raise ValueError("The ultralytics backend requires export.format=yolo")
    if config["data"].get("adapter") != "nwpu":
        raise ValueError("Unsupported data adapter. Currently supported: nwpu")
    if len(config["classes"]) != 10:
        raise ValueError("NWPU VHR-10 requires exactly 10 class names")

    ratios = [config["split"].get(name) for name in ("train", "val", "test")]
    if any(not isinstance(value, (int, float)) or value <= 0 for value in ratios):
        raise ValueError("Split ratios must be positive numbers")
    if abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError("Split ratios must sum to 1.0")
    if config["split"].get("strategy") != "multilabel":
        raise ValueError("Supported split strategy: multilabel")
    if config["evaluation"].get("split") not in {"train", "val", "test"}:
        raise ValueError("evaluation.split must be train, val, or test")


def load_detection_config(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Detection configuration not found: {path}")
    with path.open(encoding="utf-8") as file:
        loaded = yaml.safe_load(file)
    if not isinstance(loaded, dict):
        raise ValueError(f"Configuration must be a YAML mapping: {path}")

    config = _expand_environment(loaded)
    validate_detection_config(config)
    config["_config_path"] = str(path)
    return config


def resolve_path(value: str | Path, project_root: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def save_resolved_config(config: dict[str, Any], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {key: value for key, value in config.items() if not key.startswith("_")}
    with output_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(serializable, file, sort_keys=False)
    return output_path

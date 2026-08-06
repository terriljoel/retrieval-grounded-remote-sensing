from pathlib import Path

import pytest

from src.detection.inference_config import (
    load_inference_config,
    validate_inference_config,
)


CONFIG_PATH = (
    Path(__file__).resolve().parents[2]
    / "configs"
    / "inference"
    / "ultralytics_export.yaml"
)


def test_load_inference_config_is_independent_from_training(
    monkeypatch, tmp_path
):
    output_root = tmp_path / "experiments"
    monkeypatch.setenv("EXPERIMENT_OUTPUT_ROOT", str(output_root))

    config = load_inference_config(CONFIG_PATH)

    assert config["detector"]["backend"] == "ultralytics"
    assert config["detector"]["checkpoint"] == "/path/to/best.pt"
    assert config["source"]["dataset_id"] == "new_remote_sensing_dataset"
    assert config["source"]["split"] == "external"
    assert config["inference"]["confidence"] == 0.05
    assert config["ground_truth"] is None
    assert Path(config["outputs"]["root"]).resolve() == (
        output_root / "inference"
    ).resolve()
    assert "training" not in config
    assert "evaluation" not in config


def test_load_inference_config_requires_output_environment(monkeypatch):
    monkeypatch.delenv("EXPERIMENT_OUTPUT_ROOT", raising=False)

    with pytest.raises(ValueError, match="Missing environment variables"):
        load_inference_config(CONFIG_PATH)


def test_validate_inference_config_accepts_ground_truth_without_manifest():
    config = {
        "run": {"name": "evaluation"},
        "detector": {"backend": "ultralytics", "checkpoint": "best.pt"},
        "source": {
            "path": "images",
            "dataset_id": "new_dataset",
            "manifest": None,
        },
        "ground_truth": {
            "format": "yolo",
            "path": "labels",
            "iou_threshold": 0.5,
        },
        "inference": {"confidence": 0.05, "iou": 0.7},
        "outputs": {"root": "outputs"},
    }

    validate_inference_config(config)


def test_validate_inference_config_requires_dataset_root_for_manifest():
    config = {
        "run": {"name": "evaluation"},
        "detector": {"backend": "ultralytics", "checkpoint": "best.pt"},
        "source": {
            "path": "images",
            "dataset_id": "new_dataset",
            "manifest": "split.csv",
        },
        "inference": {"confidence": 0.05, "iou": 0.7},
        "outputs": {"root": "outputs"},
    }

    with pytest.raises(ValueError, match="source.dataset_root"):
        validate_inference_config(config)

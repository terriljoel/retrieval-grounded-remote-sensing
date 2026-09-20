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
    shared_root = tmp_path / "shared_resources"
    monkeypatch.setenv("SHARED_RESOURCES_ROOT", str(shared_root))

    config = load_inference_config(CONFIG_PATH)

    assert config["detector"]["backend"] == "ultralytics"
    assert Path(config["detector"]["checkpoint"]).resolve() == (
        shared_root / "checkpoints" / "yolov8s_pretrained_1024_seed42.pt"
    ).resolve()
    assert config["source"]["dataset_id"] == "nwpu_vhr10"
    assert config["source"]["split"] == "manifest"
    assert config["inference"]["confidence"] == 0.05
    assert config["ground_truth"]["format"] == "nwpu"
    assert Path(config["outputs"]["root"]).resolve() == (
        shared_root / "inference"
    ).resolve()
    assert "training" not in config
    assert "evaluation" not in config


def test_load_inference_config_requires_shared_resources_environment(monkeypatch):
    monkeypatch.delenv("SHARED_RESOURCES_ROOT", raising=False)

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
        "inference": {"batch": 8, "confidence": 0.05, "iou": 0.7},
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
        "inference": {"batch": 8, "confidence": 0.05, "iou": 0.7},
        "outputs": {"root": "outputs"},
    }

    with pytest.raises(ValueError, match="source.dataset_root"):
        validate_inference_config(config)


def test_validate_inference_config_rejects_non_positive_batch():
    config = {
        "run": {"name": "evaluation"},
        "detector": {"backend": "ultralytics", "checkpoint": "best.pt"},
        "source": {
            "path": "images",
            "dataset_id": "new_dataset",
            "manifest": None,
        },
        "ground_truth": None,
        "inference": {"batch": 0, "confidence": 0.05, "iou": 0.7},
        "outputs": {"root": "outputs"},
    }

    with pytest.raises(
        ValueError, match="inference.batch must be a positive integer"
    ):
        validate_inference_config(config)

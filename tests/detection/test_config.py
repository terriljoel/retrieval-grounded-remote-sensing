from pathlib import Path

import pytest

from src.detection.config import load_detection_config


CONFIG_PATH = (
    Path(__file__).resolve().parents[2]
    / "configs"
    / "detection"
    / "yolov8n_baseline.yaml"
)


def test_load_config_expands_runtime_paths(monkeypatch, tmp_path):
    variables = {
        "SHARED_RESOURCES_ROOT": tmp_path / "shared",
        "RAW_DATASET_ROOT": tmp_path / "raw",
        "PROCESSED_DATASET_ROOT": tmp_path / "processed",
        "MANIFEST_ROOT": tmp_path / "manifests",
        "EXPERIMENT_OUTPUT_ROOT": tmp_path / "experiments",
    }
    for name, value in variables.items():
        monkeypatch.setenv(name, str(value))

    config = load_detection_config(CONFIG_PATH)

    assert config["model"]["weights"] == "yolov8n.pt"
    assert config["data"]["raw_root"] == str(variables["RAW_DATASET_ROOT"])
    assert config["evaluation"]["split"] == "test"


def test_load_config_reports_missing_environment(monkeypatch):
    for name in (
        "SHARED_RESOURCES_ROOT",
        "RAW_DATASET_ROOT",
        "PROCESSED_DATASET_ROOT",
        "MANIFEST_ROOT",
        "EXPERIMENT_OUTPUT_ROOT",
    ):
        monkeypatch.delenv(name, raising=False)

    with pytest.raises(ValueError, match="Missing environment variables"):
        load_detection_config(CONFIG_PATH)

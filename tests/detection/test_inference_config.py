from pathlib import Path

import pytest

from src.detection.inference_config import load_inference_config


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
    assert config["inference"]["default_source_split"] == "external"
    assert config["inference"]["confidence"] == 0.05
    assert Path(config["outputs"]["root"]).resolve() == (
        output_root / "inference"
    ).resolve()
    assert "training" not in config
    assert "evaluation" not in config


def test_load_inference_config_requires_output_environment(monkeypatch):
    monkeypatch.delenv("EXPERIMENT_OUTPUT_ROOT", raising=False)

    with pytest.raises(ValueError, match="Missing environment variables"):
        load_inference_config(CONFIG_PATH)

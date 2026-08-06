from pathlib import Path
import sys

import pytest

from scripts import export_predictions


CONFIG_PATH = (
    Path(__file__).resolve().parents[2]
    / "configs"
    / "inference"
    / "ultralytics_export.yaml"
)


def test_export_predictions_requires_test_confirmation(
    monkeypatch, tmp_path, capsys
):
    for name in (
        "SHARED_RESOURCES_ROOT",
        "RAW_DATASET_ROOT",
        "PROCESSED_DATASET_ROOT",
        "MANIFEST_ROOT",
        "EXPERIMENT_OUTPUT_ROOT",
    ):
        monkeypatch.setenv(name, str(tmp_path / name.lower()))
    monkeypatch.setattr(sys, "argv", [
        "rs-export-predictions",
        "--config", str(CONFIG_PATH),
        "--checkpoint", str(tmp_path / "best.pt"),
        "--source", str(tmp_path / "images"),
        "--dataset-id", "nwpu_vhr10",
        "--source-split", "test",
    ])

    with pytest.raises(SystemExit, match="2"):
        export_predictions.main()

    assert "requires --confirm-test" in capsys.readouterr().err

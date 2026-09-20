from pathlib import Path
import sys

from scripts import export_predictions


CONFIG_PATH = (
    Path(__file__).resolve().parents[2]
    / "configs"
    / "inference"
    / "ultralytics_export.yaml"
)


def test_export_predictions_cli_uses_only_config(monkeypatch, tmp_path):
    shared_root = tmp_path / "shared_resources"
    monkeypatch.setenv("SHARED_RESOURCES_ROOT", str(shared_root))
    monkeypatch.setenv("JOB_LOG_ROOT", str(tmp_path / "jobs"))
    captured = {}

    def fake_export(config, project_root):
        captured["config"] = config
        captured["project_root"] = project_root
        return tmp_path / "export"

    monkeypatch.setattr(export_predictions, "export_predictions", fake_export)
    monkeypatch.setattr(sys, "argv", [
        "rs-export-predictions",
        "--config", str(CONFIG_PATH),
    ])

    export_predictions.main()

    assert captured["config"]["source"]["dataset_id"] == "nwpu_vhr10"
    assert Path(captured["config"]["detector"]["checkpoint"]).resolve() == (
        shared_root / "checkpoints" / "yolov8s_pretrained_1024_seed42.pt"
    ).resolve()

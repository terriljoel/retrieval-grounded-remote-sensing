import csv
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.detection import ultralytics_backend


class FakeTensor:
    def __init__(self, values):
        self.values = values

    def cpu(self):
        return self

    def tolist(self):
        return self.values


class FakeModel:
    def __init__(self, synthetic_result_paths: bool = False):
        self.predictor = None
        self.synthetic_result_paths = synthetic_result_paths

    def predict(self, **arguments):
        run_directory = Path(arguments["project"]) / arguments["name"]
        run_directory.mkdir(parents=True)
        self.predictor = SimpleNamespace(save_dir=run_directory)
        results = []
        for index, source in enumerate(arguments["source"]):
            has_detection = index == 0
            boxes = SimpleNamespace(
                xyxy=FakeTensor([[1.0, 2.0, 20.0, 30.0]] if has_detection else []),
                cls=FakeTensor([0.0] if has_detection else []),
                conf=FakeTensor([0.9] if has_detection else []),
            )
            results.append(SimpleNamespace(
                path=(
                    f"image{index}.jpg"
                    if self.synthetic_result_paths
                    else source
                ),
                orig_shape=(80, 100),
                boxes=boxes,
                names={0: "airplane"},
            ))
        return results


def test_export_predictions_accepts_external_directory(
    monkeypatch, tmp_path
):
    source = tmp_path / "new_dataset"
    first = source / "region_a" / "tile.jpg"
    second = source / "region_b" / "tile.png"
    first.parent.mkdir(parents=True)
    second.parent.mkdir(parents=True)
    first.touch()
    second.touch()
    checkpoint = tmp_path / "best.pt"
    checkpoint.touch()
    output_root = tmp_path / "inference"
    config = {
        "run": {"name": "inference"},
        "detector": {
            "backend": "ultralytics",
            "checkpoint": str(checkpoint),
        },
        "source": {
            "path": str(source),
            "dataset_id": "external tiles",
            "split": "external",
            "manifest": None,
            "dataset_root": None,
            "allow_test": False,
        },
        "ground_truth": None,
        "inference": {
            "imgsz": 640,
            "batch": 16,
            "confidence": 0.05,
            "iou": 0.7,
            "device": 0,
            "save_rendered_images": True,
            "save_yolo_labels": True,
            "save_confidence": True,
            "verbose": False,
            "exist_ok": False,
        },
        "outputs": {"root": str(output_root)},
    }
    monkeypatch.setattr(
        ultralytics_backend,
        "_load_yolo",
        lambda _: FakeModel(synthetic_result_paths=True),
    )
    monkeypatch.setattr(
        ultralytics_backend, "collect_runtime_metadata", lambda _: {}
    )

    run_directory = ultralytics_backend.export_predictions(
        config=config,
        project_root=tmp_path,
    )

    with (run_directory / "inference_images.csv").open(encoding="utf-8") as file:
        image_rows = list(csv.DictReader(file))
    with (run_directory / "detections.csv").open(encoding="utf-8") as file:
        detection_rows = list(csv.DictReader(file))
    assert len(image_rows) == 2
    assert {row["dataset_id"] for row in image_rows} == {"external tiles"}
    assert [row["prediction_count"] for row in image_rows] == ["1", "0"]
    assert len({row["image_id"] for row in image_rows}) == 2
    assert {row["source_image_path"] for row in image_rows} == {
        str(first.resolve()),
        str(second.resolve()),
    }
    assert len(detection_rows) == 1
    assert detection_rows[0]["predicted_class_name"] == "airplane"


def test_export_predictions_compares_nwpu_ground_truth_without_manifest(
    monkeypatch, tmp_path
):
    source = tmp_path / "images"
    labels = tmp_path / "annotations"
    source.mkdir()
    labels.mkdir()
    positive = source / "positive.jpg"
    positive.touch()
    (labels / "positive.txt").write_text(
        "(1,2),(20,30),1\n", encoding="utf-8"
    )
    checkpoint = tmp_path / "best.pt"
    checkpoint.touch()
    config = {
        "run": {"name": "inference"},
        "detector": {
            "backend": "ultralytics",
            "checkpoint": str(checkpoint),
        },
        "source": {
            "path": str(source),
            "dataset_id": "nwpu",
            "split": "external",
            "manifest": None,
            "dataset_root": None,
            "allow_test": False,
        },
        "ground_truth": {
            "format": "nwpu",
            "path": str(labels),
            "iou_threshold": 0.5,
        },
        "inference": {
            "imgsz": 640,
            "batch": 16,
            "confidence": 0.05,
            "iou": 0.7,
            "device": 0,
            "save_rendered_images": False,
            "save_yolo_labels": False,
            "save_confidence": False,
            "verbose": False,
            "exist_ok": False,
        },
        "outputs": {"root": str(tmp_path / "inference")},
    }
    monkeypatch.setattr(ultralytics_backend, "_load_yolo", lambda _: FakeModel())
    monkeypatch.setattr(
        ultralytics_backend, "collect_runtime_metadata", lambda _: {}
    )

    run_directory = ultralytics_backend.export_predictions(
        config=config,
        project_root=tmp_path,
    )

    with (run_directory / "inference_images.csv").open(encoding="utf-8") as file:
        image_rows = list(csv.DictReader(file))
    with (run_directory / "prediction_comparison.csv").open(
        encoding="utf-8"
    ) as file:
        comparison_rows = list(csv.DictReader(file))
    assert image_rows[0]["source_split"] == "external"
    assert image_rows[0]["ground_truth_count"] == "1"
    assert (run_directory / "ground_truth.csv").is_file()
    assert comparison_rows[0]["status"] == "true_positive"


def test_export_predictions_requires_config_authorization_for_test(tmp_path):
    source = tmp_path / "images"
    source.mkdir()
    (source / "tile.jpg").touch()
    checkpoint = tmp_path / "best.pt"
    checkpoint.touch()
    config = {
        "run": {"name": "inference"},
        "detector": {
            "backend": "ultralytics",
            "checkpoint": str(checkpoint),
        },
        "source": {
            "path": str(source),
            "dataset_id": "held_out",
            "split": "test",
            "manifest": None,
            "dataset_root": None,
            "allow_test": False,
        },
        "ground_truth": None,
        "inference": {},
        "outputs": {"root": str(tmp_path / "outputs")},
    }

    with pytest.raises(ValueError, match="source.allow_test=true"):
        ultralytics_backend.export_predictions(config, tmp_path)

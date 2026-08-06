import csv
from pathlib import Path
from types import SimpleNamespace

from src.detection import ultralytics_backend


class FakeTensor:
    def __init__(self, values):
        self.values = values

    def cpu(self):
        return self

    def tolist(self):
        return self.values


class FakeModel:
    def __init__(self):
        self.predictor = None

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
                path=source,
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
        "detector": {"backend": "ultralytics"},
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
    monkeypatch.setattr(ultralytics_backend, "_load_yolo", lambda _: FakeModel())
    monkeypatch.setattr(
        ultralytics_backend, "collect_runtime_metadata", lambda _: {}
    )

    run_directory = ultralytics_backend.export_predictions(
        config=config,
        project_root=tmp_path,
        checkpoint=checkpoint,
        source=source,
        dataset_id="external tiles",
        source_split="external",
    )

    with (run_directory / "inference_images.csv").open(encoding="utf-8") as file:
        image_rows = list(csv.DictReader(file))
    with (run_directory / "detections.csv").open(encoding="utf-8") as file:
        detection_rows = list(csv.DictReader(file))
    assert len(image_rows) == 2
    assert {row["dataset_id"] for row in image_rows} == {"external tiles"}
    assert [row["prediction_count"] for row in image_rows] == ["1", "0"]
    assert len({row["image_id"] for row in image_rows}) == 2
    assert len(detection_rows) == 1
    assert detection_rows[0]["predicted_class_name"] == "airplane"

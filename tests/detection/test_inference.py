import csv

import pytest

from src.detection.inference import (
    DetectionRecord,
    InferenceImageRecord,
    discover_inference_images,
    validate_inference_records,
    write_inference_tables,
)


def _image(image_id: str, prediction_count: int) -> InferenceImageRecord:
    return InferenceImageRecord(
        image_id=image_id,
        dataset_id="nwpu_vhr10",
        source_image_path=f"/dataset/{image_id}.jpg",
        source_split="val",
        image_width=100,
        image_height=80,
        prediction_count=prediction_count,
    )


def test_write_inference_tables_preserves_images_without_detections(tmp_path):
    images = [_image("positive_001", 1), _image("background_001", 0)]
    detections = [DetectionRecord(
        detection_id="positive_001_det_0000",
        image_id="positive_001",
        predicted_class_id=0,
        predicted_class_name="airplane",
        confidence=0.91,
        xmin=10.0,
        ymin=12.0,
        xmax=40.0,
        ymax=50.0,
    )]

    summary = write_inference_tables(tmp_path, images, detections)

    assert summary == {
        "images": 2,
        "detections": 1,
        "images_without_detections": 1,
    }
    with (tmp_path / "inference_images.csv").open(encoding="utf-8") as file:
        image_rows = list(csv.DictReader(file))
    with (tmp_path / "detections.csv").open(encoding="utf-8") as file:
        detection_rows = list(csv.DictReader(file))
    assert [row["image_id"] for row in image_rows] == [
        "positive_001", "background_001"
    ]
    assert detection_rows[0]["predicted_class_name"] == "airplane"


def test_validate_inference_records_rejects_count_mismatch():
    with pytest.raises(ValueError, match="Prediction count mismatch"):
        validate_inference_records([_image("positive_001", 1)], [])


def test_validate_inference_records_rejects_unknown_image():
    detection = DetectionRecord(
        detection_id="missing_det_0000",
        image_id="missing",
        predicted_class_id=0,
        predicted_class_name="airplane",
        confidence=0.5,
        xmin=1.0,
        ymin=1.0,
        xmax=2.0,
        ymax=2.0,
    )
    with pytest.raises(ValueError, match="unknown image"):
        validate_inference_records([], [detection])


def test_discover_inference_images_supports_nested_external_dataset(tmp_path):
    first = tmp_path / "images" / "area_a" / "tile.jpg"
    second = tmp_path / "images" / "area_b" / "tile.png"
    ignored = tmp_path / "images" / "notes.txt"
    first.parent.mkdir(parents=True)
    second.parent.mkdir(parents=True)
    first.touch()
    second.touch()
    ignored.write_text("not an image", encoding="utf-8")

    discovered = discover_inference_images(tmp_path / "images")

    assert discovered == sorted([first.resolve(), second.resolve()])

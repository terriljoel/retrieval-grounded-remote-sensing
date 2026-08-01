from pathlib import Path

from PIL import Image
import pytest

from src.data.models import BoundingBox, DatasetItem
from src.data.yolo import (
    convert_box_to_yolo,
    export_yolo_dataset,
    validate_yolo_dataset,
)


CLASS_NAMES = [
    "airplane", "ship", "storage_tank", "baseball_diamond",
    "tennis_court", "basketball_court", "ground_track_field",
    "harbor", "bridge", "vehicle",
]


def _image(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (10, 10)).save(path)
    return path


def test_box_conversion_maps_class_and_normalizes():
    converted = convert_box_to_yolo(BoundingBox(2, 2, 6, 8, 10), 10, 10)
    assert converted == pytest.approx((9, 0.4, 0.5, 0.4, 0.6))


def test_export_avoids_positive_background_filename_collisions(tmp_path):
    splits = {"train": [], "val": [], "test": []}
    for split in splits:
        positive = _image(tmp_path / "source" / split / "positive" / "001.jpg")
        annotation = tmp_path / "source" / split / "annotations" / "001.txt"
        annotation.parent.mkdir(parents=True, exist_ok=True)
        annotation.write_text("(1,1),(5,5),1\n", encoding="utf-8")
        background = _image(tmp_path / "source" / split / "background" / "001.jpg")
        splits[split] = [
            DatasetItem(positive, annotation, False),
            DatasetItem(background, None, True),
        ]

    output_root = tmp_path / "yolo"
    export_yolo_dataset(splits, output_root, CLASS_NAMES)
    report = validate_yolo_dataset(
        output_root,
        expected_images=6,
        expected_objects=3,
        expected_backgrounds=3,
    )

    assert report["valid"] is True
    assert len(list((output_root / "images" / "train").glob("*001.jpg"))) == 2
    assert (output_root / "export_manifest.csv").is_file()

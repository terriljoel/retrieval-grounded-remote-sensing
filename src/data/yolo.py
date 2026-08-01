from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path
from shutil import copy2
from typing import Any

from PIL import Image
import yaml

from src.data.models import BoundingBox, DatasetItem
from src.data.parsers.nwpu import parse_nwpu_annotation
from src.data.splitting import DatasetSplits, validate_dataset_splits


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def convert_box_to_yolo(
    box: BoundingBox,
    image_width: int,
    image_height: int,
) -> tuple[int, float, float, float, float]:
    if image_width <= 0 or image_height <= 0:
        raise ValueError("Image dimensions must be positive")
    if not (
        0 <= box.xmin < box.xmax <= image_width
        and 0 <= box.ymin < box.ymax <= image_height
    ):
        raise ValueError(f"Invalid box {box} for {image_width}x{image_height}")
    if box.class_id not in range(1, 11):
        raise ValueError(f"NWPU class ID must be 1-10, got {box.class_id}")
    return (
        box.class_id - 1,
        ((box.xmin + box.xmax) / 2) / image_width,
        ((box.ymin + box.ymax) / 2) / image_height,
        (box.xmax - box.xmin) / image_width,
        (box.ymax - box.ymin) / image_height,
    )


def format_yolo_box(box: BoundingBox, image_width: int, image_height: int) -> str:
    class_id, x_center, y_center, width, height = convert_box_to_yolo(
        box, image_width, image_height
    )
    return (
        f"{class_id} {x_center:.8f} {y_center:.8f} "
        f"{width:.8f} {height:.8f}"
    )


def _output_stem(split: str, index: int, item: DatasetItem) -> str:
    source_type = "background" if item.is_background else "positive"
    return f"{split}_{index:04d}_{source_type}_{item.image_path.stem}"


def export_yolo_dataset(
    splits: DatasetSplits,
    output_root: Path,
    class_names: list[str],
) -> dict[str, Any]:
    """Export fixed splits with collision-safe names and source provenance."""
    validate_dataset_splits(splits)
    if len(class_names) != 10:
        raise ValueError("Expected 10 class names")

    output_root = output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"YOLO output directory is not empty: {output_root}")

    export_records: list[dict[str, Any]] = []
    split_summary: dict[str, dict[str, Any]] = {}
    for split in ("train", "val", "test"):
        image_dir = output_root / "images" / split
        label_dir = output_root / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        object_counts: Counter[int] = Counter()
        background_count = 0

        for index, item in enumerate(splits[split]):
            output_stem = _output_stem(split, index, item)
            output_image = image_dir / f"{output_stem}{item.image_path.suffix.lower()}"
            output_label = label_dir / f"{output_stem}.txt"
            if output_image.exists() or output_label.exists():
                raise FileExistsError(f"Duplicate YOLO output ID: {output_stem}")
            copy2(item.image_path, output_image)

            label_lines: list[str] = []
            if item.is_background:
                background_count += 1
            else:
                if item.annotation_path is None:
                    raise ValueError(f"Positive image has no annotation: {item.image_path}")
                with Image.open(item.image_path) as image:
                    image_width, image_height = image.size
                boxes = parse_nwpu_annotation(item.annotation_path)
                label_lines = [
                    format_yolo_box(box, image_width, image_height) for box in boxes
                ]
                object_counts.update(box.class_id - 1 for box in boxes)

            output_label.write_text(
                "\n".join(label_lines) + ("\n" if label_lines else ""),
                encoding="utf-8",
            )
            export_records.append({
                "output_id": output_stem,
                "split": split,
                "source_image": str(item.image_path.resolve()),
                "source_annotation": (
                    str(item.annotation_path.resolve())
                    if item.annotation_path is not None else ""
                ),
                "is_background": item.is_background,
                "object_count": len(label_lines),
            })

        split_summary[split] = {
            "images": len(splits[split]),
            "backgrounds": background_count,
            "objects": sum(object_counts.values()),
            "objects_by_class": dict(sorted(object_counts.items())),
        }

    dataset_yaml = output_root / "dataset.yaml"
    with dataset_yaml.open("w", encoding="utf-8") as file:
        yaml.safe_dump({
            "path": output_root.as_posix(),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "names": {index: name for index, name in enumerate(class_names)},
        }, file, sort_keys=False)

    export_manifest = output_root / "export_manifest.csv"
    with export_manifest.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(export_records[0]))
        writer.writeheader()
        writer.writerows(export_records)

    summary = {
        "dataset_yaml": str(dataset_yaml),
        "export_manifest": str(export_manifest),
        "splits": split_summary,
    }
    (output_root / "export_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def validate_yolo_dataset(
    output_root: Path,
    expected_images: int | None = None,
    expected_objects: int | None = None,
    expected_backgrounds: int | None = None,
) -> dict[str, Any]:
    issues: list[str] = []
    totals = Counter()
    class_counts: Counter[int] = Counter()
    tolerance = 1e-6

    for split in ("train", "val", "test"):
        image_dir = output_root / "images" / split
        label_dir = output_root / "labels" / split
        images = [
            path for path in image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ]
        labels = list(label_dir.glob("*.txt"))
        image_stems = {path.stem for path in images}
        label_stems = {path.stem for path in labels}
        if image_stems != label_stems:
            issues.append(f"{split}: image/label stem mismatch")
        totals["images"] += len(images)
        totals["labels"] += len(labels)

        for label_path in labels:
            lines = [
                line.strip()
                for line in label_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if not lines:
                totals["backgrounds"] += 1
            for line_number, line in enumerate(lines, start=1):
                fields = line.split()
                location = f"{label_path}:{line_number}"
                if len(fields) != 5:
                    issues.append(f"{location}: expected 5 fields")
                    continue
                try:
                    class_id = int(fields[0])
                    x_center, y_center, width, height = map(float, fields[1:])
                except ValueError:
                    issues.append(f"{location}: non-numeric value")
                    continue
                values = (x_center, y_center, width, height)
                if not all(math.isfinite(value) for value in values):
                    issues.append(f"{location}: non-finite value")
                    continue
                if class_id not in range(10):
                    issues.append(f"{location}: invalid class {class_id}")
                if not (
                    0 <= x_center <= 1 and 0 <= y_center <= 1
                    and 0 < width <= 1 and 0 < height <= 1
                ):
                    issues.append(f"{location}: normalized value outside range")
                if not (
                    x_center - width / 2 >= -tolerance
                    and y_center - height / 2 >= -tolerance
                    and x_center + width / 2 <= 1 + tolerance
                    and y_center + height / 2 <= 1 + tolerance
                ):
                    issues.append(f"{location}: box outside image")
                totals["objects"] += 1
                class_counts[class_id] += 1

    expected = {
        "images": expected_images,
        "objects": expected_objects,
        "backgrounds": expected_backgrounds,
    }
    for key, value in expected.items():
        if value is not None and totals[key] != value:
            issues.append(f"Expected {value} {key}, found {totals[key]}")
    if totals["images"] != totals["labels"]:
        issues.append("Total image and label counts differ")

    report = {
        **dict(totals),
        "class_counts": dict(sorted(class_counts.items())),
        "issues": issues,
        "valid": not issues,
    }
    if issues:
        raise ValueError("YOLO export validation failed: " + "; ".join(issues[:20]))
    return report

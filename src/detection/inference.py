from __future__ import annotations

import csv
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class InferenceImageRecord:
    image_id: str
    dataset_id: str
    source_image_path: str
    source_split: str
    image_width: int
    image_height: int
    prediction_count: int


@dataclass(frozen=True)
class DetectionRecord:
    detection_id: str
    image_id: str
    predicted_class_id: int
    predicted_class_name: str
    confidence: float
    xmin: float
    ymin: float
    xmax: float
    ymax: float


def validate_inference_records(
    images: Iterable[InferenceImageRecord],
    detections: Iterable[DetectionRecord],
) -> dict[str, int]:
    image_records = list(images)
    detection_records = list(detections)
    images_by_id = {record.image_id: record for record in image_records}
    if len(images_by_id) != len(image_records):
        raise ValueError("Inference image IDs must be unique")

    detection_ids = {record.detection_id for record in detection_records}
    if len(detection_ids) != len(detection_records):
        raise ValueError("Detection IDs must be unique")

    counts_by_image: Counter[str] = Counter()
    for image in image_records:
        if not image.dataset_id:
            raise ValueError(f"Missing dataset ID for {image.image_id}")
        if not image.source_split:
            raise ValueError(f"Missing source split for {image.image_id}")
        if image.image_width <= 0 or image.image_height <= 0:
            raise ValueError(f"Invalid dimensions for {image.image_id}")
        if image.prediction_count < 0:
            raise ValueError(f"Invalid prediction count for {image.image_id}")

    for detection in detection_records:
        image = images_by_id.get(detection.image_id)
        if image is None:
            raise ValueError(
                f"Detection references unknown image: {detection.image_id}"
            )
        if detection.predicted_class_id < 0:
            raise ValueError(f"Invalid class ID for {detection.detection_id}")
        if not detection.predicted_class_name:
            raise ValueError(f"Missing class name for {detection.detection_id}")
        if not 0.0 <= detection.confidence <= 1.0:
            raise ValueError(f"Invalid confidence for {detection.detection_id}")
        if not (
            0.0 <= detection.xmin < detection.xmax <= image.image_width
            and 0.0 <= detection.ymin < detection.ymax <= image.image_height
        ):
            raise ValueError(f"Invalid box for {detection.detection_id}")
        counts_by_image[detection.image_id] += 1

    for image in image_records:
        actual = counts_by_image[image.image_id]
        if image.prediction_count != actual:
            raise ValueError(
                f"Prediction count mismatch for {image.image_id}: "
                f"expected {image.prediction_count}, found {actual}"
            )

    return {
        "images": len(image_records),
        "detections": len(detection_records),
        "images_without_detections": sum(
            record.prediction_count == 0 for record in image_records
        ),
    }


def discover_inference_images(source: Path) -> list[Path]:
    """Return a stable image list for one image or a directory tree."""
    source = source.resolve()
    if source.is_file():
        if source.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported inference image type: {source}")
        return [source]
    if not source.is_dir():
        raise FileNotFoundError(f"Inference source not found: {source}")

    images = sorted(
        path.resolve()
        for path in source.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        raise ValueError(f"No supported images found under: {source}")
    return images


def write_inference_tables(
    output_directory: Path,
    images: Iterable[InferenceImageRecord],
    detections: Iterable[DetectionRecord],
) -> dict[str, int]:
    image_records = list(images)
    detection_records = list(detections)
    summary = validate_inference_records(image_records, detection_records)
    output_directory.mkdir(parents=True, exist_ok=True)

    image_path = output_directory / "inference_images.csv"
    with image_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file, fieldnames=list(InferenceImageRecord.__dataclass_fields__)
        )
        writer.writeheader()
        writer.writerows(asdict(record) for record in image_records)

    detection_path = output_directory / "detections.csv"
    with detection_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file, fieldnames=list(DetectionRecord.__dataclass_fields__)
        )
        writer.writeheader()
        writer.writerows(asdict(record) for record in detection_records)

    return summary

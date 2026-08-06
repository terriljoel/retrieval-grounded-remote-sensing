from __future__ import annotations

import csv
import math
from collections import Counter, defaultdict
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
    ground_truth_count: int | None = None


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


@dataclass(frozen=True)
class GroundTruthRecord:
    ground_truth_id: str
    image_id: str
    class_id: int
    class_name: str
    xmin: float
    ymin: float
    xmax: float
    ymax: float


@dataclass(frozen=True)
class ComparisonRecord:
    comparison_id: str
    image_id: str
    detection_id: str
    ground_truth_id: str
    iou: float | None
    status: str


def parse_yolo_ground_truth(
    annotation_path: Path,
    image_width: int,
    image_height: int,
) -> list[tuple[int, float, float, float, float]]:
    """Read normalized YOLO labels and return pixel-space xyxy boxes."""
    if image_width <= 0 or image_height <= 0:
        raise ValueError("Image dimensions must be positive")

    boxes: list[tuple[int, float, float, float, float]] = []
    lines = annotation_path.read_text(encoding="utf-8").splitlines()
    tolerance = 1e-6
    for line_number, line in enumerate(lines, start=1):
        line = line.strip()
        if not line:
            continue
        fields = line.split()
        location = f"{annotation_path}:{line_number}"
        if len(fields) != 5:
            raise ValueError(f"{location}: expected 5 YOLO fields")
        try:
            class_id = int(fields[0])
            x_center, y_center, width, height = map(float, fields[1:])
        except ValueError as error:
            raise ValueError(f"{location}: non-numeric YOLO value") from error
        values = (x_center, y_center, width, height)
        if class_id < 0:
            raise ValueError(f"{location}: class ID must be non-negative")
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"{location}: non-finite YOLO value")
        if not (
            0.0 <= x_center <= 1.0
            and 0.0 <= y_center <= 1.0
            and 0.0 < width <= 1.0
            and 0.0 < height <= 1.0
        ):
            raise ValueError(f"{location}: normalized value outside [0, 1]")

        xmin_normalized = x_center - width / 2
        ymin_normalized = y_center - height / 2
        xmax_normalized = x_center + width / 2
        ymax_normalized = y_center + height / 2
        if not (
            xmin_normalized >= -tolerance
            and ymin_normalized >= -tolerance
            and xmax_normalized <= 1.0 + tolerance
            and ymax_normalized <= 1.0 + tolerance
        ):
            raise ValueError(f"{location}: box extends outside the image")
        boxes.append((
            class_id,
            max(0.0, xmin_normalized) * image_width,
            max(0.0, ymin_normalized) * image_height,
            min(1.0, xmax_normalized) * image_width,
            min(1.0, ymax_normalized) * image_height,
        ))
    return boxes


def _box_iou(detection: DetectionRecord, ground_truth: GroundTruthRecord) -> float:
    intersection_width = max(
        0.0, min(detection.xmax, ground_truth.xmax) - max(detection.xmin, ground_truth.xmin)
    )
    intersection_height = max(
        0.0, min(detection.ymax, ground_truth.ymax) - max(detection.ymin, ground_truth.ymin)
    )
    intersection = intersection_width * intersection_height
    detection_area = (detection.xmax - detection.xmin) * (
        detection.ymax - detection.ymin
    )
    ground_truth_area = (ground_truth.xmax - ground_truth.xmin) * (
        ground_truth.ymax - ground_truth.ymin
    )
    union = detection_area + ground_truth_area - intersection
    return intersection / union if union > 0 else 0.0


def compare_predictions_to_ground_truth(
    detections: Iterable[DetectionRecord],
    ground_truths: Iterable[GroundTruthRecord],
    iou_threshold: float = 0.5,
) -> list[ComparisonRecord]:
    """Greedily create one-to-one, class-aware diagnostic comparisons."""
    if not 0.0 < iou_threshold <= 1.0:
        raise ValueError("IoU threshold must be in (0, 1]")
    detections_by_image: dict[str, list[DetectionRecord]] = defaultdict(list)
    ground_truths_by_image: dict[str, list[GroundTruthRecord]] = defaultdict(list)
    for detection in detections:
        detections_by_image[detection.image_id].append(detection)
    for ground_truth in ground_truths:
        ground_truths_by_image[ground_truth.image_id].append(ground_truth)

    comparisons: list[ComparisonRecord] = []
    for image_id in sorted(set(detections_by_image) | set(ground_truths_by_image)):
        image_detections = detections_by_image[image_id]
        image_ground_truths = ground_truths_by_image[image_id]
        candidates = sorted(
            (
                (_box_iou(detection, ground_truth), detection, ground_truth)
                for detection in image_detections
                for ground_truth in image_ground_truths
            ),
            key=lambda item: (-item[0], item[1].detection_id, item[2].ground_truth_id),
        )
        matched_detections: set[str] = set()
        matched_ground_truths: set[str] = set()
        image_comparisons: list[ComparisonRecord] = []
        for iou, detection, ground_truth in candidates:
            if iou < iou_threshold:
                break
            if (
                detection.detection_id in matched_detections
                or ground_truth.ground_truth_id in matched_ground_truths
            ):
                continue
            matched_detections.add(detection.detection_id)
            matched_ground_truths.add(ground_truth.ground_truth_id)
            status = (
                "true_positive"
                if detection.predicted_class_id == ground_truth.class_id
                else "class_error"
            )
            image_comparisons.append(ComparisonRecord(
                comparison_id="",
                image_id=image_id,
                detection_id=detection.detection_id,
                ground_truth_id=ground_truth.ground_truth_id,
                iou=iou,
                status=status,
            ))
        for detection in image_detections:
            if detection.detection_id not in matched_detections:
                image_comparisons.append(ComparisonRecord(
                    comparison_id="",
                    image_id=image_id,
                    detection_id=detection.detection_id,
                    ground_truth_id="",
                    iou=None,
                    status="false_positive",
                ))
        for ground_truth in image_ground_truths:
            if ground_truth.ground_truth_id not in matched_ground_truths:
                image_comparisons.append(ComparisonRecord(
                    comparison_id="",
                    image_id=image_id,
                    detection_id="",
                    ground_truth_id=ground_truth.ground_truth_id,
                    iou=None,
                    status="false_negative",
                ))
        for index, comparison in enumerate(image_comparisons):
            comparisons.append(ComparisonRecord(
                comparison_id=f"{image_id}_cmp_{index:04d}",
                image_id=comparison.image_id,
                detection_id=comparison.detection_id,
                ground_truth_id=comparison.ground_truth_id,
                iou=comparison.iou,
                status=comparison.status,
            ))
    return comparisons


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
    ground_truths: Iterable[GroundTruthRecord] | None = None,
    comparisons: Iterable[ComparisonRecord] | None = None,
) -> dict[str, int]:
    image_records = list(images)
    detection_records = list(detections)
    ground_truth_records = list(ground_truths) if ground_truths is not None else None
    comparison_records = list(comparisons) if comparisons is not None else None
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

    if ground_truth_records is not None:
        ground_truth_path = output_directory / "ground_truth.csv"
        with ground_truth_path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(
                file, fieldnames=list(GroundTruthRecord.__dataclass_fields__)
            )
            writer.writeheader()
            writer.writerows(asdict(record) for record in ground_truth_records)
        summary["ground_truth_objects"] = len(ground_truth_records)

    if comparison_records is not None:
        comparison_path = output_directory / "prediction_comparison.csv"
        with comparison_path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(
                file, fieldnames=list(ComparisonRecord.__dataclass_fields__)
            )
            writer.writeheader()
            writer.writerows(asdict(record) for record in comparison_records)
        status_counts = Counter(record.status for record in comparison_records)
        for status in (
            "true_positive", "class_error", "false_positive", "false_negative"
        ):
            summary[status] = status_counts[status]

    return summary

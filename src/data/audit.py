from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import asdict
import hashlib
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError

from src.data.models import BoundingBox, DatasetItem


Parser = Callable[[Path], list[BoundingBox]]


def calculate_file_hash(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def audit_dataset(
    items: list[DatasetItem],
    parser: Parser,
    valid_class_ids: set[int],
) -> dict[str, Any]:
    issues = []
    box_records = []
    dimension_counts = Counter()
    mode_counts = Counter()
    class_counts = Counter()
    objects_per_image = Counter()
    hashes = defaultdict(list)

    positive_count = sum(
        not item.is_background for item in items
    )
    background_count = sum(
        item.is_background for item in items
    )

    for item in items:
        image_path = item.image_path

        try:
            with Image.open(image_path) as image:
                image.verify()

            with Image.open(image_path) as image:
                width, height = image.size
                mode = image.mode

        except (UnidentifiedImageError, OSError, ValueError) as error:
            issues.append({
                "image": str(image_path),
                "type": "unreadable_image",
                "message": str(error),
            })
            continue

        dimension_counts[(width, height)] += 1
        mode_counts[mode] += 1
        hashes[calculate_file_hash(image_path)].append(image_path)

        if item.is_background:
            objects_per_image[0] += 1
            continue

        if item.annotation_path is None:
            issues.append({
                "image": str(image_path),
                "type": "missing_annotation",
                "message": "Positive image has no annotation file",
            })
            continue

        try:
            boxes = parser(item.annotation_path)
        except (ValueError, OSError) as error:
            issues.append({
                "image": str(image_path),
                "type": "annotation_parse_error",
                "message": str(error),
            })
            continue

        objects_per_image[len(boxes)] += 1

        for box_number, box in enumerate(boxes, start=1):
            box_width = box.xmax - box.xmin
            box_height = box.ymax - box.ymin

            class_counts[box.class_id] += 1

            if box.class_id not in valid_class_ids:
                issues.append({
                    "image": str(image_path),
                    "type": "unknown_class",
                    "message": (
                        f"Box {box_number}: class {box.class_id}"
                    ),
                })

            valid_coordinates = (
                0 <= box.xmin < box.xmax <= width
                and 0 <= box.ymin < box.ymax <= height
            )

            if not valid_coordinates:
                issues.append({
                    "image": str(image_path),
                    "type": "invalid_box",
                    "message": (
                        f"Box {box_number}: {asdict(box)}, "
                        f"image size={width}x{height}"
                    ),
                })
                continue

            relative_area = (
                box_width * box_height
            ) / (
                width * height
            )

            if relative_area < 0.01:
                size_category = "small"
            elif relative_area <= 0.05:
                size_category = "medium"
            else:
                size_category = "large"

            box_records.append({
                "image": image_path.name,
                "class_id": box.class_id,
                "xmin": box.xmin,
                "ymin": box.ymin,
                "xmax": box.xmax,
                "ymax": box.ymax,
                "box_width": box_width,
                "box_height": box_height,
                "relative_area": relative_area,
                "size": size_category,
            })

    duplicate_groups = [
        [str(path) for path in paths]
        for paths in hashes.values()
        if len(paths) > 1
    ]

    size_counts = Counter(
        record["size"]
        for record in box_records
    )

    return {
        "total_images": len(items),
        "positive_images": positive_count,
        "background_images": background_count,
        "readable_images": len(items) - sum(
            issue["type"] == "unreadable_image"
            for issue in issues
        ),
        "total_objects": len(box_records),
        "class_counts": dict(sorted(class_counts.items())),
        "size_counts": dict(size_counts),
        "dimension_counts": {
            f"{width}x{height}": count
            for (width, height), count
            in dimension_counts.items()
        },
        "mode_counts": dict(mode_counts),
        "objects_per_image": dict(
            sorted(objects_per_image.items())
        ),
        "duplicate_groups": duplicate_groups,
        "issues": issues,
        "box_records": box_records,
    }
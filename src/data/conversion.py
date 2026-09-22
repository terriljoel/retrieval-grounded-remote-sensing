from collections import Counter
from pathlib import Path

from PIL import Image

from src.data.models import BoundingBox
from src.data.parsers.voc import parse_pascal_voc_annotation


def _format_yolo_box(box: BoundingBox, width: int, height: int) -> str:
    return (
        f"{box.class_id} "
        f"{((box.xmin + box.xmax) / 2) / width:.8f} "
        f"{((box.ymin + box.ymax) / 2) / height:.8f} "
        f"{(box.xmax - box.xmin) / width:.8f} "
        f"{(box.ymax - box.ymin) / height:.8f}"
    )


def convert_pascal_voc_to_yolo(config: dict) -> dict:
    """Convert a configured Pascal VOC dataset to YOLO label files."""
    dataset_root = Path(config["dataset"]["root"])
    image_root = dataset_root / config["source"]["images"]
    annotation_root = dataset_root / config["source"]["annotations"]
    label_root = dataset_root / config["output"]["labels"]
    extensions = {suffix.lower() for suffix in config["source"]["image_extensions"]}
    class_mapping = {
        str(name): int(class_id)
        for name, class_id in config["class_mapping"].items()
    }
    options = config.get("conversion", {})
    overwrite = bool(config["output"].get("overwrite", False))

    if config["source"]["annotation_format"] != "pascal_voc":
        raise ValueError("Source annotation format must be pascal_voc")
    if config["output"]["annotation_format"] != "yolo":
        raise ValueError("Output annotation format must be yolo")
    if not image_root.is_dir() or not annotation_root.is_dir():
        raise FileNotFoundError(f"HRRSD source folders not found under {dataset_root}")

    label_root.mkdir(parents=True, exist_ok=True)
    images = sorted(
        path for path in image_root.iterdir()
        if path.is_file() and path.suffix.lower() in extensions
    )
    class_counts = Counter()
    skipped = 0

    for image_path in images:
        annotation_path = annotation_root / f"{image_path.stem}.xml"
        label_path = label_root / f"{image_path.stem}.txt"
        if label_path.exists() and not overwrite:
            raise FileExistsError(f"YOLO label already exists: {label_path}")
        if not annotation_path.is_file():
            if options.get("fail_on_missing_annotation", True):
                raise FileNotFoundError(f"Missing annotation: {annotation_path}")
            skipped += 1
            continue

        try:
            boxes = parse_pascal_voc_annotation(annotation_path, class_mapping)
        except ValueError:
            if options.get("fail_on_invalid_box", True):
                raise
            skipped += 1
            continue

        with Image.open(image_path) as image:
            width, height = image.size

        output_boxes = []
        for box in boxes:
            if options.get("clamp_boxes", True):
                box = BoundingBox(
                    xmin=max(0, min(box.xmin, width)),
                    ymin=max(0, min(box.ymin, height)),
                    xmax=max(0, min(box.xmax, width)),
                    ymax=max(0, min(box.ymax, height)),
                    class_id=box.class_id,
                )
            if not (0 <= box.xmin < box.xmax <= width
                    and 0 <= box.ymin < box.ymax <= height):
                raise ValueError(f"Box outside {width}x{height}: {box}")
            output_boxes.append(box)
            class_counts[box.class_id] += 1

        if output_boxes or options.get("create_empty_labels", True):
            lines = [_format_yolo_box(box, width, height) for box in output_boxes]
            label_path.write_text(
                "\n".join(lines) + ("\n" if lines else ""),
                encoding="utf-8",
            )

    return {
        "images": len(images),
        "labels": len(list(label_root.glob("*.txt"))),
        "objects": sum(class_counts.values()),
        "class_counts": dict(sorted(class_counts.items())),
        "skipped": skipped,
        "label_root": str(label_root),
    }

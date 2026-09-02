from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from PIL import Image

from src.annotation.models import AnnotationRecord


def save_annotation_session(
    *,
    image: Image.Image,
    original_filename: str,
    annotations: Iterable[AnnotationRecord],
    output_root: Path,
    metadata: dict,
) -> Path:
    records = list(annotations)
    if not records:
        raise ValueError("At least one accepted or corrected annotation is required")
    created_at = datetime.now(timezone.utc)
    image_bytes = image.convert("RGB").tobytes()
    image_hash = hashlib.sha256(image_bytes).hexdigest()
    run_name = f"{created_at.strftime('%Y%m%dT%H%M%SZ')}_{image_hash[:10]}"
    run_directory = Path(output_root).resolve() / run_name
    run_directory.mkdir(parents=True, exist_ok=False)

    image_path = run_directory / "image.jpg"
    image.convert("RGB").save(image_path, format="JPEG", quality=95)

    payload = {
        "created_at_utc": created_at.isoformat(),
        "original_filename": original_filename,
        "stored_image": image_path.name,
        "image_sha256": image_hash,
        "image_width": image.width,
        "image_height": image.height,
        "annotations": [record.to_dict() for record in records],
        "metadata": metadata,
    }
    (run_directory / "annotations.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    yolo_lines = []
    for record in records:
        width = record.box.xmax - record.box.xmin
        height = record.box.ymax - record.box.ymin
        center_x = record.box.xmin + width / 2.0
        center_y = record.box.ymin + height / 2.0
        yolo_lines.append(
            f"{record.class_id} {center_x / image.width:.8f} "
            f"{center_y / image.height:.8f} {width / image.width:.8f} "
            f"{height / image.height:.8f}"
        )
    (run_directory / "labels.txt").write_text(
        "\n".join(yolo_lines) + "\n", encoding="utf-8"
    )
    return run_directory


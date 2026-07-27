import re
from pathlib import Path

from src.data.models import BoundingBox


ANNOTATION_PATTERN = re.compile(
    r"\(\s*(\d+)\s*,\s*(\d+)\s*\),"
    r"\(\s*(\d+)\s*,\s*(\d+)\s*\),"
    r"\s*(\d+)\s*"
)


def parse_nwpu_annotation(
    annotation_path: Path,
) -> list[BoundingBox]:
    boxes = []

    lines = annotation_path.read_text(
        encoding="utf-8",
        errors="replace",
    ).splitlines()

    for line_number, line in enumerate(lines, start=1):
        line = line.strip()

        if not line:
            continue

        match = ANNOTATION_PATTERN.fullmatch(line)

        if match is None:
            raise ValueError(
                f"{annotation_path}, line {line_number}: "
                f"invalid annotation {line!r}"
            )

        xmin, ymin, xmax, ymax, class_id = map(
            int,
            match.groups(),
        )

        boxes.append(
            BoundingBox(
                xmin=xmin,
                ymin=ymin,
                xmax=xmax,
                ymax=ymax,
                class_id=class_id,
            )
        )

    return boxes
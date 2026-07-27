from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BoundingBox:
    xmin: int
    ymin: int
    xmax: int
    ymax: int
    class_id: int


@dataclass(frozen=True)
class DatasetItem:
    image_path: Path
    annotation_path: Path | None
    is_background: bool = False
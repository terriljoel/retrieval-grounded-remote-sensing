from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


NWPU_CLASSES = (
    "airplane",
    "ship",
    "storage_tank",
    "baseball_diamond",
    "tennis_court",
    "basketball_court",
    "ground_track_field",
    "harbor",
    "bridge",
    "vehicle",
)


@dataclass(frozen=True)
class Box:
    xmin: float
    ymin: float
    xmax: float
    ymax: float

    def validate(self, image_width: int, image_height: int) -> None:
        if not (
            0.0 <= self.xmin < self.xmax <= image_width
            and 0.0 <= self.ymin < self.ymax <= image_height
        ):
            raise ValueError(
                f"Box {self} is outside an {image_width}x{image_height} image"
            )

    def as_list(self) -> list[float]:
        return [self.xmin, self.ymin, self.xmax, self.ymax]


@dataclass(frozen=True)
class DetectionSuggestion:
    detection_id: str
    class_id: int
    class_name: str
    confidence: float
    box: Box

    def to_dict(self) -> dict[str, Any]:
        record = asdict(self)
        record["box"] = asdict(self.box)
        return record


@dataclass(frozen=True)
class AnnotationRecord:
    annotation_id: str
    class_id: int
    class_name: str
    box: Box
    decision: str
    source_detection_id: str | None
    detector_confidence: float | None

    def to_dict(self) -> dict[str, Any]:
        record = asdict(self)
        record["box"] = asdict(self.box)
        return record


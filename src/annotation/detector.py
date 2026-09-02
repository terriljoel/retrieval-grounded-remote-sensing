from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from src.annotation.models import Box, DetectionSuggestion


class UltralyticsDetector:
    def __init__(self, checkpoint: Path):
        checkpoint = Path(checkpoint).resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Detector checkpoint not found: {checkpoint}")
        try:
            from ultralytics import YOLO
        except ImportError as error:
            raise RuntimeError(
                "Ultralytics is required. Install the project dependencies first."
            ) from error
        self.checkpoint = checkpoint
        self.model = YOLO(str(checkpoint))

    def predict(
        self,
        image: Image.Image,
        *,
        confidence: float,
        iou: float,
        image_size: int,
        device: str | int | None,
    ) -> list[DetectionSuggestion]:
        result = self.model.predict(
            source=np.asarray(image.convert("RGB")),
            conf=confidence,
            iou=iou,
            imgsz=image_size,
            device=device,
            verbose=False,
        )[0]
        coordinates = result.boxes.xyxy.detach().cpu().tolist()
        class_ids = result.boxes.cls.detach().cpu().tolist()
        confidences = result.boxes.conf.detach().cpu().tolist()
        suggestions: list[DetectionSuggestion] = []
        for index, (coordinates_row, class_value, score) in enumerate(
            zip(coordinates, class_ids, confidences, strict=True)
        ):
            class_id = int(class_value)
            box = Box(*map(float, coordinates_row))
            box.validate(image.width, image.height)
            suggestions.append(DetectionSuggestion(
                detection_id=f"det_{index:04d}",
                class_id=class_id,
            class_name=str(result.names[class_id]).strip().replace(" ", "_"),
                confidence=float(score),
                box=box,
            ))
        return suggestions

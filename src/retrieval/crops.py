from __future__ import annotations

import math

from PIL import Image

from src.annotation.models import Box


def calculate_crop_box(
    box: Box,
    image_width: int,
    image_height: int,
    margin_fraction: float = 0.0,
) -> tuple[int, int, int, int]:
    if margin_fraction < 0.0:
        raise ValueError("margin_fraction must be non-negative")
    box.validate(image_width, image_height)
    width = box.xmax - box.xmin
    height = box.ymax - box.ymin
    left = max(0, math.floor(box.xmin - width * margin_fraction))
    top = max(0, math.floor(box.ymin - height * margin_fraction))
    right = min(image_width, math.ceil(box.xmax + width * margin_fraction))
    bottom = min(image_height, math.ceil(box.ymax + height * margin_fraction))
    if right <= left or bottom <= top:
        raise ValueError(f"Invalid crop bounds: {(left, top, right, bottom)}")
    return left, top, right, bottom


def crop_image(
    image: Image.Image,
    box: Box,
    margin_fraction: float = 0.0,
) -> Image.Image:
    bounds = calculate_crop_box(
        box,
        image.width,
        image.height,
        margin_fraction=margin_fraction,
    )
    return image.convert("RGB").crop(bounds)


"""Two harder query corpora: jittered boxes and background crops.

Clean GT crops saturate at P@5 ~= 0.99, so they cannot rank encoders. These
put the two failure modes a real detector produces back into the test:

  jitter      one perturbed box per GT box, landing at IoU 0.4-0.7 with
              truth -- localisation error, the box is on the object but wrong.
  background  crops containing no annotated object at all -- false positives,
              which the gate must reject. Sizes are drawn from the GT area
              distribution so that size alone cannot separate them from real
              objects; otherwise the novelty score would just be measuring
              "this crop is an unusual size".

Both are seeded and sorted, so a rerun reproduces the same boxes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from src.data.models import BoundingBox
from src.data.parsers.nwpu import parse_nwpu_annotation
from src.retrieval.crops import CLASS_NAMES, DEFAULT_REGIONAL_SCALE, MIN_BOX_SIDE, write_streams

IOU_LOW = 0.4
IOU_HIGH = 0.7
IOU_BANDS = ((0.4, 0.5), (0.5, 0.6), (0.6, 0.7))
# A box this loose is redrawn, not dragged into place. The default corpus only
# spans 0.4-0.7, which is the "adjust" range, so a separate low corpus supplies
# the boxes that genuinely have to be rejected.
LOW_IOU_BANDS = ((0.05, 0.15), (0.15, 0.25), (0.25, 0.40))
JITTER_TRIES = 400
BACKGROUND_TRIES = 200


def iou(a, b) -> float:
    inter_w = min(a.xmax, b.xmax) - max(a.xmin, b.xmin)
    inter_h = min(a.ymax, b.ymax) - max(a.ymin, b.ymin)
    if inter_w <= 0 or inter_h <= 0:
        return 0.0
    intersection = inter_w * inter_h
    union = (
        (a.xmax - a.xmin) * (a.ymax - a.ymin)
        + (b.xmax - b.xmin) * (b.ymax - b.ymin)
        - intersection
    )
    return intersection / union


def iou_band(value: float) -> str:
    for low, high in IOU_BANDS:
        if value < high:
            return f"{low:.1f}-{high:.1f}"
    return f"{IOU_BANDS[-1][0]:.1f}-{IOU_BANDS[-1][1]:.1f}"


def _clamp(box, image_width: int, image_height: int):
    return BoundingBox(
        max(0, min(box.xmin, image_width - MIN_BOX_SIDE)),
        max(0, min(box.ymin, image_height - MIN_BOX_SIDE)),
        min(image_width, max(box.xmax, MIN_BOX_SIDE)),
        min(image_height, max(box.ymax, MIN_BOX_SIDE)),
        box.class_id,
    )


def _shifted(box, target_iou: float, image_width: int, image_height: int):
    """Pure x-translation hitting `target_iou` exactly.

    For a shift d along x with the box otherwise unchanged,
    IoU = (w-d)/(w+d), so d = w(1-t)/(1+t). Used as the fallback when
    rejection sampling cannot place a box (tiny boxes wedged into a corner).
    """
    width = box.xmax - box.xmin
    shift = width * (1 - target_iou) / (1 + target_iou)
    for direction in (1, -1):
        moved = BoundingBox(
            box.xmin + direction * shift, box.ymin,
            box.xmax + direction * shift, box.ymax, box.class_id,
        )
        if 0 <= moved.xmin and moved.xmax <= image_width:
            return moved
    return None


def jitter_box(box, image_width: int, image_height: int, rng, bands=IOU_BANDS) -> tuple:
    """Perturb by random shift and scale to land in a randomly chosen IoU band.

    The band is drawn first and then hit by rejection sampling, so the bands
    come out roughly balanced instead of piling up at the top of the range
    (which is what unconstrained jitter does -- most small perturbations barely
    move the IoU).
    """
    low, high = bands[rng.integers(len(bands))]
    width = box.xmax - box.xmin
    height = box.ymax - box.ymin

    for _ in range(JITTER_TRIES):
        scale_x, scale_y = rng.uniform(0.65, 1.45, size=2)
        shift_x, shift_y = rng.uniform(-0.45, 0.45, size=2) * (width, height)
        centre_x = (box.xmin + box.xmax) / 2 + shift_x
        centre_y = (box.ymin + box.ymax) / 2 + shift_y
        candidate = _clamp(
            BoundingBox(
                centre_x - width * scale_x / 2, centre_y - height * scale_y / 2,
                centre_x + width * scale_x / 2, centre_y + height * scale_y / 2,
                box.class_id,
            ),
            image_width, image_height,
        )
        if candidate.xmax - candidate.xmin < MIN_BOX_SIDE:
            continue
        if candidate.ymax - candidate.ymin < MIN_BOX_SIDE:
            continue
        value = iou(box, candidate)
        if low <= value < high:
            return candidate, value

    fallback = _shifted(box, (low + high) / 2, image_width, image_height)
    if fallback is None:
        return None, 0.0
    return fallback, iou(box, fallback)


def build_jitter_corpus(
    scope: list[dict],
    output_root: Path,
    sensors: dict[str, str],
    seed: int,
    regional_scales: tuple[float, ...] = (DEFAULT_REGIONAL_SCALE,),
    force: bool = False,
    bands=IOU_BANDS,
) -> list[dict]:
    """One jittered crop per GT box, same crop_id so it pairs with the clean set."""
    rng = np.random.default_rng(seed)
    index: list[dict] = []

    for record in scope:
        with Image.open(record["image_path"]) as image:
            image = image.convert("RGB")
            image_width, image_height = image.size

            for box_index, box in enumerate(parse_nwpu_annotation(record["annotation_path"])):
                if (box.xmax - box.xmin) < MIN_BOX_SIDE or (box.ymax - box.ymin) < MIN_BOX_SIDE:
                    continue
                crop_id = f"{record['image_uid']}_{box_index:03d}"
                jittered, value = jitter_box(box, image_width, image_height, rng, bands)
                if jittered is None:
                    continue

                entry = {
                    "crop_id": crop_id,
                    "image_uid": record["image_uid"],
                    "split": record["split"],
                    "class_id": box.class_id,
                    "class_name": CLASS_NAMES[box.class_id],
                    "xmin": round(jittered.xmin), "ymin": round(jittered.ymin),
                    "xmax": round(jittered.xmax), "ymax": round(jittered.ymax),
                    "area_px": round(
                        (jittered.xmax - jittered.xmin) * (jittered.ymax - jittered.ymin)
                    ),
                    "image_width": image_width,
                    "image_height": image_height,
                    "sensor": sensors.get(record["image_uid"], ""),
                    "iou": round(value, 4),
                    "iou_band": iou_band(value),
                    "skipped_reason": "",
                }
                entry.update(
                    write_streams(image, jittered, crop_id, output_root, regional_scales, force)
                )
                index.append(entry)

    return index


def _overlaps_any(candidate, boxes) -> bool:
    return any(
        min(candidate.xmax, box.xmax) > max(candidate.xmin, box.xmin)
        and min(candidate.ymax, box.ymax) > max(candidate.ymin, box.ymin)
        for box in boxes
    )


def _place(width: float, height: float, image_width: int, image_height: int, rng, avoid):
    """A box of the given size at a random in-bounds position touching nothing in `avoid`."""
    if width >= image_width or height >= image_height:
        return None
    for _ in range(BACKGROUND_TRIES):
        xmin = rng.uniform(0, image_width - width)
        ymin = rng.uniform(0, image_height - height)
        candidate = BoundingBox(xmin, ymin, xmin + width, ymin + height, 0)
        if not _overlaps_any(candidate, avoid):
            return candidate
    return None


def build_background_corpus(
    negative_scope: list[dict],
    positive_scope: list[dict],
    gt_sizes: list[tuple[float, float]],
    output_root: Path,
    sensors: dict[str, str],
    seed: int,
    count_negative: int,
    count_positive: int,
    regional_scales: tuple[float, ...] = (DEFAULT_REGIONAL_SCALE,),
    force: bool = False,
) -> list[dict]:
    """Object-free crops: `count_negative` from the negative image set, plus
    `count_positive` from regions of positive images that touch no GT box.

    Box sizes are sampled (w, h) jointly from the real GT boxes, so aspect
    ratio and area match the object corpus.
    """
    rng = np.random.default_rng(seed)
    index: list[dict] = []

    plans: list[tuple[list[dict], int, str]] = [
        (negative_scope, count_negative, "negative_image"),
        (positive_scope, count_positive, "positive_free"),
    ]

    for scope, count, source in plans:
        if not scope or count <= 0:
            continue
        # Spread the draws evenly over images rather than sampling images with
        # replacement, so no single tile dominates the background set.
        picks: dict[int, int] = {}
        for draw in range(count):
            picks[draw % len(scope)] = picks.get(draw % len(scope), 0) + 1

        for image_index in sorted(picks):
            record = scope[image_index]
            with Image.open(record["image_path"]) as image:
                image = image.convert("RGB")
                image_width, image_height = image.size
                avoid = (
                    list(parse_nwpu_annotation(record["annotation_path"]))
                    if record["annotation_path"]
                    else []
                )

                for draw in range(picks[image_index]):
                    width, height = gt_sizes[rng.integers(len(gt_sizes))]
                    placed = _place(width, height, image_width, image_height, rng, avoid)
                    if placed is None:
                        continue
                    crop_id = f"bg_{source}_{record['image_uid']}_{draw:03d}"
                    entry = {
                        "crop_id": crop_id,
                        "image_uid": record["image_uid"],
                        "split": record["split"],
                        "source": source,
                        "class_id": 0,
                        "class_name": "background",
                        "xmin": round(placed.xmin), "ymin": round(placed.ymin),
                        "xmax": round(placed.xmax), "ymax": round(placed.ymax),
                        "area_px": round(width * height),
                        "image_width": image_width,
                        "image_height": image_height,
                        "sensor": sensors.get(record["image_uid"], ""),
                        "skipped_reason": "",
                    }
                    entry.update(
                        write_streams(image, placed, crop_id, output_root, regional_scales, force)
                    )
                    index.append(entry)

    return sorted(index, key=lambda entry: entry["crop_id"])


def _self_check() -> None:
    box = BoundingBox(100, 100, 200, 200, 1)
    assert iou(box, box) == 1.0
    assert iou(box, BoundingBox(300, 300, 400, 400, 1)) == 0.0
    # Half-overlapping squares: intersection 100x50, union 2*10000-5000.
    assert abs(iou(box, BoundingBox(100, 150, 200, 250, 1)) - 5000 / 15000) < 1e-9

    # The closed-form shift lands on its target.
    for target in (0.4, 0.55, 0.7):
        moved = _shifted(box, target, 500, 500)
        assert abs(iou(box, moved) - target) < 1e-6, (target, iou(box, moved))

    # Jitter always lands inside the band, in bounds, for a range of shapes.
    rng = np.random.default_rng(0)
    shapes = [
        BoundingBox(100, 100, 200, 200, 1),      # square, room to move
        BoundingBox(0, 0, 12, 6, 1),             # tiny, wedged in a corner
        BoundingBox(400, 10, 500, 30, 1),        # elongated, against the edge
    ]
    for shape in shapes:
        for _ in range(200):
            jittered, value = jitter_box(shape, 500, 500, rng)
            assert jittered is not None
            assert IOU_LOW <= value <= IOU_HIGH, (shape, value)
            assert 0 <= jittered.xmin and jittered.xmax <= 500
            assert 0 <= jittered.ymin and jittered.ymax <= 500

    # A jittered box must not be the box: IoU 1.0 would silently re-run clean.
    jittered, value = jitter_box(shapes[0], 500, 500, rng)
    assert value < 1.0

    # Background placement never touches a forbidden box, even when the free
    # area is a thin strip.
    avoid = [BoundingBox(0, 0, 400, 500, 1)]
    for _ in range(50):
        placed = _place(50, 50, 500, 500, np.random.default_rng(1), avoid)
        assert placed is not None and not _overlaps_any(placed, avoid)
    # No room at all -> None rather than a silent overlap.
    assert _place(50, 50, 500, 500, np.random.default_rng(1), [BoundingBox(0, 0, 500, 500, 1)]) is None

    assert iou_band(0.42) == "0.4-0.5" and iou_band(0.69) == "0.6-0.7"
    print("perturb self-check OK")


if __name__ == "__main__":
    _self_check()

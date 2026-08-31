"""Crop extraction from NWPU ground-truth boxes.

Three views of every box. They are named for the *scale they cover*, not for
what they are assumed to contain -- for elongated classes the "local" crop is
already about half surroundings (harbor 0.47 box fill, ship 0.51), so calling
it an object crop and the other a context crop overstates the separation. What
is actually true is that regional covers ~16x the area of local at k=4, so
they see different scales regardless of fill.

  local      : square on the longer box side, +10% pad, clipped to image
  regional   : same centre, side = k * max(w, h)
  local_tight: the box itself, resized to 224 with aspect distortion

near an edge : shift the window inward, never pad with black

local_tight is a control, not a production stream. It removes the surroundings
that squaring pulls in, at the cost of feeding the encoder a distorted aspect
ratio. Comparing it against `local` is what separates "retrieval works because
of the object" from "retrieval works because of what squaring dragged in".
"""

from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image

from src.data.parsers.nwpu import parse_nwpu_annotation

CLASS_NAMES = {
    1: "airplane", 2: "ship", 3: "storage_tank", 4: "baseball_diamond",
    5: "tennis_court", 6: "basketball_court", 7: "ground_track_field",
    8: "harbor", 9: "bridge", 10: "vehicle",
}

LOCAL_PAD = 0.10           # square side is grown by this fraction
DEFAULT_REGIONAL_SCALE = 4.0
REGIONAL_SCALES = (2.0, 4.0, 8.0)   # the Phase 1 k sweep
CROP_SIZE = 224
JPEG_QUALITY = 95
MIN_BOX_SIDE = 2           # boxes with a 0px or 1px side are skipped
HIGH_UPSAMPLE = 6.0        # above this the local crop is mostly interpolation


def _square_window(
    centre_x: float,
    centre_y: float,
    side: float,
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    """Return an integer square window, shifted inward to stay in bounds.

    The side is clamped to the image so a window can never exceed it; the
    origin is then clamped so the window stays square rather than being
    truncated at the border.
    """
    side_px = max(1, min(round(side), image_width, image_height))
    x0 = min(max(round(centre_x - side_px / 2), 0), image_width - side_px)
    y0 = min(max(round(centre_y - side_px / 2), 0), image_height - side_px)
    return x0, y0, x0 + side_px, y0 + side_px


def local_window(box, image_width: int, image_height: int) -> tuple[int, int, int, int]:
    width = box.xmax - box.xmin
    height = box.ymax - box.ymin
    return _square_window(
        (box.xmin + box.xmax) / 2,
        (box.ymin + box.ymax) / 2,
        max(width, height) * (1.0 + LOCAL_PAD),
        image_width,
        image_height,
    )


def regional_window(
    box,
    image_width: int,
    image_height: int,
    regional_scale: float = DEFAULT_REGIONAL_SCALE,
) -> tuple[int, int, int, int]:
    width = box.xmax - box.xmin
    height = box.ymax - box.ymin
    return _square_window(
        (box.xmin + box.xmax) / 2,
        (box.ymin + box.ymax) / 2,
        max(width, height) * regional_scale,
        image_width,
        image_height,
    )


def tight_window(box, image_width: int, image_height: int) -> tuple[int, int, int, int]:
    """The box itself, clipped to the image. Not square -- resizing distorts."""
    x0 = min(max(round(box.xmin), 0), image_width - MIN_BOX_SIDE)
    y0 = min(max(round(box.ymin), 0), image_height - MIN_BOX_SIDE)
    x1 = max(min(round(box.xmax), image_width), x0 + MIN_BOX_SIDE)
    y1 = max(min(round(box.ymax), image_height), y0 + MIN_BOX_SIDE)
    return x0, y0, x1, y1


def stream_windows(
    box,
    image_width: int,
    image_height: int,
    regional_scales: tuple[float, ...] = (DEFAULT_REGIONAL_SCALE,),
) -> dict[str, tuple[int, int, int, int]]:
    """Every stream's window for one box, keyed by stream name."""
    windows = {
        "local": local_window(box, image_width, image_height),
        "local_tight": tight_window(box, image_width, image_height),
    }
    for scale in regional_scales:
        windows[f"regional_k{scale:g}"] = regional_window(
            box, image_width, image_height, scale
        )
    return windows


def box_fill(box, window: tuple[int, int, int, int]) -> float:
    """Fraction of the local window actually occupied by the box.

    Squaring an elongated box on its longer side and then padding pulls in
    surroundings: a perfect square box only reaches 1/1.1^2 = 0.826, and a
    2:1 box reaches 0.413. Low fill means the local crop is already half
    surroundings, which entangles the two streams before comparison.
    """
    side = window[2] - window[0]
    return (box.xmax - box.xmin) * (box.ymax - box.ymin) / (side * side)


def load_scope(
    split_csv: Path,
    dataset_root: Path,
    splits: tuple[str, ...] = ("train", "val"),
    background: bool = False,
) -> list[dict]:
    """Images from the given splits, sorted by image_uid.

    `background=True` returns the negative (object-free) images instead of the
    positives; those have no annotation file.

    Keyed on image_uid: image_id collides 150 times across the positive and
    negative folders, 64 of those pairs across different splits.
    """
    with split_csv.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    required = {"image_uid", "image_path", "annotation_path", "is_background", "split"}
    missing = required - set(rows[0] if rows else {})
    if missing:
        raise ValueError(f"{split_csv} is missing columns: {sorted(missing)}")

    wanted = "True" if background else "False"
    scope = [
        {
            "image_uid": row["image_uid"],
            "split": row["split"],
            "image_path": dataset_root / row["image_path"],
            "annotation_path": (
                dataset_root / row["annotation_path"] if row["annotation_path"] else None
            ),
        }
        for row in rows
        if row["split"] in splits and row["is_background"] == wanted
    ]
    if len({record["image_uid"] for record in scope}) != len(scope):
        raise ValueError("image_uid is not unique within the selected splits")
    return sorted(scope, key=lambda record: record["image_uid"])


def write_streams(
    image: Image.Image,
    box,
    crop_id: str,
    output_root: Path,
    regional_scales: tuple[float, ...] = (DEFAULT_REGIONAL_SCALE,),
    force: bool = False,
) -> dict:
    """Write every stream's JPEG for one box, return the local window's geometry.

    Shared by the clean, jittered and background corpora so all three get
    byte-identical treatment -- a geometry difference between query set and
    memory would show up as a similarity difference and be misread as novelty.
    """
    image_width, image_height = image.size
    windows = stream_windows(box, image_width, image_height, regional_scales)

    for stream, window in windows.items():
        directory = output_root / stream
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{crop_id}.jpg"
        if path.is_file() and not force:
            continue
        image.crop(window).resize((CROP_SIZE, CROP_SIZE), Image.BICUBIC).save(
            path, quality=JPEG_QUALITY
        )

    local_side = windows["local"][2] - windows["local"][0]
    return {
        "local_window_px": local_side,
        # How much of the 224x224 local crop is interpolated rather than
        # measured. >1 means upsampling; >HIGH_UPSAMPLE means mostly invented.
        "upsample_factor": round(CROP_SIZE / local_side, 3),
        "high_upsample": CROP_SIZE / local_side > HIGH_UPSAMPLE,
        "box_fill": round(box_fill(box, windows["local"]), 3),
    }


def extract_crops(
    scope: list[dict],
    output_root: Path,
    regional_scales: tuple[float, ...] = (DEFAULT_REGIONAL_SCALE,),
    force: bool = False,
    sensors: dict[str, str] | None = None,
    parser=parse_nwpu_annotation,
    class_names: dict[int, str] = CLASS_NAMES,
) -> list[dict]:
    """Write every stream's JPEGs, return one index record per box.

    `parser` and `class_names` are swappable so HRRSD (VOC XML, 13 classes) can
    reuse the identical geometry -- a second extraction path would be a second
    place for the crop geometry to drift, and the whole point of the open-set
    test is that the two corpora are cropped the same way.
    """
    index: list[dict] = []
    for record in scope:
        with Image.open(record["image_path"]) as image:
            image = image.convert("RGB")
            image_width, image_height = image.size

            for box_index, box in enumerate(parser(record["annotation_path"])):
                crop_id = f"{record['image_uid']}_{box_index:03d}"
                width = box.xmax - box.xmin
                height = box.ymax - box.ymin
                entry = {
                    "crop_id": crop_id,
                    "image_uid": record["image_uid"],
                    "split": record["split"],
                    "class_id": box.class_id,
                    "class_name": class_names[box.class_id],
                    "xmin": box.xmin, "ymin": box.ymin,
                    "xmax": box.xmax, "ymax": box.ymax,
                    "area_px": width * height,
                    "image_width": image_width,
                    "image_height": image_height,
                    "sensor": (sensors or {}).get(record["image_uid"], ""),
                    "skipped_reason": "",
                }

                if width < MIN_BOX_SIDE or height < MIN_BOX_SIDE:
                    entry["skipped_reason"] = "degenerate_box"
                    entry["local_window_px"] = 0
                    entry["upsample_factor"] = 0.0
                    entry["high_upsample"] = False
                    entry["box_fill"] = 0.0
                    index.append(entry)
                    continue

                entry.update(
                    write_streams(image, box, crop_id, output_root, regional_scales, force)
                )
                index.append(entry)

    return index


def write_crop_index(index: list[dict], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(index[0]))
        writer.writeheader()
        writer.writerows(index)
    return path


def contact_sheet(index: list[dict], crop_root: Path, path: Path, count: int = 10) -> Path:
    """Object/context pairs side by side, for eyeballing crop geometry."""
    from PIL import ImageDraw

    usable = [entry for entry in index if not entry["skipped_reason"]]

    # Show what is most likely to be wrong, not what is most common: round-robin
    # across classes, smallest box first, and force in an edge-touching box.
    by_class: dict[str, list[dict]] = {}
    for entry in sorted(usable, key=lambda entry: (entry["area_px"], entry["crop_id"])):
        by_class.setdefault(entry["class_name"], []).append(entry)

    sample: list[dict] = []
    for rank in range(max(len(entries) for entries in by_class.values())):
        for name in sorted(by_class):
            if rank < len(by_class[name]) and len(sample) < count:
                sample.append(by_class[name][rank])
        if len(sample) >= count:
            break

    touching = next(
        (
            entry for entry in usable
            if entry["xmin"] <= 0 or entry["ymin"] <= 0
            or entry["xmax"] >= entry["image_width"]
            or entry["ymax"] >= entry["image_height"]
        ),
        None,
    )
    if touching is not None and touching not in sample:
        sample[-1] = touching

    label_height = 16
    sheet = Image.new(
        "RGB",
        (CROP_SIZE * 2, (CROP_SIZE + label_height) * len(sample)),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    for row, entry in enumerate(sample):
        top = row * (CROP_SIZE + label_height)
        draw.text(
            (4, top + 3),
            f"{entry['crop_id']}  {entry['class_name']}  "
            f"{entry['xmax'] - entry['xmin']}x{entry['ymax'] - entry['ymin']}px  "
            f"area={entry['area_px']}   [left: local, right: regional]",
            fill="black",
        )
        for column, stream in enumerate(("local", "regional_k4")):
            with Image.open(crop_root / stream / f"{entry['crop_id']}.jpg") as crop:
                sheet.paste(crop, (column * CROP_SIZE, top + label_height))

    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, quality=JPEG_QUALITY)
    return path


def _self_check() -> None:
    """Geometry invariants: square, in bounds, centred when there is room."""
    from src.data.models import BoundingBox

    # Centre box, plenty of room: window is centred and padded.
    box = BoundingBox(100, 100, 140, 180, 1)          # 40x80, longer side 80
    x0, y0, x1, y1 = local_window(box, 500, 500)
    assert x1 - x0 == y1 - y0 == round(80 * 1.10), (x0, y0, x1, y1)
    assert (x0 + x1) // 2 == 120 and (y0 + y1) // 2 == 140

    # Corner box: shifted inward, still square, never negative.
    box = BoundingBox(0, 0, 20, 20, 1)
    x0, y0, x1, y1 = regional_window(box, 500, 500, 4.0)
    assert (x0, y0) == (0, 0) and x1 - x0 == y1 - y0 == 80

    # Bottom-right corner: shifted inward, stays inside the image.
    box = BoundingBox(480, 480, 500, 500, 1)
    x0, y0, x1, y1 = regional_window(box, 500, 500, 4.0)
    assert x1 == 500 and y1 == 500 and x1 - x0 == y1 - y0 == 80

    # Context larger than the image: clamped to the short side, still square.
    box = BoundingBox(100, 100, 300, 300, 1)
    x0, y0, x1, y1 = regional_window(box, 400, 250, 4.0)
    assert x1 - x0 == y1 - y0 == 250
    assert 0 <= x0 and x1 <= 400 and 0 <= y0 and y1 <= 250

    # Fill: a square box cannot exceed 1/1.1^2; a 2:1 box gets half that.
    square = BoundingBox(100, 100, 200, 200, 1)
    assert abs(box_fill(square, local_window(square, 500, 500)) - 0.826) < 0.01
    wide = BoundingBox(100, 100, 300, 200, 1)
    assert abs(box_fill(wide, local_window(wide, 500, 500)) - 0.413) < 0.01

    # Tight window is the box itself -- no squaring, no padding.
    wide = BoundingBox(100, 100, 300, 200, 1)
    assert tight_window(wide, 500, 500) == (100, 100, 300, 200)
    # ...but never escapes the image, and never collapses to zero width.
    x0, y0, x1, y1 = tight_window(BoundingBox(-30, 480, 40, 700, 1), 500, 500)
    assert (x0, y0, x1, y1) == (0, 480, 40, 500)
    x0, y0, x1, y1 = tight_window(BoundingBox(499, 499, 499, 499, 1), 500, 500)
    assert x1 > x0 and y1 > y0 and x1 <= 500 and y1 <= 500

    # A tight window is strictly inside its local window, which is the whole
    # point of the control: same box, less of everything around it.
    tight, local = tight_window(wide, 500, 500), local_window(wide, 500, 500)
    assert (tight[2] - tight[0]) * (tight[3] - tight[1]) < (local[2] - local[0]) ** 2

    assert set(stream_windows(wide, 500, 500, (2.0, 4.0, 8.0))) == {
        "local", "local_tight", "regional_k2", "regional_k4", "regional_k8"
    }

    # Non-square image, wide box near an edge.
    box = BoundingBox(700, 10, 760, 40, 1)
    x0, y0, x1, y1 = local_window(box, 800, 600)
    assert x1 - x0 == y1 - y0 and 0 <= x0 and x1 <= 800 and 0 <= y0 and y1 <= 600

    print("crops self-check OK")


if __name__ == "__main__":
    _self_check()

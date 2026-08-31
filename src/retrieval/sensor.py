"""Detect which sensor an NWPU image came from.

NWPU VHR-10 concatenates Google Earth colour imagery (0.5-2 m GSD) with
pan-sharpened colour-infrared crops from Vaihingen (~0.08 m GSD). In CIR the
near-infrared band is mapped to red, so vegetation is magenta and roads and
roofs are teal -- a colour signature no RGB tile has.

This matters because a class concentrated in one sensor can post high
retrieval purity that reflects sensor matching rather than semantics.

Discriminator: the 99th percentile of (R - G). Whole-image means do NOT work
-- they are unimodal across the corpus, because averaging washes the
vegetation signature out. A high percentile fires on the presence of any
strongly infrared-red pixels and so is robust to how much vegetation a tile
happens to contain: image 442 is CIR with only 3.8% vegetation pixels and is
still caught.

Measured on the 555 train+val positives: RGB tiles span [-20, 84], CIR tiles
span [98, 160]. Nothing lands in between. The threshold sits in that gap.
The resulting CIR set is exactly the contiguous ID block 422-453, which is
independent corroboration -- concatenated datasets keep their source order.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

CIR_P99_THRESHOLD = 91.0     # midpoint of the observed empty gap [84, 98]
MIN_GAP_MARGIN = 5.0         # fail loudly if a corpus lands on the threshold
STATS_THUMBNAIL = 160        # means and percentiles are stable at this size

RGB = "rgb"
CIR = "cir"


def image_sensor_stats(image_path: Path) -> dict:
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        image.thumbnail((STATS_THUMBNAIL, STATS_THUMBNAIL))
        pixels = np.asarray(image, dtype=np.float32)

    red, green, blue = pixels[..., 0], pixels[..., 1], pixels[..., 2]
    return {
        "mean_r": float(red.mean()),
        "mean_g": float(green.mean()),
        "mean_b": float(blue.mean()),
        "mean_r_minus_g": float((red - green).mean()),
        "p99_r_minus_g": float(np.quantile(red - green, 0.99)),
        "frac_r_minus_g_gt30": float((red - green > 30).mean()),
    }


def classify(p99_r_minus_g: float) -> str:
    return CIR if p99_r_minus_g >= CIR_P99_THRESHOLD else RGB


def sensor_table(image_paths: dict[str, Path]) -> dict[str, dict]:
    """Map image_uid -> stats dict with a `sensor` key, checking the margin."""
    table = {
        uid: {**image_sensor_stats(path), "sensor": None}
        for uid, path in sorted(image_paths.items())
    }
    for record in table.values():
        record["sensor"] = classify(record["p99_r_minus_g"])

    values = np.array([record["p99_r_minus_g"] for record in table.values()])
    below = values[values < CIR_P99_THRESHOLD]
    above = values[values >= CIR_P99_THRESHOLD]
    if len(below) and len(above):
        margin = float(above.min() - below.max())
        if margin < MIN_GAP_MARGIN:
            raise ValueError(
                f"Sensor split is not clean: nearest values straddle the "
                f"threshold by only {margin:.1f} (rgb max {below.max():.1f}, "
                f"cir min {above.min():.1f}). Re-examine before trusting it."
            )
    return table


def _self_check() -> None:
    assert classify(160.0) == CIR and classify(98.0) == CIR
    assert classify(84.0) == RGB and classify(-20.0) == RGB

    # A synthetic CIR tile: vegetation with NIR in the red channel.
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "cir.png"
        pixels = np.zeros((64, 64, 3), dtype=np.uint8)
        pixels[..., 2] = 120                 # teal ground
        pixels[..., 1] = 110
        pixels[:8, :, 0] = 240               # 12.5% bright infrared vegetation
        pixels[:8, :, 1] = 40
        Image.fromarray(pixels).save(path)
        assert classify(image_sensor_stats(path)["p99_r_minus_g"]) == CIR

        path = Path(directory) / "rgb.png"
        pixels = np.full((64, 64, 3), 110, dtype=np.uint8)
        pixels[:8, :, 1] = 160               # green vegetation
        Image.fromarray(pixels).save(path)
        assert classify(image_sensor_stats(path)["p99_r_minus_g"]) == RGB

    print("sensor self-check OK")


if __name__ == "__main__":
    _self_check()

"""Test-time crop stability: does the embedding survive a small box perturbation?

Each proposal is re-cropped four more times with the box shifted and scaled by
up to +-10%, and all five views are embedded. The feature is the mean pairwise
cosine across those five embeddings.

Hypothesis: a well-localised box is stable under perturbation because the
object stays centred and fully inside the window, so all five views are
near-identical. A badly-localised box is not, because the object crosses the
window edge and moves in and out of frame as the box wobbles.

This is the **only localisation signal available without the detector**, and it
is worth trying because the crop geometry already hints that it exists: the
`local_tight` arm beats `local` at separating localisation errors (AUROC 0.780
vs 0.693), which is the same mechanism seen from a different angle -- how the
window frames the object carries information the object's identity does not.

The perturbation is applied to the *box*, then the usual local window is
recomputed from it, so a view differs from the original exactly the way a
slightly different detector output would. Perturbing the window directly would
instead measure sensitivity to framing at a fixed box, which is not the
question.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from src.data.models import BoundingBox
from src.retrieval.crops import CROP_SIZE, JPEG_QUALITY, MIN_BOX_SIDE, local_window

VIEWS = 4                  # extra views; the original local crop is the fifth
JITTER = 0.10              # +-10% of box side, on both offset and scale
STREAM_PREFIX = "stability_v"


def stream_names(views: int = VIEWS) -> list[str]:
    return [f"{STREAM_PREFIX}{index + 1}" for index in range(views)]


def perturb_box(
    box, image_width: int, image_height: int, rng: np.random.Generator,
    jitter: float = JITTER,
) -> BoundingBox:
    """Shift and rescale a box by up to +-`jitter` of its own size."""
    width = box.xmax - box.xmin
    height = box.ymax - box.ymin
    # One draw of exactly four numbers, so a resumed run can skip an image by
    # advancing the generator by 4 per view and land in the same place.
    shift_x, shift_y, scale_x, scale_y = rng.uniform(-jitter, jitter, size=4)
    centre_x = (box.xmin + box.xmax) / 2 + shift_x * width
    centre_y = (box.ymin + box.ymax) / 2 + shift_y * height
    new_width = width * (1.0 + scale_x)
    new_height = height * (1.0 + scale_y)

    xmin = int(round(centre_x - new_width / 2))
    ymin = int(round(centre_y - new_height / 2))
    xmax = int(round(centre_x + new_width / 2))
    ymax = int(round(centre_y + new_height / 2))
    # Clip to the image, keeping at least a degenerate-safe side. A box that
    # runs off the edge is exactly the badly-localised case we want to measure,
    # so it is clipped rather than rejected and resampled.
    xmin = min(max(xmin, 0), image_width - MIN_BOX_SIDE)
    ymin = min(max(ymin, 0), image_height - MIN_BOX_SIDE)
    xmax = max(min(xmax, image_width), xmin + MIN_BOX_SIDE)
    ymax = max(min(ymax, image_height), ymin + MIN_BOX_SIDE)
    return BoundingBox(xmin, ymin, xmax, ymax, getattr(box, "class_id", 0))


def write_views(
    index: list[dict],
    image_paths: dict[str, Path],
    output_root: Path,
    seed: int,
    views: int = VIEWS,
    jitter: float = JITTER,
    force: bool = False,
) -> list[str]:
    """Write `views` perturbed local crops per index entry. Returns stream names.

    Images are opened once per image_uid and all of that image's boxes and
    views written together -- opening a 1000x800 TIFF once per crop instead
    would dominate the runtime.
    """
    streams = stream_names(views)
    for stream in streams:
        (output_root / stream).mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    by_image: dict[str, list[dict]] = {}
    for entry in index:
        by_image.setdefault(entry["image_uid"], []).append(entry)

    # Sorted so the rng draw order -- and therefore every view -- is identical
    # on a rerun regardless of how the index happened to be ordered.
    for image_uid in sorted(by_image):
        entries = sorted(by_image[image_uid], key=lambda entry: entry["crop_id"])
        missing = [
            entry for entry in entries for stream in streams
            if force or not (output_root / stream / f"{entry['crop_id']}.jpg").is_file()
        ]
        if not missing:
            # Nothing to write, but the generator must still advance by exactly
            # the draws this image would have consumed, or a resumed run would
            # produce different crops from a run that did it all at once.
            rng.uniform(size=4 * len(entries) * len(streams))
            continue

        with Image.open(image_paths[image_uid]) as image:
            image = image.convert("RGB")
            image_width, image_height = image.size
            for entry in entries:
                box = BoundingBox(
                    int(entry["xmin"]), int(entry["ymin"]),
                    int(entry["xmax"]), int(entry["ymax"]), 0,
                )
                for stream in streams:
                    moved = perturb_box(box, image_width, image_height, rng, jitter)
                    path = output_root / stream / f"{entry['crop_id']}.jpg"
                    if path.is_file() and not force:
                        continue
                    window = local_window(moved, image_width, image_height)
                    image.crop(window).resize(
                        (CROP_SIZE, CROP_SIZE), Image.BICUBIC
                    ).save(path, quality=JPEG_QUALITY)
    return streams


def mean_pairwise_cosine(views: list[np.ndarray]) -> np.ndarray:
    """Mean pairwise cosine across L2-normalised view embeddings, per row.

    `views` is a list of (n, dim) matrices, one per view. Returns (n,).
    """
    stacked = np.stack(views, axis=1)                    # (n, v, dim)
    grams = np.einsum("nvd,nwd->nvw", stacked, stacked)  # (n, v, v)
    count = stacked.shape[1]
    upper = np.triu_indices(count, k=1)
    return grams[:, upper[0], upper[1]].mean(axis=1)


def _self_check() -> None:
    rng = np.random.default_rng(0)

    box = BoundingBox(100, 100, 200, 160, 1)
    for _ in range(200):
        moved = perturb_box(box, 500, 400, rng)
        assert 0 <= moved.xmin < moved.xmax <= 500
        assert 0 <= moved.ymin < moved.ymax <= 400
        # A +-10% jitter cannot move the box more than ~20% of its own size.
        assert abs(moved.xmin - box.xmin) <= 0.25 * (box.xmax - box.xmin) + 1
    # A box at the image edge must clip, not produce negative coordinates.
    edge = perturb_box(BoundingBox(0, 0, 6, 6, 0), 8, 8, rng)
    assert edge.xmin >= 0 and edge.xmax <= 8, edge

    # Seeded: the same seed gives the same perturbations.
    a = [perturb_box(box, 500, 400, np.random.default_rng(7)) for _ in range(1)]
    b = [perturb_box(box, 500, 400, np.random.default_rng(7)) for _ in range(1)]
    assert a == b

    # Resume-equivalence: skipping an already-written image by advancing the
    # generator 4 draws per view must land on the same box a full run would.
    full = np.random.default_rng(11)
    for _ in range(3):
        perturb_box(box, 500, 400, full)
    resumed = np.random.default_rng(11)
    resumed.uniform(size=4 * 3)
    assert perturb_box(box, 500, 400, full) == perturb_box(box, 500, 400, resumed)

    # Identical views are perfectly stable; orthogonal ones are not.
    same = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    assert np.allclose(mean_pairwise_cosine([same] * 5), 1.0)

    spread = [
        np.array([[1.0, 0.0]]), np.array([[0.0, 1.0]]),
        np.array([[1.0, 0.0]]), np.array([[0.0, 1.0]]),
    ]
    # Four views, two orthogonal pairs: 6 pairs, 2 at cosine 1, 4 at cosine 0.
    assert np.allclose(mean_pairwise_cosine(spread), 2.0 / 6.0), mean_pairwise_cosine(spread)

    # A stable row and an unstable row must be ordered correctly.
    stable = np.array([[1.0, 0.0], [1.0, 0.0]])
    wobbly = np.array([[1.0, 0.0], [0.6, 0.8]])
    scores = mean_pairwise_cosine([stable, wobbly, stable])
    assert scores[0] > scores[1], scores

    print("stability self-check OK")


if __name__ == "__main__":
    _self_check()

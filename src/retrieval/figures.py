"""Diagnostic figures: box-area histogram (8.2) and UMAP panels (8.4)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

HIST_CLASSES = ("vehicle", "ship", "storage_tank")
UMAP_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1


def otsu_threshold(values: np.ndarray, bins: int = 64) -> tuple[float, float]:
    """1-D Otsu split of `values`, returning (threshold, between-class variance ratio).

    The ratio is between-class variance over total variance: near 1 means two
    tight, well-separated modes; near 0 means one blob and the "threshold" is
    meaningless. Reported alongside the threshold so bimodality is a number,
    not an impression from looking at a histogram.
    """
    counts, edges = np.histogram(values, bins=bins)
    centres = (edges[:-1] + edges[1:]) / 2
    weights = counts / counts.sum()
    total_mean = float((weights * centres).sum())

    best_variance, best_threshold = -1.0, float(centres[0])
    for split in range(1, bins):
        w0 = weights[:split].sum()
        if w0 <= 0 or w0 >= 1:
            continue
        mean0 = float((weights[:split] * centres[:split]).sum() / w0)
        mean1 = float((weights[split:] * centres[split:]).sum() / (1 - w0))
        variance = w0 * (1 - w0) * (mean0 - mean1) ** 2
        if variance > best_variance:
            best_variance, best_threshold = variance, float(edges[split])

    total_variance = float((weights * (centres - total_mean) ** 2).sum())
    return best_threshold, best_variance / total_variance if total_variance else 0.0


def box_area_hist(index: list[dict], path: Path) -> Path:
    """Box areas for vehicle/ship/storage_tank, split by sensor.

    Bimodality here would mean a hidden GSD split, and retrieval could be
    matching sensor rather than semantics.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, len(HIST_CLASSES), figsize=(4 * len(HIST_CLASSES), 3.4))
    for axis, name in zip(axes, HIST_CLASSES):
        rows = [entry for entry in index if entry["class_name"] == name]
        areas = np.log10([max(entry["area_px"], 1) for entry in rows])
        bins = np.linspace(areas.min(), areas.max(), 40)
        for sensor, colour in (("rgb", "#4C72B0"), ("cir", "#C44E52")):
            selection = [
                np.log10(max(entry["area_px"], 1))
                for entry in rows
                if entry["sensor"] == sensor
            ]
            if selection:
                axis.hist(selection, bins=bins, color=colour, alpha=0.75,
                          label=f"{sensor} (n={len(selection)})")
        threshold, separation = otsu_threshold(areas)
        axis.axvline(threshold, color="black", linestyle="--", linewidth=1)
        axis.set_title(f"{name}  n={len(rows)}\nOtsu 10^{threshold:.2f} px^2, "
                       f"separation {separation:.2f}", fontsize=9)
        axis.set_xlabel("log10 box area (px^2)")
        axis.legend(fontsize=7)
    axes[0].set_ylabel("boxes")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=140)
    plt.close(figure)
    return path


def umap_panel(
    embeddings: np.ndarray,
    class_names: np.ndarray,
    path: Path,
    title: str,
    seed: int = 42,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import umap

    reducer = umap.UMAP(
        n_neighbors=UMAP_NEIGHBORS, min_dist=UMAP_MIN_DIST,
        metric="cosine", random_state=seed,
    )
    points = reducer.fit_transform(embeddings)

    figure, axis = plt.subplots(figsize=(6.5, 5.5))
    names = sorted(set(class_names.tolist()))
    colours = plt.cm.tab10(np.linspace(0, 1, max(len(names), 10)))
    for colour, name in zip(colours, names):
        selection = class_names == name
        axis.scatter(points[selection, 0], points[selection, 1], s=4,
                     color=colour, label=f"{name} ({selection.sum()})", alpha=0.7)
    axis.set_title(title, fontsize=10)
    axis.set_xticks([])
    axis.set_yticks([])
    axis.legend(fontsize=7, markerscale=2, loc="best")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=140)
    plt.close(figure)
    return path


def _self_check() -> None:
    # Two well-separated modes: Otsu lands between them with high separation.
    values = np.concatenate([np.full(100, 1.0), np.full(100, 5.0)])
    threshold, separation = otsu_threshold(values)
    assert 1.0 < threshold < 5.0 and separation > 0.95, (threshold, separation)

    # One mode: separation must be low, so a spurious threshold is visible as such.
    _, separation = otsu_threshold(np.random.default_rng(0).normal(0, 1, 2000))
    assert separation < 0.8, separation
    print("figures self-check OK")


if __name__ == "__main__":
    _self_check()

"""Build the test-time crop-stability views: 4 perturbed re-crops per box, embedded.

Writes `stability_v1..v4` streams alongside the existing ones for all three
corpora, then embeds them. The feature itself (mean pairwise cosine across the
five views) is computed where it is used -- this script only produces the
inputs, because they cost ~31k ViT-L/14 forward passes and must be cached.

Idempotent: crops and embeddings already on disk are left alone, and the
generator is advanced so a resumed run produces identical views.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.retrieval.crops import load_scope
from src.retrieval.embed import embed_stream
from src.retrieval.phase1 import (
    DATASET_ROOT, ENCODER, MAIN_SCALE, SEED, SPLIT_CSV, build_corpora, spaced,
)
from src.retrieval.stability import VIEWS, mean_pairwise_cosine, stream_names, write_views
from src.utils.paths import find_project_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder", default=ENCODER)
    parser.add_argument("--scale", type=float, default=MAIN_SCALE)
    parser.add_argument("--views", type=int, default=VIEWS)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def image_path_table(root: Path, limit: int | None) -> dict[str, Path]:
    table = {}
    for background in (False, True):
        scope = spaced(
            load_scope(root / SPLIT_CSV, root / DATASET_ROOT, background=background), limit
        )
        table.update({record["image_uid"]: record["image_path"] for record in scope})
    return table


def main() -> None:
    args = parse_args()
    root = find_project_root(Path(__file__).resolve())
    crop_root = root / "data" / "crops"
    cache_root = root / "data" / "embeddings"

    print(f"Building corpora at k={args.scale:g}")
    index, _ = build_corpora(root, args.scale, limit=args.limit)
    paths = image_path_table(root, args.limit)
    streams = stream_names(args.views)

    for corpus, entries in index.items():
        print(f"\n{corpus}: {len(entries)} boxes x {args.views} views")
        write_views(
            entries, paths, crop_root / corpus, SEED, args.views, force=args.force
        )
        ids = [entry["crop_id"] for entry in entries]
        views = [
            embed_stream(ids, crop_root / corpus, cache_root / corpus,
                         args.encoder, stream, force=args.force)
            for stream in ["local"] + streams
        ]
        score = mean_pairwise_cosine(views)
        print(f"  stability: mean {score.mean():.4f}  "
              f"p5 {np.percentile(score, 5):.4f}  p95 {np.percentile(score, 95):.4f}")

    print("\nDone. Run scripts.run_baselines to use it as a feature.")


if __name__ == "__main__":
    main()

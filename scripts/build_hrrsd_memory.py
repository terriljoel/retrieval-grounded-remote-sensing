"""Build the HRRSD memory partition: crops, embeddings, cases.db with dataset='hrrsd'.

The committed 4,000-image subset of HRRSD's train split, cropped with exactly
the NWPU geometry (same `extract_crops`, only the parser and class map differ).

Every case is an `accept` -- these are verified ground-truth boxes, with no
jitter or background corpus behind them. The partition is therefore a
*positive evidence pool*: usable for purity, for the memory-growth curve and
for the contamination check, and usable as extra positives in a combined
memory, but it cannot drive the gate on its own because `evidence()` needs
negatives too.

RemoteCLIP was trained on HRRSD and not on NWPU, so numbers measured here are
inflated. That is the reason for the `dataset` column, and the reason the two
partitions are separate files rather than one merged memory.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from src.retrieval.crops import DEFAULT_REGIONAL_SCALE, extract_crops, write_crop_index
from src.retrieval.embed import embed_stream
from src.retrieval.hrrsd import load_class_map, load_scope, open_set_classes, voc_parser
from src.retrieval.memory import build_memory
from src.utils.paths import find_project_root

SEED = 42
ENCODER = "remoteclip_l14"
ARCHIVE = "data/hrrsd/TGRS-HRRSD-Dataset/archive"
SUBSET = "configs/splits/hrrsd_memory_subset.txt"
STREAMS = ("local", "local_tight", f"regional_k{DEFAULT_REGIONAL_SCALE:g}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder", default=ENCODER)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="Smoke run: N images.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = find_project_root(Path(__file__).resolve())
    archive = root / ARCHIVE
    crop_root = root / "data" / "crops" / "hrrsd"
    cache_root = root / "data" / "embeddings" / "hrrsd"
    memory_root = root / "data" / "memory"

    class_map = load_class_map(archive)
    scope = load_scope(archive, root / SUBSET)
    if args.limit:
        scope = scope[: args.limit]
    print(f"HRRSD subset: {len(scope)} images, {len(class_map)} classes")
    print(f"  open-set (not in NWPU): {open_set_classes(class_map)}")

    index = [
        entry for entry in extract_crops(
            scope, crop_root, (DEFAULT_REGIONAL_SCALE,), args.force,
            sensors=None, parser=voc_parser(class_map), class_names=class_map,
        )
        if not entry["skipped_reason"]
    ]
    write_crop_index(index, root / "reports" / "phase1" / "hrrsd_crop_index.csv")
    counts = Counter(entry["class_name"] for entry in index)
    print(f"\n{len(index)} boxes, {len(index) / len(scope):.2f} per image")
    for name, count in sorted(counts.items()):
        print(f"  {name:22}{count:6d}")

    ids = [entry["crop_id"] for entry in index]
    embeddings = {
        stream: embed_stream(ids, crop_root, cache_root, args.encoder, stream, force=args.force)
        for stream in STREAMS
    }

    # Ground truth, so predicted == verified and every decision is accept.
    cases = [
        {
            "proposal_id": f"hrrsd:{entry['crop_id']}",
            "corpus": "clean", "crop_id": entry["crop_id"],
            "image_uid": entry["image_uid"], "split": entry["split"],
            "dataset": "hrrsd", "sensor": "rgb",
            "xmin": entry["xmin"], "ymin": entry["ymin"],
            "xmax": entry["xmax"], "ymax": entry["ymax"],
            "area_px": entry["area_px"], "upsample_factor": entry["upsample_factor"],
            "box_fill": entry["box_fill"], "detector_confidence": None, "iou": 1.0,
            "predicted_class": entry["class_name"], "verified_class": entry["class_name"],
            "verdict": "accept",
        }
        for entry in index
    ]
    build_memory(
        cases, embeddings,
        memory_root / "cases_hrrsd.db", memory_root / "embeddings_hrrsd.npy",
        args.encoder, SEED,
    )
    print(f"\nWrote {memory_root}/cases_hrrsd.db ({len(cases)} accept cases, dataset='hrrsd')")


if __name__ == "__main__":
    main()

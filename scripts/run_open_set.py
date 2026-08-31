"""Leave-one-class-out open-set test: does novelty flag a class the memory has never seen?

For each NWPU class C, every case whose verified class is C is removed from the
memory -- accepts, relabels and localisation rejects alike, so no class-C
pixels remain in the memory under any decision. Val accepts of class C are then
queried against that reduced memory and compared against val accepts of the
nine classes it still knows.

This is a stronger test than swapping in a different dataset: the imagery,
sensor, resolution, crop geometry and encoder are all held fixed, so the only
thing that changes is whether the memory has seen the class. An HRRSD-based
open-set test cannot separate "novel class" from "different dataset", and
RemoteCLIP trained on HRRSD would confound it further.

Reported per class at the operating point, not by AUROC alone: the threshold
that flags 95% of the held-out class as novel, and the fraction of familiar
accepts that threshold falsely flags.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from src.retrieval.fuse import fused_similarity
from src.retrieval.gate import auroc, rejection_threshold
from src.retrieval.phase1 import ENCODER, MAIN_SCALE, SEED, prepare, selected_alpha
from src.utils.paths import find_project_root

TARGET_NOVEL_RATE = 0.95


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder", default=ENCODER)
    parser.add_argument("--scale", type=float, default=MAIN_SCALE)
    parser.add_argument("--alpha", type=float, default=None,
                        help="Default: whatever the sweep selected on bg95 cost.")
    parser.add_argument("--arm", default="local", choices=("local", "local_tight"))
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    np.random.seed(SEED)
    root = find_project_root(Path(__file__).resolve())
    report_root = root / "reports" / "phase1"

    alpha = args.alpha
    if alpha is None:
        alpha = selected_alpha(report_root / "alpha_sweep_summary.csv")
    print(f"Open-set test: arm={args.arm} alpha={alpha} k={args.scale:g} "
          f"encoder={args.encoder}")

    setup = prepare(root, args.encoder, args.scale, limit=args.limit)

    # Only accepts carry positive evidence, and only accepts are queried: the
    # question is whether an unseen *object* looks unfamiliar, not whether a
    # background patch does (that is the background corpus's job).
    memory = [r for r in setup.train if r["verdict"] == "accept"]
    queries = [r for r in setup.val if r["verdict"] == "accept"]

    memory_local = setup.gather(memory, args.arm)
    memory_regional = setup.gather(memory, setup.regional)
    query_local = setup.gather(queries, args.arm)
    query_regional = setup.gather(queries, setup.regional)

    memory_class = np.array([r["verified_class"] for r in memory])
    query_class = np.array([r["verified_class"] for r in queries])

    # One matmul for everything; each fold is a column mask over it.
    similarity = fused_similarity(
        query_local, query_regional, memory_local, memory_regional, alpha
    )

    rows = []
    for held_out in setup.class_names:
        keep = memory_class != held_out
        is_novel = query_class == held_out
        if not is_novel.any() or keep.sum() < 5:
            continue

        sim_at_1 = similarity[:, keep].max(axis=1)
        novelty = 1.0 - sim_at_1

        novel_scores = novelty[is_novel]
        familiar_scores = novelty[~is_novel]
        # rejection_threshold expects (keep-scores, reject-scores) with higher
        # meaning more likely to be kept, so novelty is passed negated.
        threshold, false_flag = rejection_threshold(
            -familiar_scores, -novel_scores, TARGET_NOVEL_RATE
        )
        rows.append({
            "held_out_class": held_out,
            "n_query_novel": int(is_novel.sum()),
            "n_query_familiar": int((~is_novel).sum()),
            "n_memory_after_removal": int(keep.sum()),
            "n_memory_removed": int((~keep).sum()),
            "mean_novelty_novel": round(float(novel_scores.mean()), 4),
            "mean_novelty_familiar": round(float(familiar_scores.mean()), 4),
            "auroc": round(auroc(novel_scores, familiar_scores), 4),
            "novelty_threshold_at_95": round(float(-threshold), 4),
            "false_flag_rate_at_95": round(float(false_flag), 4),
        })

    print(f"\n{'held-out class':20}{'n':>5}{'mem':>7}{'nov':>8}{'fam':>8}"
          f"{'AUROC':>8}{'thr':>8}{'false-flag':>12}")
    for row in sorted(rows, key=lambda r: -r["auroc"]):
        print(f"{row['held_out_class']:20}{row['n_query_novel']:5d}"
              f"{row['n_memory_after_removal']:7d}"
              f"{row['mean_novelty_novel']:8.3f}{row['mean_novelty_familiar']:8.3f}"
              f"{row['auroc']:8.3f}{row['novelty_threshold_at_95']:8.3f}"
              f"{row['false_flag_rate_at_95']:11.1%}")

    areas = [row["auroc"] for row in rows]
    costs = [row["false_flag_rate_at_95"] for row in rows]
    print(f"\nmacro AUROC {np.mean(areas):.3f} (min {min(areas):.3f}, max {max(areas):.3f})")
    print(f"macro false-flag at 95% novel detection: {np.mean(costs):.1%} "
          f"(min {min(costs):.1%}, max {max(costs):.1%})")

    report_root.mkdir(parents=True, exist_ok=True)
    path = report_root / "open_set_leave_one_class_out.csv"
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()

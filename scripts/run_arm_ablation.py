"""Ablation: does a third evidence arm buy anything B2 does not already have?

Finding 14 left this open. `local` is the better background detector and
`local_tight` the better localisation detector, neither dominates, and finding
20 showed B2 already uses the two *oppositely* -- paired features carry
opposite weights, so the model reads their difference. The obvious next
question is whether the regional view adds a third independent read or is
already spent inside the alpha fusion.

Three arm sets, refit from scratch, everything else held fixed:

    local                              one view
    local + local_tight                current B2
    local + local_tight + regional_k4  the proposed third arm

Each arm contributes its own block of retrieval features, computed from its
own similarity matrix. Note what the regional arm actually is: `build_features`
scores an arm as `alpha*cos(arm) + (1-alpha)*cos(regional)`, so passing the
regional stream *as* an arm collapses to plain `cos(regional)` at any alpha --
which is exactly the wanted thing, an unmixed regional evidence stream rather
than a second copy of the mixture already inside the other two arms.

The holdout split is drawn from the same seed for every arm set, so the three
rows differ only in how many feature blocks the model sees.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from src.retrieval.phase1 import ENCODER, MAIN_SCALE, SEED, prepare, selected_alpha
from src.utils.paths import find_project_root
from scripts.run_baselines import (
    FALSE_ACCEPT_BUDGETS, HARMS, HOLDOUT_FRACTION, aurc, build_features,
    coverage_at_budget, harm_curves, stability_available,
)

REPORTED = ("class_error", "background_accept", "localisation_accept", "combined")
# Coverage @1% on ~116 relabels is a small-sample statistic. A single holdout
# draw cannot tell a real arm effect from which images landed in the fit set,
# and this ablation is being read against a recorded finding, so it is averaged.
REPEATS = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder", default=ENCODER)
    parser.add_argument("--scale", type=float, default=MAIN_SCALE)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-stability", action="store_true")
    parser.add_argument("--repeats", type=int, default=REPEATS,
                        help="Holdout draws per arm set; results are averaged.")
    return parser.parse_args()


def holdout_split(train: list[dict], seed: int) -> tuple[list[dict], list[dict]]:
    """Split the memory by image, never by proposal.

    Splitting by proposal would leave a fitting example's own tile in the
    memory it is scored against, and the model would learn sim@1 == 1.
    """
    rng = np.random.default_rng(seed)
    images = sorted({r["image_uid"] for r in train})
    shuffled = np.array(images)[rng.permutation(len(images))]
    holdout = set(shuffled[: int(len(shuffled) * HOLDOUT_FRACTION)].tolist())
    fit_memory = [r for r in train if r["image_uid"] not in holdout]
    fit_queries = [r for r in train if r["image_uid"] in holdout]
    assert not ({r["image_uid"] for r in fit_memory} & holdout)
    return fit_memory, fit_queries


def fit_and_score(setup, fit_queries, fit_memory, val, memory, alpha, arms, stability):
    """Refit B2 over `arms` only and score `val`. Returns (scores, weights)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    fit_x, names = build_features(setup, fit_queries, fit_memory, alpha, arms, stability)
    fit_y = np.array([r["verdict"] == "accept" for r in fit_queries])
    val_x, _ = build_features(setup, val, memory, alpha, arms, stability)

    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=SEED))
    model.fit(fit_x, fit_y)
    weights = dict(zip(names, model[-1].coef_[0]))
    return model.predict_proba(val_x)[:, 1], weights


def main() -> None:
    args = parse_args()
    root = find_project_root(Path(__file__).resolve())
    report_root = root / "reports" / "phase1"

    alpha = args.alpha
    if alpha is None:
        alpha = selected_alpha(report_root / "alpha_sweep_summary.csv")

    setup = prepare(root, args.encoder, args.scale, limit=args.limit)
    arm_sets = {
        "local": ("local",),
        "local+tight": ("local", "local_tight"),
        "local+tight+regional": ("local", "local_tight", setup.regional),
    }
    print(f"Arm ablation: alpha={alpha} k={args.scale:g} encoder={args.encoder}")

    train, val = setup.train, setup.val
    stability = stability_available(setup) and not args.no_stability
    if not stability:
        print("crop_stability: NOT included")

    # Every arm set sees the SAME sequence of holdout draws, so the three rows
    # differ only in feature blocks and the repeat-to-repeat noise is shared.
    splits = [holdout_split(train, SEED + repeat) for repeat in range(args.repeats)]
    print(f"Holdout: {len(splits[0][1])} fitting proposals against {len(splits[0][0])} "
          f"memory cases; query {len(val)}; {args.repeats} draws\n")

    verdicts = np.array([r["verdict"] for r in val])
    accepts = verdicts == "accept"

    rows, weight_rows = [], []
    table: dict[tuple[str, str], list[float]] = {}
    for label, arms in arm_sets.items():
        coverages: dict[str, list[list[float]]] = {harm: [] for harm in HARMS}
        aurcs: dict[str, list[float]] = {harm: [] for harm in HARMS}
        weight_draws: dict[str, list[float]] = {}
        for fit_memory, fit_queries in splits:
            scores, weights = fit_and_score(
                setup, fit_queries, fit_memory, val, train, alpha, arms, stability
            )
            for harm, (coverage, risk) in harm_curves(scores, verdicts, accepts).items():
                coverages[harm].append([
                    coverage_at_budget(coverage, risk, budget)
                    for budget in FALSE_ACCEPT_BUDGETS
                ])
                aurcs[harm].append(aurc(risk))
            for name, weight in weights.items():
                weight_draws.setdefault(name, []).append(float(weight))

        for harm in HARMS:
            budgets = np.array(coverages[harm])
            row = {"arms": label, "n_arms": len(arms), "harm": harm,
                   "repeats": args.repeats,
                   "aurc": round(float(np.mean(aurcs[harm])), 4)}
            for position, budget in enumerate(FALSE_ACCEPT_BUDGETS):
                row[f"coverage_at_{budget:.0%}"] = round(float(budgets[:, position].mean()), 4)
                row[f"coverage_at_{budget:.0%}_sd"] = round(float(budgets[:, position].std()), 4)
            rows.append(row)
            table[label, harm] = [row["coverage_at_1%"], row["coverage_at_1%_sd"]]
        for name, draws in sorted(weight_draws.items(), key=lambda kv: -abs(np.mean(kv[1]))):
            weight_rows.append({"arms": label, "feature": name,
                                "weight": round(float(np.mean(draws)), 4),
                                "weight_sd": round(float(np.std(draws)), 4)})
        print(f"  {label:24}{len(weight_draws):3d} features  "
              f"class_error {table[label, 'class_error'][0]:.3f}  "
              f"background {table[label, 'background_accept'][0]:.3f}")

    print(f"\ncoverage @1% false accept, mean +- sd over {args.repeats} holdout draws\n")
    print(f"{'harm':22}" + "".join(f"{label:>24}" for label in arm_sets))
    for harm in REPORTED:
        print(f"{harm:22}" + "".join(
            f"{f'{table[label, harm][0]:.3f} +-{table[label, harm][1]:.3f}':>24}"
            for label in arm_sets))

    print(f"\nDeltas (coverage @1%), against the current two-arm B2:")
    for harm in REPORTED:
        one, two, three = (table[label, harm][0] for label in arm_sets)
        print(f"  {harm:22}drop tight {one - two:+.3f}   add regional {three - two:+.3f}")

    report_root.mkdir(parents=True, exist_ok=True)
    for data, name in ((rows, "arm_ablation.csv"), (weight_rows, "arm_ablation_weights.csv")):
        path = report_root / name
        with path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
        print(f"Wrote {path}")


def _self_check() -> None:
    # The regional stream used as an arm must collapse to plain cos(regional),
    # independent of alpha -- that is the whole premise of the third arm.
    from src.retrieval.fuse import fused_similarity

    rng = np.random.default_rng(0)
    query = rng.normal(size=(6, 8)).astype(np.float32)
    query /= np.linalg.norm(query, axis=1, keepdims=True)
    memory = rng.normal(size=(9, 8)).astype(np.float32)
    memory /= np.linalg.norm(memory, axis=1, keepdims=True)

    plain = query @ memory.T
    for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
        assert np.allclose(
            fused_similarity(query, query, memory, memory, alpha), plain, atol=1e-6
        ), alpha

    # Arm sets must be nested, or a delta between rows is not attributable.
    sets = (("local",), ("local", "local_tight"), ("local", "local_tight", "regional_k4"))
    for smaller, larger in zip(sets, sets[1:]):
        assert set(smaller) < set(larger), (smaller, larger)
    print("arm-ablation self-check OK")


if __name__ == "__main__":
    import sys

    if "--self-check" in sys.argv:
        _self_check()
    else:
        main()

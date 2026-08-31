"""Ablation: how fast does B2 degrade as the case memory's labels go wrong?

The case memory is a **positive feedback loop with no damping**. Proposals the
gate auto-accepts become verified cases; those cases are the evidence the gate
uses to auto-accept the next batch. A wrong label entering the memory therefore
does not just cost one annotation -- it becomes evidence in favour of the same
mistake being made again. Nothing in the design pushes back on that, and it has
never been tested.

This corrupts a fraction of the memory's `verified_class` labels to a different
class, refits B2 from scratch on the corrupted memory, and reports per-harm
coverage. Corruption is applied to **accepted** cases only, because those form
the positive evidence pool that `class_margin` and `agreement_at_k` read; the
verdicts themselves are untouched, so the pool keeps its size and only the
class identities rot.

Both the memory used for fitting and the memory used at query time are
corrupted, which is the realistic case: a poisoned memory is poisoned for
everything downstream, not just for evaluation.
"""

from __future__ import annotations

import argparse
import copy
import csv
from pathlib import Path

import numpy as np

from src.retrieval.phase1 import ENCODER, MAIN_SCALE, SEED, prepare, selected_alpha
from src.utils.paths import find_project_root
from scripts.run_baselines import (
    FALSE_ACCEPT_BUDGETS, HARMS, HOLDOUT_FRACTION, aurc, build_features,
    coverage_at_budget, harm_curves, stability_available,
)

NOISE_LEVELS = (0.0, 0.05, 0.10, 0.20)
REPEATS = 3
# Fraction of accepted cases that arrived by machine auto-accept rather than
# human verification. All corruption is placed inside this pool: that is what
# the feedback loop actually does, and it is the only setup in which
# provenance damping can be measured at all -- if wrong labels were spread
# uniformly over human and machine cases, no weighting on provenance could
# find them, and the experiment would only measure the weight's dead cost.
AUTO_FRACTION = 0.5
# 1.0 is the undamped control. The point is the slope across these, not which
# one is "right" -- the right value depends on a human/machine error ratio this
# project has not measured.
AUTO_WEIGHTS = (1.0, 0.75, 0.5, 0.25)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder", default=ENCODER)
    parser.add_argument("--scale", type=float, default=MAIN_SCALE)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--levels", nargs="+", type=float, default=list(NOISE_LEVELS))
    parser.add_argument("--repeats", type=int, default=REPEATS,
                        help="Seeds per level; results are averaged.")
    parser.add_argument("--auto-fraction", type=float, default=AUTO_FRACTION,
                        help="Share of accepted cases marked machine-accepted.")
    parser.add_argument("--auto-weights", nargs="+", type=float, default=list(AUTO_WEIGHTS),
                        help="Provenance weights to sweep; 1.0 is undamped.")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def slope(levels: list[float], coverages: list[float]) -> float:
    """Least-squares points of coverage lost per 1% of memory corruption.

    A single line over the whole range, not the endpoint difference: the
    endpoint is one noisy number and the question is about the trend.
    """
    fit = np.polyfit(np.array(levels) * 100.0, np.array(coverages), 1)
    return float(-fit[0])


def corrupt(memory: list[dict], rate: float, class_names: list[str], seed: int,
            auto_fraction: float = AUTO_FRACTION) -> list[dict]:
    """Relabel `rate` of the accepted cases to a different class.

    `rate` stays a fraction of the *whole* accepted pool, so the x-axis is
    unchanged and damped runs are directly comparable to the earlier undamped
    ones. What changes is placement: an `auto_fraction` slice of the accepted
    pool is marked machine-accepted, and every corrupted label is drawn from
    inside that slice. So at rate 0.20 with auto_fraction 0.5, 40% of the
    machine-accepted cases are wrong and no human-verified case is.

    That is deliberately the *favourable* case for damping. Real human labels
    contain errors too, and any of those are invisible to a provenance weight;
    the numbers this produces are therefore an upper bound on what damping buys.

    Returns a copy: the caller's memory must stay clean so the levels are
    independent rather than cumulative.
    """
    rng = np.random.default_rng(seed)
    corrupted = [dict(record) for record in memory]
    accepted = [i for i, r in enumerate(corrupted) if r["verdict"] == "accept"]

    # The auto-accepted pool exists whether or not anything is corrupted --
    # otherwise the rate-0 control would carry no damping cost and the slope
    # would start from a different model than it continues with.
    auto = set(rng.choice(accepted, size=int(round(auto_fraction * len(accepted))),
                          replace=False).tolist()) if accepted else set()
    for index in accepted:
        corrupted[index]["auto_accepted"] = index in auto
        # Proposals carry no `provenance` of their own -- `memory.build_memory`
        # derives it when the case is persisted -- so this appends to whatever
        # is there, or starts it, matching the column's `source|origin` shape.
        origin = "auto_accept" if index in auto else "human"
        existing = corrupted[index].get("provenance")
        corrupted[index]["provenance"] = f"{existing}|{origin}" if existing else origin

    wanted = int(round(rate * len(accepted)))
    if not wanted:
        return corrupted
    if wanted > len(auto):
        raise ValueError(
            f"noise rate {rate} needs {wanted} corruptible cases but the "
            f"auto-accepted pool holds {len(auto)}; raise --auto-fraction"
        )
    for index in rng.choice(sorted(auto), size=wanted, replace=False):
        true_class = corrupted[index]["verified_class"]
        pool = [name for name in class_names if name != true_class]
        corrupted[index]["verified_class"] = str(rng.choice(pool))
    return corrupted


def fit_and_score(setup, memory, val, alpha, stability, seed, auto_weight: float = 1.0):
    """Refit B2 on `memory` and score `val`. Returns per-harm coverage rows.

    The damping weight is applied at *fit* time as well as at query time. A
    model fitted on undamped features and then run on damped ones would be
    reading a differently-scaled input than it was trained for, and the drop
    would be a mismatch artifact rather than a property of the weighting.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(seed)
    images = sorted({r["image_uid"] for r in memory})
    shuffled = np.array(images)[rng.permutation(len(images))]
    holdout = set(shuffled[: int(len(shuffled) * HOLDOUT_FRACTION)].tolist())
    fit_memory = [r for r in memory if r["image_uid"] not in holdout]
    fit_queries = [r for r in memory if r["image_uid"] in holdout]

    fit_x, _ = build_features(setup, fit_queries, fit_memory, alpha,
                              stability=stability, auto_weight=auto_weight)
    fit_y = np.array([r["verdict"] == "accept" for r in fit_queries])
    val_x, _ = build_features(setup, val, memory, alpha,
                              stability=stability, auto_weight=auto_weight)

    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=SEED))
    model.fit(fit_x, fit_y)
    return model.predict_proba(val_x)[:, 1]


def main() -> None:
    args = parse_args()
    root = find_project_root(Path(__file__).resolve())
    report_root = root / "reports" / "phase1"

    alpha = args.alpha
    if alpha is None:
        alpha = selected_alpha(report_root / "alpha_sweep_summary.csv")
    print(f"Label-noise ablation: alpha={alpha} k={args.scale:g} "
          f"levels={args.levels} repeats={args.repeats}")

    setup = prepare(root, args.encoder, args.scale, limit=args.limit)
    train, val = setup.train, setup.val
    stability = stability_available(setup)
    verdicts = np.array([r["verdict"] for r in val])
    accepts = verdicts == "accept"
    n_accepted = sum(r["verdict"] == "accept" for r in train)
    print(f"memory {len(train)} cases ({n_accepted} accepted, the corruptible pool), "
          f"query {len(val)}\n")

    rows = []
    for auto_weight in args.auto_weights:
        print(f"provenance weight {auto_weight:.2f}"
              f"{' (undamped control)' if auto_weight == 1.0 else ''}:")
        for rate in args.levels:
            per_harm: dict[str, list[float]] = {harm: [] for harm in HARMS}
            aurcs: dict[str, list[float]] = {harm: [] for harm in HARMS}
            for repeat in range(args.repeats if rate else 1):
                memory = corrupt(train, rate, setup.class_names, SEED + repeat,
                                 args.auto_fraction)
                scores = fit_and_score(setup, memory, val, alpha, stability, SEED,
                                       auto_weight)
                curves = harm_curves(scores, verdicts, accepts)
                for harm, (coverage, risk) in curves.items():
                    per_harm[harm].append(coverage_at_budget(coverage, risk, 0.01))
                    aurcs[harm].append(aurc(risk))
            for harm in HARMS:
                rows.append({
                    "auto_weight": auto_weight,
                    "auto_fraction": args.auto_fraction,
                    "noise_rate": rate,
                    "n_corrupted": int(round(rate * n_accepted)),
                    "harm": harm,
                    "repeats": len(per_harm[harm]),
                    "coverage_at_1pct": round(float(np.mean(per_harm[harm])), 4),
                    "coverage_sd": round(float(np.std(per_harm[harm])), 4),
                    "aurc": round(float(np.mean(aurcs[harm])), 4),
                })
            shown = {h: np.mean(per_harm[h]) for h in ("class_error", "background_accept")}
            print(f"  {rate:.0%} noise ({int(round(rate * n_accepted)):4d} labels): "
                  f"class_error {shown['class_error']:.3f}  "
                  f"background {shown['background_accept']:.3f}")

    def cell(weight, harm, rate):
        return next(r for r in rows if r["auto_weight"] == weight
                    and r["harm"] == harm and r["noise_rate"] == rate)

    for harm in ("class_error", "background_accept"):
        print(f"\n{harm} coverage @1%, by provenance weight\n")
        print(f"{'weight':>8}" + "".join(f"{f'{r:.0%}':>10}" for r in args.levels)
              + f"{'pts/1%':>10}{'vs 1.0':>9}")
        undamped = slope(args.levels,
                         [cell(1.0, harm, r)["coverage_at_1pct"] for r in args.levels]) \
            if 1.0 in args.auto_weights else float("nan")
        for weight in args.auto_weights:
            coverages = [cell(weight, harm, r)["coverage_at_1pct"] for r in args.levels]
            gradient = slope(args.levels, coverages)
            print(f"{weight:8.2f}" + "".join(f"{c:10.3f}" for c in coverages)
                  + f"{gradient:10.3f}{gradient - undamped:+9.3f}")

    print(f"\nThe result is the slope column: coverage points lost per 1% of memory\n"
          f"corruption, damped vs undamped. Auto-accepted pool = "
          f"{args.auto_fraction:.0%} of accepted cases, all corruption inside it.")

    report_root.mkdir(parents=True, exist_ok=True)
    path = report_root / "label_noise.csv"
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {path}")


def _self_check() -> None:
    names = ["ship", "vehicle", "airplane"]
    memory = [
        {"image_uid": f"i{i}", "verdict": "accept" if i % 2 else "reject_background",
         "verified_class": "ship" if i % 2 else None}
        for i in range(100)
    ]
    clean = corrupt(memory, 0.0, names, 1)
    assert [r["verified_class"] for r in clean] == [r["verified_class"] for r in memory]

    noisy = corrupt(memory, 0.20, names, 1)
    accepted = [i for i, r in enumerate(memory) if r["verdict"] == "accept"]
    changed = [i for i in accepted if noisy[i]["verified_class"] != memory[i]["verified_class"]]
    assert len(changed) == int(round(0.20 * len(accepted))), (len(changed), len(accepted))
    # A corrupted label must be a *different* class, never the original.
    assert all(noisy[i]["verified_class"] != "ship" for i in changed)

    # Every corrupted case must be inside the auto-accepted pool, or provenance
    # weighting is being asked to find labels it cannot see and the experiment
    # measures nothing but the weight's dead cost.
    assert all(noisy[i]["auto_accepted"] for i in changed)
    auto = [i for i in accepted if noisy[i]["auto_accepted"]]
    assert len(auto) == int(round(AUTO_FRACTION * len(accepted))), len(auto)
    assert all("auto_accept" in noisy[i]["provenance"] for i in auto)
    assert all("human" in noisy[i]["provenance"] for i in accepted if i not in auto)
    # The pool must exist even at zero noise, so the control shares its model.
    assert sum(r.get("auto_accepted", False) for r in clean) == len(auto)
    # More corruption than the pool can hold is an error, not a silent clamp.
    try:
        corrupt(memory, 0.9, names, 1, auto_fraction=0.5)
    except ValueError:
        pass
    else:
        raise AssertionError("accepted a noise rate larger than the auto pool")

    # A weight of 1.0 must produce no weight vector at all, so the undamped
    # control stays bit-identical to the pre-damping runs.
    from scripts.run_baselines import case_weights
    assert case_weights(noisy, 1.0) is None
    weights = case_weights(noisy, 0.5)
    assert weights is not None and set(np.unique(weights).tolist()) == {0.5, 1.0}
    assert weights[[i for i in accepted if not noisy[i]["auto_accepted"]][0]] == 1.0
    # Non-accepted cases are untouched -- the positive pool keeps its size.
    assert all(noisy[i]["verified_class"] is None
               for i, r in enumerate(memory) if r["verdict"] != "accept")
    # The caller's memory must not be mutated, or levels would compound.
    assert all(r["verified_class"] == "ship" for r in memory if r["verdict"] == "accept")
    # Different seeds corrupt different cases.
    other = corrupt(memory, 0.20, names, 2)
    assert [r["verified_class"] for r in other] != [r["verified_class"] for r in noisy]
    print("label-noise self-check OK")


if __name__ == "__main__":
    import sys

    if "--self-check" in sys.argv:
        _self_check()
    else:
        main()

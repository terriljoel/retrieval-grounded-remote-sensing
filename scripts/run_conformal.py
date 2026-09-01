"""Conformal calibration of the auto-accept threshold, per harm type.

Turns "we picked 0.85" into "the auto-accepted error rate is at most alpha with
90% confidence". Only worth doing on the harms the gate actually handles --
background admissions and class errors -- because a guarantee at 0.5% coverage
is a guarantee about nothing.

`val` is the calibration split by the project's fixed decision, and `test` stays
untouched. To show the guarantee holds out of sample *without* spending test,
val is split in half by image: calibrate on one half, evaluate on the other.
The threshold reported for downstream use is the one calibrated on all of val,
which is what should be carried into the test-set evaluation later.

Splitting by image rather than by proposal matters: two boxes in one tile are
near-duplicates, so a proposal-level split would put a case and its twin on
opposite sides and make the held-out half look easier than it is.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from src.retrieval.conformal import DELTA, calibrate, evaluate
from src.retrieval.phase1 import ENCODER, MAIN_SCALE, SEED, prepare, selected_alpha
from src.utils.paths import find_project_root
from scripts.run_baselines import (
    HARMS, HOLDOUT_FRACTION, build_features, stability_available,
)

ALPHAS = (0.01, 0.05)
GUARANTEED_HARMS = ("background_accept", "class_error", "combined", "loose_box_accept")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder", default=ENCODER)
    parser.add_argument("--scale", type=float, default=MAIN_SCALE)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--delta", type=float, default=DELTA)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(SEED)
    root = find_project_root(Path(__file__).resolve())
    report_root = root / "reports" / "phase1"

    alpha_fusion = args.alpha
    if alpha_fusion is None:
        alpha_fusion = selected_alpha(report_root / "alpha_sweep_summary.csv")
    print(f"Conformal calibration: fusion alpha={alpha_fusion} k={args.scale:g} "
          f"delta={args.delta} (confidence {1 - args.delta:.0%})")

    setup = prepare(root, args.encoder, args.scale, limit=args.limit)
    train, val = setup.train, setup.val
    use_stability = stability_available(setup)
    if not use_stability:
        print("crop_stability: NOT included (run scripts.build_stability first)")

    # Same B2 as the baselines: fitted on a train-side holdout, never on val.
    train_images = sorted({r["image_uid"] for r in train})
    shuffled = np.array(train_images)[rng.permutation(len(train_images))]
    holdout = set(shuffled[: int(len(shuffled) * HOLDOUT_FRACTION)].tolist())
    fit_memory = [r for r in train if r["image_uid"] not in holdout]
    fit_queries = [r for r in train if r["image_uid"] in holdout]

    fit_x, _ = build_features(setup, fit_queries, fit_memory, alpha_fusion, stability=use_stability)
    fit_y = np.array([r["verdict"] == "accept" for r in fit_queries])
    val_x, _ = build_features(setup, val, train, alpha_fusion, stability=use_stability)

    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=SEED))
    model.fit(fit_x, fit_y)
    scores = model.predict_proba(val_x)[:, 1]
    verdicts = np.array([r["verdict"] for r in val])

    # Split val in half by image.
    val_images = sorted({r["image_uid"] for r in val})
    order = np.array(val_images)[rng.permutation(len(val_images))]
    calibration_images = set(order[: len(order) // 2].tolist())
    is_calibration = np.array([r["image_uid"] in calibration_images for r in val])
    print(f"\nval split by image: {is_calibration.sum()} calibration / "
          f"{(~is_calibration).sum()} evaluation proposals "
          f"({len(calibration_images)}/{len(val_images)} images)")

    rows = []
    for harm in GUARANTEED_HARMS:
        verdict = HARMS[harm]
        harmful = (verdicts != "accept") if verdict is None else (verdicts == verdict)
        for alpha in ALPHAS:
            fitted = calibrate(scores[is_calibration], harmful[is_calibration],
                               alpha, args.delta)
            held = evaluate(scores[~is_calibration], harmful[~is_calibration],
                            fitted["threshold"])
            full = calibrate(scores, harmful, alpha, args.delta)
            rows.append({
                "harm": harm, "alpha": alpha, "delta": args.delta,
                "n_harmful_in_val": int(harmful.sum()),
                "cal_threshold": round(fitted["threshold"], 6)
                if np.isfinite(fitted["threshold"]) else "inf",
                "cal_coverage": round(fitted["coverage"], 4),
                "cal_empirical_rate": round(fitted["empirical_rate"], 4),
                "cal_bounded_rate": round(fitted["bounded_rate"], 4),
                "heldout_coverage": round(held["coverage"], 4),
                "heldout_observed_rate": round(held["observed_rate"], 4),
                "heldout_n_harmful": held["n_harmful_admitted"],
                "within_budget": bool(held["observed_rate"] <= alpha),
                "full_val_threshold": round(full["threshold"], 6)
                if np.isfinite(full["threshold"]) else "inf",
                "full_val_coverage": round(full["coverage"], 4),
            })

    print(f"\nCalibrate on half of val, evaluate on the other half:\n")
    print(f"{'harm':22}{'alpha':>7}{'thr':>9}{'cal cov':>9}{'bound':>8}"
          f"{'held cov':>10}{'held rate':>11}{'ok':>5}")
    for row in rows:
        threshold = row["cal_threshold"]
        shown = f"{threshold:9.4f}" if isinstance(threshold, float) else f"{'inf':>9}"
        print(f"{row['harm']:22}{row['alpha']:7.2f}{shown}"
              f"{row['cal_coverage']:9.3f}{row['cal_bounded_rate']:8.3f}"
              f"{row['heldout_coverage']:10.3f}{row['heldout_observed_rate']:11.4f}"
              f"{'  yes' if row['within_budget'] else '   NO':>5}")

    print(f"\nThresholds calibrated on all of val -- carry these to the test set:\n")
    print(f"{'harm':22}{'alpha':>7}{'threshold':>12}{'coverage':>10}")
    for row in rows:
        threshold = row["full_val_threshold"]
        shown = f"{threshold:12.4f}" if isinstance(threshold, float) else f"{'inf':>12}"
        print(f"{row['harm']:22}{row['alpha']:7.2f}{shown}{row['full_val_coverage']:10.3f}")

    report_root.mkdir(parents=True, exist_ok=True)
    path = report_root / "conformal_thresholds.csv"
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()

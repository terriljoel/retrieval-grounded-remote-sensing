"""Phase 1 task 3: baselines B1 (kNN vote) and B2 (logistic regression).

The gate is scored as selective prediction. Every val proposal gets a
confidence that it is a true `accept`; proposals are ranked by it; at coverage
c the top c fraction is auto-accepted and the rest escalated to a human.

    risk(c) = fraction of the auto-accepted that were not true accepts
            = the false-accept rate the human never sees

AURC is the mean of risk(c) over all coverages, so lower is better. Coverage
at 1% and 5% false-accept is the headline: how much of the annotator's work
the gate can take away while keeping errors under a stated budget.

B1  kNN majority vote -- the share of the k nearest cases that were accepted.
    No training at all; this is what retrieval alone can do.
B2  Logistic regression over the feature vector, **fitted on a train-side
    holdout**. The holdout's own features are computed against a memory with
    the holdout removed, otherwise every fitting example retrieves itself at
    similarity 1.0 and the model learns that sim@1 == 1 means accept.
B0  Detector confidence -- waits for real proposals, the column is NULL here.

`random` is a floor, not a baseline: it says what AURC looks like when the
ranking carries no information, which is the only way to read whether B1's
number is good.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from src.retrieval.fuse import TOP_K, evidence, features, fused_similarity
from src.retrieval.embed import embedding_path
from src.retrieval.gate import auroc, rejection_threshold, topk_from_similarity
from src.retrieval.phase1 import ARMS, ENCODER, MAIN_SCALE, SEED, prepare, selected_alpha
from src.retrieval.stability import mean_pairwise_cosine, stream_names
from src.utils.paths import find_project_root

HOLDOUT_FRACTION = 0.25
FALSE_ACCEPT_BUDGETS = (0.01, 0.05)
RETRIEVAL_FEATURES = ("sim_at_1", "agreement_at_k", "class_margin", "evidence_margin")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder", default=ENCODER)
    parser.add_argument("--scale", type=float, default=MAIN_SCALE)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-stability", action="store_true",
                        help="Ablate the crop-stability feature out of B2.")
    return parser.parse_args()


def risk_coverage(
    scores: np.ndarray, correct: np.ndarray, seed: int = SEED
) -> tuple[np.ndarray, np.ndarray]:
    """Risk at every coverage level, ranking by `scores` descending.

    Ties are broken by a seeded shuffle, not by input order. B1 can only take
    six distinct values at k=5, so a large share of the ranking is ties, and
    the proposal list is sorted by proposal_id -- which begins with the corpus
    name. Leaving that in place would rank every background proposal ahead of
    every clean one inside each tie and report an alphabetical artifact as a
    risk curve.
    """
    jitter = np.random.default_rng(seed).permutation(len(scores))
    order = np.lexsort((jitter, -scores))
    ranked = correct[order].astype(np.float64)
    taken = np.arange(1, len(ranked) + 1)
    return taken / len(ranked), 1.0 - np.cumsum(ranked) / taken


def aurc(risk: np.ndarray) -> float:
    """Mean risk across coverage levels. Lower is better."""
    return float(risk.mean())


def coverage_at_budget(coverage: np.ndarray, risk: np.ndarray, budget: float) -> float:
    """Largest coverage whose risk is still within budget, 0 if never.

    `max`, not "first crossing": risk is not monotone in coverage -- one lucky
    correct proposal early on can dip it back under budget -- and the operator
    cares about the most work they can shed, not the first place it dips.
    """
    within = risk <= budget
    return float(coverage[within].max()) if within.any() else 0.0


HARMS = {
    # Weighting every false accept the same conflates two very different
    # outcomes. A loose box is a sloppy annotation a human can tighten later;
    # a wrong class or a box around empty ground is a corrupted label that
    # silently poisons anything trained on the exported dataset.
    "combined": None,
    "class_error": "relabel",
    "background_accept": "reject_background",
    "localisation_accept": "reject_localisation",
    # Its own column, never folded into the others and never given a relative
    # cost here -- the weighting is a stated assumption for the write-up.
    "loose_box_accept": "adjust",
}


def harm_curves(
    scores: np.ndarray, verdicts: np.ndarray, accepts: np.ndarray
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Risk-coverage per harm type, all sharing one ranking.

    The ranking is identical across harms -- same scores, same tie-break seed --
    so only the numerator changes: what counts as a harmful admission.
    """
    curves = {}
    for harm, verdict in HARMS.items():
        harmless = accepts if verdict is None else (verdicts != verdict)
        curves[harm] = risk_coverage(scores, harmless)
    return curves


LOCALISATION_RATES = (None, 0.10, 0.05, 0.02, 0.0)


def mix_sensitivity(
    scores: np.ndarray, verdicts: np.ndarray, accepts: np.ndarray,
    rates=LOCALISATION_RATES, seed: int = SEED,
) -> list[dict]:
    """Coverage as a function of how much localisation error is in the mix.

    The synthetic query set is ~18% localisation errors, all in the IoU 0.4-0.7
    band, because the jitter corpus was built to sample that band uniformly.
    A real detector's IoU distribution is bimodal, not uniform -- mAP50-95
    0.635 against mAP50 0.962 says that when it fires, the box is usually
    tight. So the headline coverage number is measured on a mix that
    over-represents the one failure mode retrieval cannot see.

    This subsamples the localisation rejects down to a range of base rates and
    reports what coverage would have been. It is a sensitivity analysis, not a
    correction: the true rate is unknown until the export lands.
    """
    rng = np.random.default_rng(seed)
    localisation = np.flatnonzero(verdicts == "reject_localisation")
    others = np.flatnonzero(verdicts != "reject_localisation")

    rows = []
    for rate in rates:
        if rate is None:
            keep = np.arange(len(scores))
        else:
            wanted = min(int(round(rate * len(others) / (1.0 - rate))), len(localisation))
            keep = np.sort(np.concatenate([
                others, rng.choice(localisation, wanted, replace=False)
            ]))
        coverage, risk = risk_coverage(scores[keep], accepts[keep])
        actual = float((verdicts[keep] == "reject_localisation").mean())
        row = {
            "localisation_rate": round(actual, 4),
            "n_queries": int(len(keep)),
            "aurc": round(aurc(risk), 4),
        }
        for budget in FALSE_ACCEPT_BUDGETS:
            row[f"coverage_at_{budget:.0%}"] = round(
                coverage_at_budget(coverage, risk, budget), 4)
        rows.append(row)
    return rows


def stability_available(setup) -> bool:
    """True when every corpus has all five view embeddings cached.

    Checked by path rather than by calling embed_stream, because a miss there
    would silently start a 31k-image GPU job inside what should be a two-minute
    script.
    """
    for corpus, entries in setup.index.items():
        for stream in ["local"] + stream_names():
            path = embedding_path(
                setup.root / "data" / "embeddings" / corpus,
                setup.encoder, stream, len(entries),
            )
            if not path.is_file():
                return False
    return True


def crop_stability(setup, records: list[dict]) -> np.ndarray:
    """Mean pairwise cosine across the five views of each record's box."""
    return mean_pairwise_cosine(
        [setup.gather(records, stream) for stream in ["local"] + stream_names()]
    )


def case_weights(memory: list[dict], auto_weight: float) -> np.ndarray | None:
    """Provenance damping weights, one per memory case.

    A case whose `provenance` marks it machine-accepted counts for
    `auto_weight`; everything human-verified counts for 1.0. Returns None when
    there is nothing to damp, so the undamped path stays bit-identical rather
    than multiplying by a vector of ones.
    """
    auto = np.array([bool(r.get("auto_accepted")) for r in memory])
    if auto_weight == 1.0 or not auto.any():
        return None
    return np.where(auto, auto_weight, 1.0)


def build_features(
    setup, queries: list[dict], memory: list[dict], alpha: float, arms=ARMS,
    stability: bool = False, auto_weight: float = 1.0,
) -> tuple[np.ndarray, list[str]]:
    """Feature matrix for `queries` retrieved against `memory`.

    Both arms go in as separate columns rather than one being chosen: they are
    complementary, not competing -- local_tight is the better localisation
    detector and local the better background detector -- so the model is left
    to weight them. (Finding 31 later showed one arm beats two at the operating
    point; the default is unchanged pending B0, deliberately.)

    `auto_weight` < 1.0 turns on provenance damping -- see `fuse.features`.
    """
    weights = case_weights(memory, auto_weight)
    positive_mask = np.array([r["verdict"] == "accept" for r in memory])
    memory_verified = np.array([r["verified_class"] or "__background__" for r in memory])
    query_predicted = np.array([r["predicted_class"] for r in queries])
    regional_memory = setup.gather(memory, setup.regional)
    regional_query = setup.gather(queries, setup.regional)

    columns: dict[str, np.ndarray] = {}
    for arm in arms:
        similarity = fused_similarity(
            setup.gather(queries, arm), regional_query,
            setup.gather(memory, arm), regional_memory, alpha,
        )
        computed = features(
            evidence(similarity, positive_mask, TOP_K),
            query_predicted, memory_verified, TOP_K, similarity, positive_mask,
            case_weights=weights,
        )
        for name in RETRIEVAL_FEATURES:
            columns[f"{name}__{arm}"] = computed[name]

    # Geometry and provenance. novelty is exactly 1 - sim_at_1, so it is
    # carried in the exported CSV for completeness but not handed to the model
    # twice under two names.
    columns["log_area"] = np.log(np.array([r["area_px"] for r in queries], dtype=np.float64) + 1.0)
    columns["upsample_factor"] = np.array([r["upsample_factor"] for r in queries], dtype=np.float64)
    columns["sensor_is_cir"] = np.array(
        [1.0 if r["sensor"] == "cir" else 0.0 for r in queries], dtype=np.float64
    )
    if stability:
        # The only localisation signal available without the detector.
        columns["crop_stability"] = crop_stability(setup, queries)
    # detector_confidence stays out until the export lands; it is NULL for
    # every synthetic proposal and a constant column would just be dropped.
    names = sorted(columns)
    return np.column_stack([columns[name] for name in names]), names


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(SEED)
    root = find_project_root(Path(__file__).resolve())
    report_root = root / "reports" / "phase1"

    alpha = args.alpha
    if alpha is None:
        alpha = selected_alpha(report_root / "alpha_sweep_summary.csv")
    print(f"Baselines: alpha={alpha} k={args.scale:g} encoder={args.encoder}")

    setup = prepare(root, args.encoder, args.scale, limit=args.limit)
    train, val = setup.train, setup.val

    # --- train-side holdout, split by image so nothing leaks --------------
    train_images = sorted({r["image_uid"] for r in train})
    shuffled = np.array(train_images)[rng.permutation(len(train_images))]
    holdout_images = set(shuffled[: int(len(shuffled) * HOLDOUT_FRACTION)].tolist())
    fit_memory = [r for r in train if r["image_uid"] not in holdout_images]
    fit_queries = [r for r in train if r["image_uid"] in holdout_images]
    print(f"\nTrain-side holdout: {len(fit_queries)} fitting proposals from "
          f"{len(holdout_images)} images, against {len(fit_memory)} memory cases "
          f"from {len(train_images) - len(holdout_images)} images")
    assert not ({r["image_uid"] for r in fit_memory} & holdout_images)

    use_stability = stability_available(setup) and not args.no_stability
    if not use_stability:
        print("crop_stability: NOT included (run scripts.build_stability first)")
    fit_x, names = build_features(setup, fit_queries, fit_memory, alpha, stability=use_stability)
    fit_y = np.array([r["verdict"] == "accept" for r in fit_queries])
    val_x, _ = build_features(setup, val, train, alpha, stability=use_stability)
    val_y = np.array([r["verdict"] == "accept" for r in val])
    verdicts = np.array([r["verdict"] for r in val])
    print(f"Feature vector ({len(names)}): {', '.join(names)}")

    stability_rows = []
    if use_stability:
        # Standalone value of the feature, before any model sees it.
        score = crop_stability(setup, val)
        accepts = verdicts == "accept"
        print(f"\ncrop_stability: accept mean {score[accepts].mean():.4f}")
        for reject in ("reject_localisation", "reject_background", "relabel"):
            selected = verdicts == reject
            area = auroc(score[accepts], score[selected])
            threshold, cost = rejection_threshold(score[accepts], score[selected], 0.95)
            stability_rows.append({
                "vs": reject, "n": int(selected.sum()),
                "mean_accept": round(float(score[accepts].mean()), 4),
                "mean_reject": round(float(score[selected].mean()), 4),
                "auroc": round(area, 4),
                "threshold_at_95": round(float(threshold), 4),
                "accept_cost_at_95": round(float(cost), 4),
            })
            print(f"  vs {reject:22} mean {score[selected].mean():.4f}  "
                  f"AUROC {area:.4f}  cost@95% {cost:.3f}")

        # Against IoU, not against verdict. The hypothesis is that stability
        # degrades smoothly as the box gets worse, so the informative view is
        # a sweep across IoU -- which now spans 0.05-0.7, because `adjust` and
        # `reject_localisation` between them cover the whole range.
        ious = np.array([r["iou"] for r in val], dtype=np.float64)
        box_error = np.isin(verdicts, ("adjust", "reject_localisation"))
        bands = ((0.05, 0.15), (0.15, 0.25), (0.25, 0.40),
                 (0.40, 0.50), (0.50, 0.60), (0.60, 0.70))
        print(f"\n  by IoU band (vs accept):")
        print(f"  {'band':>12}{'n':>6}{'mean':>9}{'AUROC':>9}")
        for low, high in bands:
            selected = box_error & (ious >= low) & (ious < high)
            if selected.sum() < 5:
                continue
            area = auroc(score[accepts], score[selected])
            stability_rows.append({
                "vs": f"iou_{low:.2f}_{high:.2f}", "n": int(selected.sum()),
                "mean_accept": round(float(score[accepts].mean()), 4),
                "mean_reject": round(float(score[selected].mean()), 4),
                "auroc": round(area, 4),
                "threshold_at_95": "", "accept_cost_at_95": "",
            })
            print(f"  {f'{low:.2f}-{high:.2f}':>12}{int(selected.sum()):6d}"
                  f"{score[selected].mean():9.4f}{area:9.4f}")
    print(f"Val: {val_y.sum()} accepts / {(~val_y).sum()} rejects")

    # --- B1: kNN majority vote -------------------------------------------
    # No fitting, so it runs straight against the full train memory.
    regional_memory = setup.gather(train, setup.regional)
    similarity = fused_similarity(
        setup.gather(val, "local"), setup.gather(val, setup.regional),
        setup.gather(train, "local"), regional_memory, alpha,
    )
    neighbours, _ = topk_from_similarity(similarity, None, TOP_K)
    accepted = np.array([r["verdict"] == "accept" for r in train])
    b1_scores = accepted[neighbours].mean(axis=1)

    # --- B2: logistic regression on the holdout --------------------------
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, random_state=SEED),
    )
    model.fit(fit_x, fit_y)
    b2_scores = model.predict_proba(val_x)[:, 1]

    floor = rng.random(len(val_y))

    # --- risk-coverage, per harm type -------------------------------------
    scored = {"B1_knn_vote": b1_scores, "B2_logreg": b2_scores, "random_floor": floor}
    summary, points, harm_rows = [], [], []
    for label, scores in scored.items():
        curves = harm_curves(scores, verdicts, val_y)
        coverage, risk = curves["combined"]
        row = {
            "baseline": label,
            "aurc": round(aurc(risk), 4),
            "risk_at_full_coverage": round(float(risk[-1]), 4),
        }
        for budget in FALSE_ACCEPT_BUDGETS:
            row[f"coverage_at_{budget:.0%}_fa"] = round(
                coverage_at_budget(coverage, risk, budget), 4
            )
        summary.append(row)

        for harm, (harm_coverage, harm_risk) in curves.items():
            harm_row = {
                "baseline": label, "harm": harm,
                "n_harmful": int((verdicts == HARMS[harm]).sum()
                                 if HARMS[harm] else (~val_y).sum()),
                "aurc": round(aurc(harm_risk), 4),
            }
            for budget in FALSE_ACCEPT_BUDGETS:
                harm_row[f"coverage_at_{budget:.0%}"] = round(
                    coverage_at_budget(harm_coverage, harm_risk, budget), 4
                )
            harm_rows.append(harm_row)
            # Thin the curve for plotting; the full arrays are 1015 points each.
            for position in range(0, len(harm_coverage), max(1, len(harm_coverage) // 200)):
                points.append({
                    "baseline": label, "harm": harm,
                    "coverage": round(float(harm_coverage[position]), 4),
                    "risk": round(float(harm_risk[position]), 4),
                })

    print(f"\n{'baseline':16}{'AURC':>8}{'risk@1.0':>10}"
          + "".join(f"{f'cov@{b:.0%}':>10}" for b in FALSE_ACCEPT_BUDGETS))
    for row in summary:
        print(f"{row['baseline']:16}{row['aurc']:8.4f}{row['risk_at_full_coverage']:10.4f}"
              + "".join(f"{row[f'coverage_at_{b:.0%}_fa']:10.3f}"
                        for b in FALSE_ACCEPT_BUDGETS))

    print(f"\nCoverage by harm type -- all four share one ranking, only what counts\n"
          f"as a harmful admission changes:\n")
    print(f"{'baseline':16}{'harm':22}{'n':>5}{'AURC':>9}"
          + "".join(f"{f'cov@{b:.0%}':>10}" for b in FALSE_ACCEPT_BUDGETS))
    for row in harm_rows:
        print(f"{row['baseline']:16}{row['harm']:22}{row['n_harmful']:5d}{row['aurc']:9.4f}"
              + "".join(f"{row[f'coverage_at_{b:.0%}']:10.3f}"
                        for b in FALSE_ACCEPT_BUDGETS))

    # --- oracle ablation: which failure mode is the bottleneck? -----------
    # Each row deletes one reject type from the query set entirely and rescores
    # B2 on what is left. This is an ORACLE, not a simulated feature: it asks
    # "what is the ceiling if this failure mode were solved perfectly", which
    # is an upper bound, not a prediction of what any real signal would buy.
    ablation = []
    for dropped in ("reject_localisation", "reject_background", "relabel"):
        keep = verdicts != dropped
        coverage, risk = risk_coverage(b2_scores[keep], val_y[keep])
        row = {"removed": dropped, "n_remaining": int(keep.sum()),
               "aurc": round(aurc(risk), 4)}
        for budget in FALSE_ACCEPT_BUDGETS:
            row[f"coverage_at_{budget:.0%}_fa"] = round(
                coverage_at_budget(coverage, risk, budget), 4)
        ablation.append(row)

    print(f"\nB2 oracle ablation -- remove one failure mode entirely (upper bound):")
    print(f"{'removed':24}{'n':>6}{'AURC':>8}"
          + "".join(f"{f'cov@{b:.0%}':>10}" for b in FALSE_ACCEPT_BUDGETS))
    for row in ablation:
        print(f"{row['removed']:24}{row['n_remaining']:6d}{row['aurc']:8.4f}"
              + "".join(f"{row[f'coverage_at_{b:.0%}_fa']:10.3f}"
                        for b in FALSE_ACCEPT_BUDGETS))

    mix = mix_sensitivity(b2_scores, verdicts, val_y)
    print(f"\nB2 sensitivity to the localisation base rate "
          f"(subsampling reject_localisation):")
    print(f"{'loc rate':>10}{'n':>7}{'AURC':>9}"
          + "".join(f"{f'cov@{b:.0%}':>10}" for b in FALSE_ACCEPT_BUDGETS))
    for row in mix:
        print(f"{row['localisation_rate']:10.1%}{row['n_queries']:7d}{row['aurc']:9.4f}"
              + "".join(f"{row[f'coverage_at_{b:.0%}']:10.3f}"
                        for b in FALSE_ACCEPT_BUDGETS))

    order = np.argsort(-b2_scores, kind="stable")
    first_error = int(np.argmax(~val_y[order])) + 1
    print(f"First non-accept in the B2 ranking: rank {first_error} "
          f"({verdicts[order][first_error - 1]})")

    weights = model.named_steps["logisticregression"].coef_[0]
    print("\nB2 weights (standardised, so comparable):")
    for name, weight in sorted(zip(names, weights), key=lambda pair: -abs(pair[1])):
        print(f"  {name:32}{weight:+8.3f}")

    report_root.mkdir(parents=True, exist_ok=True)
    for rows, filename in (
        (summary, "baseline_summary.csv"),
        (harm_rows, "coverage_by_harm.csv"),
        (stability_rows or [{"vs": "not_built", "n": 0}], "crop_stability.csv"),
        (mix, "mix_sensitivity.csv"),
        (ablation, "b2_oracle_ablation.csv"),
        (points, "risk_coverage.csv"),
        ([{"feature": n, "weight": round(float(w), 4)} for n, w in zip(names, weights)],
         "b2_weights.csv"),
    ):
        with (report_root / filename).open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(f"\nWrote {report_root}")


def _self_check() -> None:
    # A perfect ranking: every accept scored above every reject.
    correct = np.array([True] * 5 + [False] * 5)
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0])
    coverage, risk = risk_coverage(scores, correct)
    assert risk[:5].max() == 0.0, risk
    assert abs(risk[-1] - 0.5) < 1e-12, risk[-1]
    # At 0% budget it can cover exactly the half that is right.
    assert abs(coverage_at_budget(coverage, risk, 0.0) - 0.5) < 1e-12

    # A reversed ranking must be strictly worse on AURC than a perfect one.
    _, bad = risk_coverage(-scores, correct)
    assert aurc(bad) > aurc(risk), (aurc(bad), aurc(risk))

    # An all-ties score must not inherit the input ordering: with accepts
    # listed first, input order would look perfect. Seeded shuffle prevents it.
    flat = np.zeros(10)
    _, tied = risk_coverage(flat, correct)
    assert tied[:5].max() > 0.0, "tie-breaking leaked the input order"

    # An unreachable budget yields zero coverage, not a crash.
    assert coverage_at_budget(coverage, risk, -1.0) == 0.0
    print("baselines self-check OK")


if __name__ == "__main__":
    import sys

    if "--self-check" in sys.argv:
        _self_check()
    else:
        main()

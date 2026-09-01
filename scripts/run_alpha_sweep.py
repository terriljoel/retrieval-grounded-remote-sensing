"""Phase 1, tasks 1-2: build the case memory, then sweep alpha at k=4.

Memory is the train split, queries are the val split, test is untouched. Both
sides are synthetic proposals (see src/retrieval/proposals.py) so this runs
without the detector export; swap that module's output when the export lands.

The headline is the alpha curve at the fixed regional scale k=4, run twice:
once on the `local` stream and once on `local_tight`, the control that removes
what squaring drags in. If the two arms agree, the surroundings inside the
local crop were not doing the work.

k in {2, 8} is a sensitivity row at the selected alpha only, not a third sweep
axis -- the full k x alpha x verdict x arm cross-product costs an extra ~15k
ViT-L/14 forward passes to answer a question nobody asked. If the selected
alpha is 1.0 the regional stream is unused entirely and the k row is skipped,
because there is nothing for k to change.

**Alpha is selected on `bg95_accept_cost`, never on AUROC.** AUROC integrates
over every threshold; the gate runs at exactly one. The two disagree here and
the operating point is the one that decides how many correct proposals get
needlessly escalated. Same rule applies to k and to every later comparison.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from src.retrieval.fuse import TOP_K, evidence, features, fused_similarity
from src.retrieval.gate import auroc, rejection_threshold
from src.retrieval.memory import build_memory, decision_mask, load_memory, stream_matrix
from src.retrieval.phase1 import prepare
from src.utils.paths import find_project_root

SEED = 42
ENCODER = "remoteclip_l14"
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
ARMS = ("local", "local_tight")
MAIN_SCALE = 4.0
SENSITIVITY_SCALES = (2.0, 8.0)
REJECT_VERDICTS = ("reject_background", "reject_localisation", "relabel", "adjust")
# Which failure mode each feature is expected to catch. The full
# feature x reject-type matrix is written out; this is only the summary pick.
TARGET_FEATURE = {
    "reject_background": "sim_at_1",
    "reject_localisation": "sim_at_1",
    "relabel": "class_margin",
    "adjust": "sim_at_1",
}
FEATURES = ("sim_at_1", "agreement_at_k", "class_margin", "evidence_margin")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder", default=ENCODER)
    parser.add_argument("--alphas", nargs="+", type=float, default=list(ALPHAS))
    parser.add_argument("--scale", type=float, default=MAIN_SCALE)
    parser.add_argument("--sensitivity-scales", nargs="*", type=float,
                        default=list(SENSITIVITY_SCALES))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="Smoke run: N images.")
    return parser.parse_args()


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    np.random.seed(SEED)
    regional_main = f"regional_k{args.scale:g}"

    root = find_project_root(Path(__file__).resolve())
    memory_root = root / "data" / "memory"
    report_root = root / "reports" / "phase1"

    # --- corpora, embeddings and proposals ---------------------------------
    # All of it comes from src.retrieval.phase1 so the sweep, the open-set test
    # and the baselines cannot drift apart on the seed, the IoU thresholds or
    # the memory/query split. prepare() is cheap to call twice: everything it
    # touches is cached on disk.
    setups: dict[float, object] = {}

    def setup_at(scale: float):
        if scale not in setups:
            setups[scale] = prepare(root, args.encoder, scale, args.force, args.limit)
        return setups[scale]

    setup = setup_at(args.scale)
    clean = setup.index["clean"]
    class_names = setup.class_names
    confusable = setup.confusable
    print("Relabel target distribution (sampled per proposal, not argmax):")
    for name in class_names:
        targets, weights = confusable[name]
        top = sorted(zip(weights, targets), reverse=True)[:3]
        print(f"  {name:20} -> " + ", ".join(f"{t} {w:.2f}" for w, t in top))

    proposals, train, val = setup.proposals, setup.train, setup.val
    print(f"\nProposals: {len(proposals)}  memory(train) {len(train)}  query(val) {len(val)}")
    print(f"  {'verdict':22}{'memory':>8}{'query':>8}")
    for verdict in sorted({record["verdict"] for record in proposals}):
        print(f"  {verdict:22}"
              f"{sum(r['verdict'] == verdict for r in train):8d}"
              f"{sum(r['verdict'] == verdict for r in val):8d}")

    # --- task 1: persist the case memory ----------------------------------
    memory_streams_built = list(ARMS) + [regional_main]
    build_memory(
        train, {stream: setup.gather(train, stream) for stream in memory_streams_built},
        memory_root / "cases.db", memory_root / "embeddings.npy", args.encoder, SEED,
    )
    memory_rows, memory_embeddings, memory_streams = load_memory(
        memory_root / "cases.db", memory_root / "embeddings.npy"
    )
    positive_mask = decision_mask(memory_rows)
    memory_verified = np.array([row["verified_class"] or "__background__" for row in memory_rows])
    print(f"\nMemory: {len(memory_rows)} cases "
          f"({positive_mask.sum()} accepted / {(~positive_mask).sum()} rejected), "
          f"embeddings {memory_embeddings.shape}, streams {memory_streams}")
    print(f"  annotators {sorted({row['annotator_id'] for row in memory_rows})}, "
          f"timestamps {memory_rows[0]['timestamp'][:10]} .. "
          f"{max(row['timestamp'] for row in memory_rows)[:10]}")

    # --- task 2: the sweep ------------------------------------------------
    query_verdicts = np.array([record["verdict"] for record in val])
    query_predicted = np.array([record["predicted_class"] for record in val])
    query_verified = np.array([record["verified_class"] or "__background__" for record in val])
    accepts = query_verdicts == "accept"

    mean_rows, auroc_rows, summary_rows = [], [], []
    computed_at: dict[tuple[str, float, float], dict] = {}

    def run(arm: str, alpha: float, scale: float) -> dict:
        regional = f"regional_k{scale:g}"
        similarity = fused_similarity(
            setup_at(scale).gather(val, arm), setup_at(scale).gather(val, regional),
            (stream_matrix(memory_embeddings, memory_streams, arm)
             if arm in memory_streams else setup_at(scale).gather(train, arm)),
            (stream_matrix(memory_embeddings, memory_streams, regional)
             if regional in memory_streams else setup_at(scale).gather(train, regional)),
            alpha,
        )
        computed = features(
            evidence(similarity, positive_mask, TOP_K),
            query_predicted, memory_verified, TOP_K, similarity, positive_mask,
        )
        computed_at[arm, alpha, scale] = computed

        for verdict in sorted(set(query_verdicts.tolist())):
            selection = query_verdicts == verdict
            for name in FEATURES:
                mean_rows.append({
                    "arm": arm, "alpha": alpha, "k": scale, "verdict": verdict,
                    "n": int(selection.sum()), "feature": name,
                    "mean": round(float(computed[name][selection].mean()), 4),
                })

        targets = []
        for verdict in REJECT_VERDICTS:
            rejects = query_verdicts == verdict
            for name in FEATURES:
                area = auroc(computed[name][accepts], computed[name][rejects])
                auroc_rows.append({
                    "arm": arm, "alpha": alpha, "k": scale,
                    "reject_type": verdict, "feature": name,
                    "n_accept": int(accepts.sum()), "n_reject": int(rejects.sum()),
                    "auroc": round(area, 4),
                })
                if name == TARGET_FEATURE[verdict]:
                    targets.append(area)

        # AUROC ranks the whole score range; the gate only ever runs at one
        # threshold. Carry the same operating point checkpoint 4 reported, so
        # an alpha that wins on AUROC but not at 95% background rejection is
        # visible as such.
        threshold, cost = rejection_threshold(
            computed["sim_at_1"][accepts],
            computed["sim_at_1"][query_verdicts == "reject_background"],
            0.95,
        )
        row = {
            "arm": arm, "alpha": alpha, "k": scale,
            "purity_at_5_on_accepts": round(
                float(computed["agreement_at_k"][accepts].mean()), 4),
            "bg95_threshold": round(float(threshold), 4),
            "bg95_accept_cost": round(float(cost), 4),
            "auroc_background": round(targets[0], 4),
            "auroc_localisation": round(targets[1], 4),
            "auroc_relabel": round(targets[2], 4),
            "auroc_adjust": round(targets[3], 4),
            "macro_auroc": round(float(np.mean(targets)), 4),
        }
        summary_rows.append(row)
        return row

    def show(rows: list[dict]) -> None:
        print(f"{'arm':12}{'alpha':>7}{'k':>4}{'P@5':>8}{'bg':>8}{'loc':>8}"
              f"{'relabel':>9}{'MACRO':>8}{'bg95cost':>10}")
        for row in rows:
            print(f"{row['arm']:12}{row['alpha']:7.2f}{row['k']:4.0f}"
                  f"{row['purity_at_5_on_accepts']:8.3f}{row['auroc_background']:8.3f}"
                  f"{row['auroc_localisation']:8.3f}{row['auroc_relabel']:9.3f}"
                  f"{row['macro_auroc']:8.3f}{row['bg95_accept_cost']:10.3f}")

    print(f"\n=== alpha sweep at k={args.scale:g} ===")
    main_rows = [run(arm, alpha, args.scale) for arm in ARMS for alpha in args.alphas]
    show(main_rows)

    # Selection is on the operating point, never on AUROC. AUROC integrates
    # over every threshold; the gate runs at exactly one. On this data the two
    # disagree: AUROC picks alpha=1.0, which costs 9.0% false rejects against
    # 0.2% at alpha=0.75. See reports/phase1/ALPHA_SWEEP.md, methodology note.
    best = min((r for r in main_rows if r["arm"] == "local"),
               key=lambda r: r["bg95_accept_cost"])
    by_auroc = max((r for r in main_rows if r["arm"] == "local"),
                   key=lambda r: r["macro_auroc"])
    print(f"\nSelected (bg95 cost): alpha={best['alpha']} "
          f"cost {best['bg95_accept_cost']:.4f}, macro AUROC {best['macro_auroc']:.3f}")
    print(f"AUROC would have picked: alpha={by_auroc['alpha']} "
          f"cost {by_auroc['bg95_accept_cost']:.4f}, macro AUROC {by_auroc['macro_auroc']:.3f}")

    # --- k sensitivity, at the winning alpha only -------------------------
    if best["alpha"] == 1.0:
        print("\nk sensitivity skipped: the selected alpha is 1.0, so the regional "
              "stream carries no weight and k cannot change the result.")
    elif args.sensitivity_scales:
        print(f"\n=== k sensitivity at alpha={best['alpha']} ===")
        sensitivity = [best]
        for scale in args.sensitivity_scales:
            setup_at(scale)         # writes the missing regional crops
            sensitivity.append(run("local", best["alpha"], scale))
        show(sensitivity)

    # --- local vs local_tight, per class ----------------------------------
    # The section 7 control: if squaring's extra surroundings were load-bearing,
    # tight crops lose purity, and the loss should land on the elongated classes.
    per_class = []
    for alpha in args.alphas:
        for name in class_names:
            selection = accepts & (query_verified == name)
            if not selection.any():
                continue
            values = {
                arm: float(computed_at[arm, alpha, args.scale]["agreement_at_k"][selection].mean())
                for arm in ARMS
            }
            fills = [r["box_fill"] for r in val
                     if r["verdict"] == "accept" and r["verified_class"] == name]
            per_class.append({
                "alpha": alpha, "class": name, "n": int(selection.sum()),
                "mean_box_fill": round(float(np.mean(fills)), 3),
                "local_p_at_5": round(values["local"], 4),
                "local_tight_p_at_5": round(values["local_tight"], 4),
                "delta": round(values["local_tight"] - values["local"], 4),
            })

    print(f"\n=== local vs local_tight per class (alpha=1.0, k={args.scale:g}) ===")
    print(f"{'class':20}{'n':>5}{'fill':>7}{'local':>8}{'tight':>8}{'delta':>8}")
    for row in sorted((r for r in per_class if r["alpha"] == 1.0), key=lambda r: r["delta"]):
        print(f"{row['class']:20}{row['n']:5d}{row['mean_box_fill']:7.2f}"
              f"{row['local_p_at_5']:8.3f}{row['local_tight_p_at_5']:8.3f}"
              f"{row['delta']:+8.3f}")

    write_csv(mean_rows, report_root / "alpha_sweep_means.csv")
    write_csv(auroc_rows, report_root / "alpha_sweep_auroc.csv")
    write_csv(summary_rows, report_root / "alpha_sweep_summary.csv")
    write_csv(per_class, report_root / "tight_vs_local_per_class.csv")
    write_csv(
        [{"class": name, "relabel_target": target, "probability": round(float(weight), 4)}
         for name in class_names
         for weight, target in sorted(zip(*confusable[name][::-1]), reverse=True)],
        report_root / "relabel_map.csv",
    )
    write_csv(
        [{"proposal_id": r["proposal_id"], "split": r["split"], "verdict": r["verdict"],
          "corpus": r["corpus"], "image_uid": r["image_uid"],
          "predicted_class": r["predicted_class"], "verified_class": r["verified_class"],
          "iou": r["iou"], "sensor": r["sensor"], "box_fill": r["box_fill"]} for r in proposals],
        report_root / "proposals.csv",
    )
    print(f"\nWrote {report_root}")


if __name__ == "__main__":
    main()

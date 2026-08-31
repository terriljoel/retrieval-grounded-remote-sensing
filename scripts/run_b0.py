"""B0: real detector proposals, replacing the synthetic query set with the export.

`src/retrieval/proposals.py` says it plainly: "When the real export lands,
swap this module's output for it -- everything downstream consumes the
proposal records, not the corpora." This script is that swap, for the query
side only. The memory stays the synthetic train corpus
(`data/memory/cases.db`, built by `phase1.prepare()`) -- a detector-vs-
verified memory is a separate, unstarted question.

Pipeline:
  1. Load the teammate's YOLO export (`detections.csv` + `inference_images.csv`,
     the exact schema `src/detection/inference.py` writes) and join every
     detection to `image_uid` via `src.retrieval.export_join` -- never
     `image_id`, which collides 150 times across positive/negative folders.
  2. Filter to val, both positive and background source images (a detector
     can and does fire on background tiles -- those are real false positives,
     not noise to exclude).
  3. Assign each detection a verdict in the same five-way taxonomy
     `proposals.py` uses (accept / relabel / adjust / reject_localisation /
     reject_background), by IoU against the image's real GT boxes. The exact
     IoU cut points (0.4, 0.7) are carried over from `perturb.IOU_LOW` /
     `IOU_HIGH`, which is a judgement call -- the original run_b0.py that
     chose these cuts did not survive the data loss, so if a stricter/looser
     cut was used before, this reproduces the corpora's own definition
     instead of guessing at a different one.
  4. Crop and embed each detection box (corpus name `detector`, streams
     `local` / `local_tight` / `regional_k4`, crop_id
     `{image_uid}_det{NNN:03d}` -- this matches the crops already on disk
     under `data/crops/detector/`, which survived the data loss and confirm
     the convention).
  5. Score B0 (raw detector confidence, no fitting) against B2 (the same
     logistic regression `run_baselines.py` fits) on the harm curves
     `run_baselines.HARMS` defines. (A B2+confidence variant was tried and
     dropped -- see the note in `main()`: the training holdout has no real
     detector_confidence to fit against.)

Needs `detections.csv` and `inference_images.csv` from the detector export;
point `--detections` at wherever the teammate's export landed (defaults to
`~/Desktop/detections.csv`, alongside `inference_images.csv`).
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image

from src.data.models import BoundingBox
from src.retrieval.crops import CLASS_NAMES, write_streams
from src.retrieval.export_join import image_uid, load_split, parse_image_id, uid_by_index
from src.retrieval.fuse import TOP_K, evidence, fused_similarity
from src.retrieval.gate import rejection_threshold, topk_from_similarity
from src.retrieval.perturb import IOU_HIGH, IOU_LOW, iou
from src.retrieval.phase1 import DATASET_ROOT, ENCODER, MAIN_SCALE, SEED, SPLIT_CSV, prepare, selected_alpha
from src.retrieval.sensor import sensor_table
from src.utils.paths import find_project_root
from scripts.run_baselines import (
    FALSE_ACCEPT_BUDGETS, HARMS, aurc, build_features, coverage_at_budget,
    harm_curves, risk_coverage, stability_available,
)

CORPUS = "detector"
# Real train-split detections, used only for refitting B2 -- kept in a
# separate corpus/embedding cache from CORPUS so registering it on `setup`
# can never invalidate the val-side embeddings already cached under CORPUS.
CORPUS_TRAIN = "detector_train"
DEFAULT_DETECTIONS = Path.home() / "Desktop" / "detections.csv"
# The dir export_predictions rglobs to build source_index -- must hold exactly
# the 800 NWPU images (both image sets) and nothing else, or the sorted
# listing the export's index was assigned from will not reproduce. HRRSD
# lives one level up under the same DATASET_ROOT, so it is not a safe default.
DEFAULT_SOURCE_DIR = "NWPU VHR-10 dataset"
ACCEPT_IOU = IOU_HIGH   # 0.7 -- box tight enough not to need dragging
ADJUST_IOU = IOU_LOW    # 0.4 -- below this the box must be redrawn, not adjusted


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detections", type=Path, default=DEFAULT_DETECTIONS)
    parser.add_argument("--images", type=Path, default=None,
                        help="inference_images.csv, if the export has one. Defaults to "
                             "the sibling of --detections; falls back to --source if absent.")
    parser.add_argument("--source", type=Path, default=None,
                        help="Source image directory to re-derive source_index from, when "
                             "inference_images.csv is not available. Defaults to "
                             f"DATASET_ROOT/{DEFAULT_SOURCE_DIR!r}.")
    parser.add_argument("--encoder", default=ENCODER)
    parser.add_argument("--scale", type=float, default=MAIN_SCALE)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--force", action="store_true", help="Re-crop/re-embed even if cached.")
    return parser.parse_args()


# --- 1-2: load the export, join to image_uid, filter to val ----------------

def load_detections(
    detections_csv: Path, images_csv: Path | None, source: Path | None = None,
) -> list[dict]:
    """Every detection row, with `image_uid` resolved from the export's own image table.

    Two routes, `export_join`'s order of preference: `images_csv`
    (`inference_images.csv`, a direct `image_id -> source_image_path` lookup)
    when it exists, else re-deriving `source_index` by rediscovering `source`
    with the export's own `sorted(rglob)` ordering. The second route is
    positional and silently wrong if `source`'s file listing has changed
    since the export ran -- there is no way to detect that from the CSV
    alone, so a stale source directory is a real risk of this route, not a
    hypothetical one.
    """
    if not detections_csv.is_file():
        raise FileNotFoundError(
            f"{detections_csv} not found. Export it with the detection pipeline's "
            f"export_predictions (src/detection/ultralytics_backend.py) -- it writes "
            f"detections.csv and inference_images.csv side by side."
        )
    with detections_csv.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    if images_csv is not None and images_csv.is_file():
        with images_csv.open(encoding="utf-8", newline="") as file:
            uid_of = {row["image_id"]: image_uid(row["source_image_path"])
                      for row in csv.DictReader(file)}
        print(f"  joined via inference_images.csv ({images_csv})")
    else:
        if source is None or not source.is_dir():
            raise FileNotFoundError(
                f"Neither an inference_images.csv nor a valid --source directory "
                f"({source}) is available -- the image_id -> image_uid join has no "
                f"route. Pass --source pointing at the exact directory the export "
                f"was run over (must contain only the images the export discovered)."
            )
        by_index = uid_by_index(source)
        uid_of = {}
        for row in rows:
            _, index, _ = parse_image_id(row["image_id"])
            if index not in by_index:
                raise ValueError(
                    f"source_index {index} from {row['image_id']!r} has no match under "
                    f"{source} ({len(by_index)} images found) -- the source directory's "
                    f"listing has likely changed since the export ran."
                )
            uid_of[row["image_id"]] = by_index[index]
        print(f"  joined via source_index ({source}, {len(by_index)} images re-discovered)")

    missing = {row["image_id"] for row in rows} - set(uid_of)
    if missing:
        raise ValueError(
            f"{len(missing)} image_id(s) in {detections_csv} have no resolved image_uid, "
            f"e.g. {sorted(missing)[:3]}"
        )
    for row in rows:
        row["image_uid"] = uid_of[row["image_id"]]
    return rows


def load_ground_truth(root: Path) -> dict[str, list[BoundingBox]]:
    """image_uid -> GT boxes, straight from the split CSV's own annotation paths.

    Background images have no annotation_path and get an empty list -- any
    detection on one is a false positive by construction, matching
    `reject_background`.
    """
    from src.data.parsers.nwpu import parse_nwpu_annotation

    dataset_root = root / DATASET_ROOT
    with (root / SPLIT_CSV).open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    boxes: dict[str, list[BoundingBox]] = {}
    for row in rows:
        if row["is_background"] == "True" or not row["annotation_path"]:
            boxes[row["image_uid"]] = []
        else:
            boxes[row["image_uid"]] = parse_nwpu_annotation(
                dataset_root / row["annotation_path"]
            )
    return boxes


def image_paths(root: Path) -> dict[str, Path]:
    dataset_root = root / DATASET_ROOT
    with (root / SPLIT_CSV).open(encoding="utf-8", newline="") as file:
        return {row["image_uid"]: dataset_root / row["image_path"]
                for row in csv.DictReader(file)}


# --- 3: verdict by IoU against real GT --------------------------------------

def verdict_for(iou_value: float, class_match: bool) -> str:
    """The same five-way taxonomy `proposals.py` uses, applied to a real box.

    accept / adjust need the right class; a wrong class at any IoU above the
    background floor is `relabel`, because a class error is the harm that
    matters, not the box tightness. Mirrors the boundary
    `proposals.py`/`perturb.py` draw between the jitter corpus (0.4-0.7,
    "adjust") and jitter_low (<0.4, "reject_localisation").
    """
    if iou_value <= 0.0:
        return "reject_background"
    if iou_value < ADJUST_IOU:
        return "reject_localisation"
    if not class_match:
        return "relabel"
    return "accept" if iou_value >= ACCEPT_IOU else "adjust"


def build_real_proposals(
    root: Path, detections: list[dict], split: dict[str, str], gt: dict[str, list[BoundingBox]],
    target_split: str = "val", corpus: str = CORPUS,
) -> list[dict]:
    rows = [row for row in detections if split.get(row["image_uid"]) == target_split]
    rows.sort(key=lambda row: (row["image_uid"], row["detection_id"]))

    proposals = []
    counters: dict[str, int] = {}
    for row in rows:
        uid = row["image_uid"]
        box = BoundingBox(
            xmin=round(float(row["xmin"])), ymin=round(float(row["ymin"])),
            xmax=round(float(row["xmax"])), ymax=round(float(row["ymax"])),
            class_id=int(row["predicted_class_id"]),
        )
        best_iou, best_class = 0.0, None
        for truth in gt.get(uid, []):
            value = iou(box, truth)
            if value > best_iou:
                best_iou, best_class = value, CLASS_NAMES[truth.class_id]

        predicted_class = row["predicted_class_name"]
        verdict = verdict_for(best_iou, best_class == predicted_class)
        index = counters.get(uid, 0)
        counters[uid] = index + 1

        crop_id = f"{uid}_det{index:03d}"
        proposals.append({
            # `{corpus}:{crop_id}` matches the convention `proposals.py` uses
            # for every synthetic corpus, so a proposal_id can always be split
            # on the first ":" to recover which data/crops/<corpus>/ directory
            # a crop lives under -- the app's similar-case display relies on this.
            "proposal_id": f"{corpus}:{crop_id}",
            "crop_id": crop_id,
            "corpus": corpus,
            "image_uid": uid,
            "split": target_split,
            "dataset": "nwpu",
            "box": box,
            "xmin": box.xmin, "ymin": box.ymin, "xmax": box.xmax, "ymax": box.ymax,
            "area_px": (box.xmax - box.xmin) * (box.ymax - box.ymin),
            "predicted_class": predicted_class,
            "verified_class": best_class,
            "detector_confidence": float(row["confidence"]),
            "verdict": verdict,
            "iou": round(best_iou, 4),
        })
    return proposals


# --- 4: crop + embed --------------------------------------------------------

def extract_detector_crops(
    root: Path, proposals: list[dict], sensors: dict[str, str],
    scale: float, force: bool,
) -> None:
    """Write crops under data/crops/<record's own corpus>/ -- never a fixed
    directory. `val` and `real_train` proposals carry different `corpus`
    values (`detector` / `detector_train`) precisely so their embedding
    caches never collide; hardcoding one output directory here would silently
    write both into the same place regardless of which was passed in."""
    paths = image_paths(root)
    by_image: dict[str, list[dict]] = {}
    for record in proposals:
        by_image.setdefault(record["image_uid"], []).append(record)

    for uid, records in sorted(by_image.items()):
        with Image.open(paths[uid]) as image:
            image = image.convert("RGB")
            for record in records:
                output_root = root / "data" / "crops" / record["corpus"]
                geometry = write_streams(
                    image, record["box"], record["crop_id"], output_root,
                    regional_scales=(scale,), force=force,
                )
                record["upsample_factor"] = geometry["upsample_factor"]
                record["box_fill"] = geometry["box_fill"]
                record["sensor"] = sensors.get(uid, "")


def main() -> None:
    args = parse_args()
    root = find_project_root(Path(__file__).resolve())
    report_root = root / "reports" / "phase1"

    images_csv = args.images or args.detections.with_name("inference_images.csv")
    if not images_csv.is_file():
        images_csv = None
    source = args.source or (root / DATASET_ROOT / DEFAULT_SOURCE_DIR)

    alpha = args.alpha
    if alpha is None:
        alpha = selected_alpha(report_root / "alpha_sweep_summary.csv")
    print(f"B0: detections={args.detections} images={images_csv or '(none, using source_index)'} "
          f"source={source} alpha={alpha} k={args.scale:g}")

    detections = load_detections(args.detections, images_csv, source)
    split = load_split(root)
    gt = load_ground_truth(root)
    val = build_real_proposals(root, detections, split, gt)
    if not val:
        raise ValueError("No detections landed on a val-split image_uid -- nothing to score.")
    print(f"{len(val)} detections on val-split images "
          f"({sum(1 for r in val if r['verdict'] == 'reject_background')} on background tiles)")

    sensors = sensor_table({uid: path for uid, path in image_paths(root).items()
                             if uid in {r["image_uid"] for r in val}})
    sensors = {uid: record["sensor"] for uid, record in sensors.items()}
    extract_detector_crops(root, val, sensors, args.scale, args.force)

    setup = prepare(root, args.encoder, args.scale)
    setup.index[CORPUS] = val
    setup._row_of.update({(CORPUS, record["crop_id"]): row for row, record in enumerate(val)})

    verdicts = np.array([r["verdict"] for r in val])
    val_y = verdicts == "accept"
    train = setup.train
    print(f"Val (real): {val_y.sum()} accept / {(~val_y).sum()} reject, by verdict: "
          + ", ".join(f"{v}={int((verdicts == v).sum())}" for v in HARMS.values() if v)
          + f", accept={int(val_y.sum())}")

    use_stability = stability_available(setup)
    if not use_stability:
        print("crop_stability: NOT included (detector corpus lacks the stability view cache)")

    # --- B0: raw detector confidence, no fitting --------------------------
    b0_scores = np.array([r["detector_confidence"] for r in val])

    # --- B2: the same fit run_baselines.py uses, applied to real queries --
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(SEED)
    train_images = sorted({r["image_uid"] for r in train})
    shuffled = np.array(train_images)[rng.permutation(len(train_images))]
    holdout_images = set(shuffled[: int(len(shuffled) * 0.25)].tolist())
    fit_memory = [r for r in train if r["image_uid"] not in holdout_images]
    fit_queries = [r for r in train if r["image_uid"] in holdout_images]

    fit_x, names = build_features(setup, fit_queries, fit_memory, alpha, stability=use_stability)
    fit_y = np.array([r["verdict"] == "accept" for r in fit_queries])
    val_x, _ = build_features(setup, val, train, alpha, stability=use_stability)

    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=SEED))
    model.fit(fit_x, fit_y)
    b2_scores = model.predict_proba(val_x)[:, 1]

    # A "B2+confidence" variant (retrieval features plus detector_confidence
    # as a 12th column) was tried and dropped: the training holdout comes
    # from proposals.py's synthetic corpus, which sets detector_confidence to
    # None by construction (no real detector ran on those images -- see
    # docs/STATE.md). Fitting on a NaN-filled column is not a real model, so
    # this stays a B0-vs-B2 comparison, not B0-vs-B2-vs-B2+conf.

    # --- B2, refit on real train-split detections --------------------------
    # Tests whether the collapse is about what B2 was trained on: fit_queries
    # here are real detector crops (this same export, filtered to train)
    # instead of proposals.py's synthetic ones. Image-level holdout, and the
    # synthetic memory drops every case from a holdout image, so a real fit
    # query can never retrieve a case built from its own tile.
    real_train = build_real_proposals(root, detections, split, gt, "train", CORPUS_TRAIN)
    if not real_train:
        raise ValueError("No detections landed on a train-split image_uid.")
    real_train_sensors = sensor_table({uid: path for uid, path in image_paths(root).items()
                                        if uid in {r["image_uid"] for r in real_train}})
    real_train_sensors = {uid: rec["sensor"] for uid, rec in real_train_sensors.items()}
    extract_detector_crops(root, real_train, real_train_sensors, args.scale, args.force)
    setup.index[CORPUS_TRAIN] = real_train
    setup._row_of.update({(CORPUS_TRAIN, r["crop_id"]): row for row, r in enumerate(real_train)})

    real_train_images = sorted({r["image_uid"] for r in real_train})
    real_shuffled = np.array(real_train_images)[rng.permutation(len(real_train_images))]
    real_holdout_images = set(real_shuffled[: int(len(real_shuffled) * 0.25)].tolist())
    real_fit_queries = [r for r in real_train if r["image_uid"] in real_holdout_images]
    real_fit_memory = [r for r in train if r["image_uid"] not in real_holdout_images]
    if not real_fit_queries:
        raise ValueError("Real train-split holdout produced no fitting queries.")
    print(f"\nReal-data refit: {len(real_fit_queries)} fitting proposals from "
          f"{len(real_holdout_images)} images, against {len(real_fit_memory)} synthetic "
          f"memory cases from the remaining {len(real_train_images) - len(real_holdout_images)} images")

    real_fit_x, real_names = build_features(
        setup, real_fit_queries, real_fit_memory, alpha, stability=use_stability
    )
    assert real_names == names, "feature column order must match the synthetic-fit run"
    real_fit_y = np.array([r["verdict"] == "accept" for r in real_fit_queries])
    model_real = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=SEED))
    model_real.fit(real_fit_x, real_fit_y)
    b2_real_scores = model_real.predict_proba(val_x)[:, 1]

    # --- B2, evidence_margin dropped (tests the transfer hypothesis) ------
    # From the earlier synthetic-fit run: evidence_margin carries a large
    # learned weight but barely separates real accept from real background
    # (0.003 vs -0.017), unlike the synthetic data it was fit on. Refitting
    # the *original* synthetic-data B2 without it isolates whether that one
    # feature explains the collapse, independent of the real-data refit above.
    drop = [name for name in names if name.startswith("evidence_margin__")]
    keep = [name not in drop for name in names]
    model_no_em = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=SEED))
    model_no_em.fit(fit_x[:, keep], fit_y)
    b2_no_em_scores = model_no_em.predict_proba(val_x[:, keep])[:, 1]
    print(f"Evidence-margin test: dropped {drop}, refit on the original synthetic fit_queries")

    scored = {
        "B0_confidence": b0_scores, "B2_logreg": b2_scores,
        "B2_real_fit": b2_real_scores, "B2_no_evidence_margin": b2_no_em_scores,
    }
    summary, harm_rows = [], []
    for label, scores in scored.items():
        curves = harm_curves(scores, verdicts, val_y)
        coverage, risk = curves["combined"]
        row = {"baseline": label, "aurc": round(aurc(risk), 4),
               "risk_at_full_coverage": round(float(risk[-1]), 4)}
        for budget in FALSE_ACCEPT_BUDGETS:
            row[f"coverage_at_{budget:.0%}_fa"] = round(coverage_at_budget(coverage, risk, budget), 4)
        summary.append(row)
        for harm, (harm_coverage, harm_risk) in curves.items():
            n_harmful = int((verdicts == HARMS[harm]).sum() if HARMS[harm] else (~val_y).sum())
            harm_row = {"baseline": label, "harm": harm, "n_harmful": n_harmful,
                        "aurc": round(aurc(harm_risk), 4)}
            for budget in FALSE_ACCEPT_BUDGETS:
                harm_row[f"coverage_at_{budget:.0%}"] = round(
                    coverage_at_budget(harm_coverage, harm_risk, budget), 4)
            harm_rows.append(harm_row)

    print(f"\n{'baseline':24}{'AURC':>8}{'risk@1.0':>10}"
          + "".join(f"{f'cov@{b:.0%}':>10}" for b in FALSE_ACCEPT_BUDGETS))
    for row in summary:
        print(f"{row['baseline']:24}{row['aurc']:8.4f}{row['risk_at_full_coverage']:10.4f}"
              + "".join(f"{row[f'coverage_at_{b:.0%}_fa']:10.3f}" for b in FALSE_ACCEPT_BUDGETS))

    print(f"\nCoverage by harm type (real B0 proposals):")
    print(f"{'baseline':24}{'harm':22}{'n':>5}{'AURC':>9}"
          + "".join(f"{f'cov@{b:.0%}':>10}" for b in FALSE_ACCEPT_BUDGETS))
    for row in harm_rows:
        print(f"{row['baseline']:24}{row['harm']:22}{row['n_harmful']:5d}{row['aurc']:9.4f}"
              + "".join(f"{row[f'coverage_at_{b:.0%}']:10.3f}" for b in FALSE_ACCEPT_BUDGETS))

    # --- threshold recalibration: does the app's fixed threshold hold? ----
    # docs/UI_TEST_CASES.md case B: a confident, correctly-localised real
    # accept scored sim@1 0.931, just under app/gate_explorer.py's
    # NOVELTY_ACCEPT_THRESHOLD (0.9436) -- so it escalated when it should
    # have auto-accepted. That threshold came from Phase 0's CLEAN GT crops
    # (GATE_REPORT.md Sec.6, background_reject_rate=0.95 on constructed
    # corpora); recompute it the same way (gate.rejection_threshold) but on
    # REAL accept vs REAL reject_background sim@1, at this project's
    # standard 1% false-accept budget (background_reject_rate=0.99).
    OLD_THRESHOLD = 0.9436
    sim_at_1_local = val_x[:, names.index("sim_at_1__local")]
    real_accept_sim = sim_at_1_local[val_y]
    real_background_sim = sim_at_1_local[verdicts == "reject_background"]
    new_threshold, new_cost = rejection_threshold(real_accept_sim, real_background_sim, 0.99)
    old_cost = float((real_accept_sim < OLD_THRESHOLD).mean())
    old_bg_leak = float((real_background_sim >= OLD_THRESHOLD).mean())
    new_bg_leak = float((real_background_sim >= new_threshold).mean())
    print(f"\nThreshold recalibration (real accept n={len(real_accept_sim)}, "
          f"real background n={len(real_background_sim)}):")
    print(f"  old (Phase 0 clean crops) threshold={OLD_THRESHOLD:.4f}  "
          f"real-accept coverage={1 - old_cost:.1%}  real-background leak={old_bg_leak:.1%}")
    print(f"  new (real data, 1% budget) threshold={new_threshold:.4f}  "
          f"real-accept coverage={1 - new_cost:.1%}  real-background leak={new_bg_leak:.1%}")
    print(f"  recalibration alone recovers real-accept coverage "
          f"{1 - old_cost:.1%} -> {1 - new_cost:.1%} "
          f"({(1 - new_cost) - (1 - old_cost):+.1%}) at a matched ~1% background leak")

    threshold_rows = [
        {"which": "old_phase0_clean", "threshold": round(OLD_THRESHOLD, 4),
         "real_accept_coverage": round(1 - old_cost, 4), "real_background_leak": round(old_bg_leak, 4)},
        {"which": "new_real_data_1pct", "threshold": round(float(new_threshold), 4),
         "real_accept_coverage": round(1 - new_cost, 4), "real_background_leak": round(new_bg_leak, 4)},
    ]

    report_root.mkdir(parents=True, exist_ok=True)
    with (report_root / "threshold_recalibration.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(threshold_rows[0]))
        writer.writeheader()
        writer.writerows(threshold_rows)
    proposal_fields = ["proposal_id", "image_uid", "predicted_class", "verified_class",
                        "detector_confidence", "verdict", "iou", "xmin", "ymin", "xmax", "ymax"]
    with (report_root / "detector_proposals.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=proposal_fields)
        writer.writeheader()
        writer.writerows({key: record[key] for key in proposal_fields} for record in val)
    with (report_root / "b0_vs_b2.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    with (report_root / "b0_vs_b2_by_harm.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(harm_rows[0]))
        writer.writeheader()
        writer.writerows(harm_rows)
    print(f"\nWrote {report_root}/detector_proposals.csv, b0_vs_b2.csv, "
          f"b0_vs_b2_by_harm.csv, threshold_recalibration.csv")


def _self_check() -> None:
    assert verdict_for(0.0, True) == "reject_background"
    assert verdict_for(0.2, True) == "reject_localisation"
    assert verdict_for(0.5, False) == "relabel"
    assert verdict_for(0.5, True) == "adjust"
    assert verdict_for(0.9, True) == "accept"
    assert verdict_for(0.9, False) == "relabel"
    print("run_b0 self-check OK")


if __name__ == "__main__":
    import sys

    if "--self-check" in sys.argv:
        _self_check()
    else:
        main()

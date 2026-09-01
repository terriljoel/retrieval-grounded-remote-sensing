"""Phase 0 retrieval gate test: do embedding neighbours share the query's class?

Ground-truth boxes only, train+val only. The test split is never touched.

Three query corpora, all retrieving against the same memory (the clean GT crops):

  clean       GT boxes. Saturated -- every encoder scores ~0.99, so this
              cannot rank encoders and is kept only as the reference row.
  jitter      one perturbed box per GT box at IoU 0.4-0.7, i.e. detector
              localisation error.
  background  object-free crops, sized from the GT area distribution, from
              the negative image set and from empty regions of positives.

Encoders are ranked on jitter and background, never on clean.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import numpy as np

from src.retrieval.crops import (
    DEFAULT_REGIONAL_SCALE,
    contact_sheet,
    extract_crops,
    load_scope,
    write_crop_index,
)
from src.retrieval.embed import embed_stream
from src.retrieval.figures import box_area_hist, umap_panel
from src.retrieval.gate import (
    TOP_K,
    auroc,
    group_mean,
    purity,
    random_baseline,
    rejection_threshold,
    retrieve,
    size_bucket,
    tertile_edges,
    upsample_bucket,
)
from src.retrieval.perturb import build_background_corpus, build_jitter_corpus
from src.retrieval.sensor import sensor_table
from src.utils.paths import find_project_root

SEED = 42
ENCODER_KEYS = ("remoteclip_b32", "remoteclip_l14", "openclip_b32", "dinov2_b14")
SPLIT_CSV = "configs/splits/nwpu_vhr10_multilabel_split_seed42_fixed.csv"
DATASET_ROOT = "docs/shared_resources/datasets/raw"

BACKGROUND_FROM_NEGATIVES = 500
BACKGROUND_FROM_POSITIVES = 500
LOW_FILL = 0.6              # below this the local crop is mostly surroundings
REDUNDANT_CORRELATION = 0.90
CORRELATION_PAIRS = 200_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoders", nargs="+", default=list(ENCODER_KEYS), choices=ENCODER_KEYS)
    parser.add_argument("--streams", nargs="+", default=["local", "regional_k4"],
                        choices=["local", "local_tight", "regional_k2", "regional_k4", "regional_k8"])
    parser.add_argument("--regional-scale", dest="regional_scales", type=float, nargs="+", default=[DEFAULT_REGIONAL_SCALE])
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Smoke run: evenly spaced sample of N images. NWPU is ordered by "
             "class, so the first N would be almost all airplanes.",
    )
    parser.add_argument("--force", action="store_true", help="Ignore crop and embedding caches.")
    parser.add_argument("--no-figures", action="store_true", help="Skip UMAP and histogram.")
    parser.add_argument("--split-csv", type=Path, default=None)
    parser.add_argument("--dataset-root", type=Path, default=None)
    return parser.parse_args()


def write_csv(rows: list[dict], path: Path) -> Path | None:
    if not rows:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def codes(values: list, lookup: dict | None = None) -> np.ndarray:
    """Stable integer codes for a list of hashables.

    Pass `lookup` to code a second list against the first one's vocabulary --
    query and database arrays must share codes or the comparison is nonsense.
    """
    lookup = lookup if lookup is not None else {v: i for i, v in enumerate(sorted(set(values)))}
    return np.array([lookup[value] for value in values])


def spaced(items: list, limit: int | None) -> list:
    if limit is None or limit >= len(items):
        return items
    step = len(items) / limit
    return [items[int(index * step)] for index in range(limit)]


def main() -> None:
    args = parse_args()
    np.random.seed(SEED)

    project_root = find_project_root(Path(__file__).resolve())
    split_csv = args.split_csv or project_root / SPLIT_CSV
    dataset_root = args.dataset_root or project_root / DATASET_ROOT
    crop_root = project_root / "data" / "crops"
    cache_root = project_root / "data" / "embeddings"
    report_root = project_root / "reports" / "gate"

    positives = spaced(load_scope(split_csv, dataset_root), args.limit)
    negatives = spaced(load_scope(split_csv, dataset_root, background=True), args.limit)
    print(f"Scope: {len(positives)} positive + {len(negatives)} negative images, train+val")

    # --- sensor -----------------------------------------------------------
    sensors_full = sensor_table({
        record["image_uid"]: record["image_path"] for record in positives + negatives
    })
    sensors = {uid: record["sensor"] for uid, record in sensors_full.items()}
    print(f"Sensor: {dict(Counter(sensors.values()))}")
    write_csv(
        [{"image_uid": uid, **record} for uid, record in sensors_full.items()],
        report_root / "image_sensor.csv",
    )

    # --- corpora ----------------------------------------------------------
    clean = extract_crops(positives, crop_root / "clean", args.regional_scales, args.force, sensors)
    skipped = [entry for entry in clean if entry["skipped_reason"]]
    clean = [entry for entry in clean if not entry["skipped_reason"]]
    print(f"clean:      {len(clean)} crops ({len(skipped)} skipped), "
          f"{sum(entry['high_upsample'] for entry in clean)} above 6x upsample")

    jitter = build_jitter_corpus(
        positives, crop_root / "jitter", sensors, SEED, args.regional_scales, args.force
    )
    print(f"jitter:     {len(jitter)} crops, "
          f"IoU mean {np.mean([e['iou'] for e in jitter]):.3f}, "
          f"bands {dict(sorted(Counter(e['iou_band'] for e in jitter).items()))}")

    gt_sizes = [(e["xmax"] - e["xmin"], e["ymax"] - e["ymin"]) for e in clean]
    background = build_background_corpus(
        negatives, positives, gt_sizes, crop_root / "background", sensors, SEED,
        BACKGROUND_FROM_NEGATIVES, BACKGROUND_FROM_POSITIVES, args.regional_scales, args.force,
    )
    print(f"background: {len(background)} crops, "
          f"{dict(sorted(Counter(e['source'] for e in background).items()))}")

    write_crop_index(clean, report_root / "crop_index.csv")
    write_crop_index(jitter, report_root / "jitter_index.csv")
    write_crop_index(background, report_root / "background_index.csv")
    contact_sheet(clean, crop_root / "clean", report_root / "contact_sheet.jpg")

    # Crop-level class x sensor. A class living entirely inside one sensor
    # cannot have its purity read as semantic.
    print(f"\n  {'class':20}{'cir':>6}{'rgb':>6}{'%cir':>7}{'box_fill':>10}{'flag':>6}")
    fill_rows = []
    for name in sorted({entry["class_name"] for entry in clean}):
        rows = [entry for entry in clean if entry["class_name"] == name]
        counts = Counter(entry["sensor"] for entry in rows)
        fill = float(np.mean([entry["box_fill"] for entry in rows]))
        flag = "LOW" if fill < LOW_FILL else ""
        print(f"  {name:20}{counts.get('cir', 0):6d}{counts.get('rgb', 0):6d}"
              f"{100 * counts.get('cir', 0) / len(rows):7.1f}{fill:10.3f}{flag:>6}")
        fill_rows.append({
            "class": name, "n": len(rows),
            "mean_box_fill": round(fill, 4),
            "median_box_fill": round(float(np.median([e["box_fill"] for e in rows])), 4),
            "min_box_fill": round(min(e["box_fill"] for e in rows), 4),
            "frac_below_0.6": round(float(np.mean([e["box_fill"] < LOW_FILL for e in rows])), 4),
            "low_fill_flag": flag == "LOW",
        })
    write_csv(fill_rows, report_root / "box_fill_by_class.csv")

    # --- shared arrays ----------------------------------------------------
    class_lookup = {name: index for index, name in
                    enumerate(sorted({entry["class_name"] for entry in clean}))}
    image_lookup = {uid: index for index, uid in
                    enumerate(sorted({record["image_uid"] for record in positives + negatives}))}

    def arrays(index: list[dict]) -> dict:
        return {
            "ids": [entry["crop_id"] for entry in index],
            "class_names": np.array([entry["class_name"] for entry in index]),
            # Background gets -1, which matches no memory class, so a background
            # query can never be scored as pure by accident.
            "class_ids": np.array([class_lookup.get(e["class_name"], -1) for e in index]),
            "image_codes": codes([entry["image_uid"] for entry in index], image_lookup),
            "sensors": np.array([entry["sensor"] for entry in index]),
        }

    memory = arrays(clean)
    memory["sensor_codes"] = codes(memory["sensors"].tolist())
    queries = {"clean": memory, "jitter": arrays(jitter), "background": arrays(background)}

    areas = np.array([entry["area_px"] for entry in clean], dtype=np.float64)
    lower, upper = tertile_edges(areas)
    strata = {
        "upsample": np.array([upsample_bucket(e["upsample_factor"]) for e in clean]),
        "size_coco": np.array([size_bucket(area) for area in areas]),
        "size_tertile": np.array([
            f"<{lower:.0f}" if a < lower else (f"{lower:.0f}-{upper:.0f}" if a < upper else f">{upper:.0f}")
            for a in areas
        ]),
    }
    print(f"\nArea tertile edges: {lower:.0f} / {upper:.0f} px^2")

    iou_bands = np.array([entry["iou_band"] for entry in jitter])
    bg_sources = np.array([entry["source"] for entry in background])

    per_class_rows, stratified_rows, sensor_rows = [], [], []
    iou_rows, novelty_rows, correlation_rows, confusion_rows = [], [], [], []

    for encoder_key in args.encoders:
        embeddings = {}
        for stream in args.streams:
            for corpus, query in queries.items():
                embeddings[corpus, stream] = embed_stream(
                    query["ids"], crop_root / corpus, cache_root / corpus,
                    encoder_key, stream, force=args.force,
                )

        # 8.1 -- does the context crop tell us anything the object crop does not?
        if {"local", "regional_k4"} <= set(args.streams):
            obj, ctx = embeddings["clean", "local"], embeddings["clean", "regional_k4"]
            paired = float(np.mean(np.sum(obj * ctx, axis=1)))
            rng = np.random.default_rng(SEED)
            rows = rng.integers(0, len(obj), CORRELATION_PAIRS)
            cols = rng.integers(0, len(obj), CORRELATION_PAIRS)
            keep = rows != cols
            obj_sim = np.sum(obj[rows[keep]] * obj[cols[keep]], axis=1)
            ctx_sim = np.sum(ctx[rows[keep]] * ctx[cols[keep]], axis=1)
            ranking_r = float(np.corrcoef(obj_sim, ctx_sim)[0, 1])
            print(f"\n[{encoder_key}] object/context: paired cosine {paired:.3f}, "
                  f"neighbour-ranking r {ranking_r:.3f}"
                  f"{'  REDUNDANT' if paired > REDUNDANT_CORRELATION else ''}")
            correlation_rows.append({
                "encoder": encoder_key,
                "mean_paired_cosine": round(paired, 4),
                "similarity_rank_correlation": round(ranking_r, 4),
                "redundant": paired > REDUNDANT_CORRELATION,
            })

        for stream in args.streams:
            db = embeddings["clean", stream]
            object_sim = None      # set by the clean pass, used by the novelty test

            for corpus in ("clean", "jitter"):
                query = queries[corpus]
                query_emb = embeddings[corpus, stream]
                neighbours, scores, eligible = retrieve(
                    query_emb, query["image_codes"], db, memory["image_codes"], TOP_K
                )
                p1, p5 = purity(neighbours, query["class_ids"], memory["class_ids"], TOP_K)
                chance = random_baseline(
                    query["image_codes"], query["class_ids"],
                    memory["image_codes"], memory["class_ids"], TOP_K, SEED,
                )
                excluded = len(memory["ids"]) - eligible

                by_class = group_mean(p5, query["class_names"])
                by_class_p1 = group_mean(p1, query["class_names"])
                by_class_chance = group_mean(chance, query["class_names"])
                macro = float(np.mean([v for v, _ in by_class.values()]))
                macro_chance = float(np.mean([v for v, _ in by_class_chance.values()]))

                print(f"\n=== {encoder_key} / {stream} / {corpus} ===\n"
                      f"  queries {len(query['ids'])} vs memory {len(memory['ids'])}   "
                      f"excluded/query mean {excluded.mean():.1f} "
                      f"(min {excluded.min()}, max {excluded.max()})\n"
                      f"  micro P@1 {p1.mean():.3f}  micro P@5 {p5.mean():.3f}  "
                      f"MACRO P@5 {macro:.3f}  chance {macro_chance:.3f}  "
                      f"lift {macro - macro_chance:+.3f}  sim@1 mean {scores[:, 0].mean():.3f}")

                if corpus == "clean":
                    # Class purity once the sensor shortcut is taken away.
                    same_sensor, _, _ = retrieve(
                        query_emb, query["image_codes"], db, memory["image_codes"], TOP_K,
                        query["sensor_codes"], memory["sensor_codes"],
                    )
                    _, p5_ss = purity(same_sensor, query["class_ids"], memory["class_ids"], TOP_K)
                    chance_ss = random_baseline(
                        query["image_codes"], query["class_ids"],
                        memory["image_codes"], memory["class_ids"], TOP_K, SEED,
                        query["sensor_codes"], memory["sensor_codes"],
                    )
                    sensor_p5 = purity(neighbours, query["sensor_codes"],
                                       memory["sensor_codes"], TOP_K)[1]
                    print(f"  SENSOR purity@5 {sensor_p5.mean():.3f}")
                    by_class_ss = group_mean(p5_ss, query["class_names"])
                    by_class_chance_ss = group_mean(chance_ss, query["class_names"])
                    sensor_rows.append({
                        "encoder": encoder_key, "stream": stream, "query_sensor": "ALL",
                        "class": "__sensor_purity@5__", "n": len(query["ids"]),
                        "p_at_5": round(float(sensor_p5.mean()), 4),
                        "random_p_at_5": "", "lift": "",
                    })
                    for sensor_value in sorted(set(query["sensors"].tolist())):
                        mask = query["sensors"] == sensor_value
                        for name in sorted(set(query["class_names"][mask].tolist())):
                            selection = mask & (query["class_names"] == name)
                            sensor_rows.append({
                                "encoder": encoder_key, "stream": stream,
                                "query_sensor": sensor_value, "class": name,
                                "n": int(selection.sum()),
                                "p_at_5": round(float(p5[selection].mean()), 4),
                                "random_p_at_5": round(float(chance[selection].mean()), 4),
                                "lift": round(float((p5 - chance)[selection].mean()), 4),
                            })
                else:
                    by_class_ss = by_class_chance_ss = None

                header = f"  {'class':20}{'n':>5}{'P@1':>7}{'P@5':>7}{'chance':>8}{'lift':>7}"
                print(header + (f"{'P@5|same-sensor':>17}" if by_class_ss else ""))
                for name in sorted(by_class, key=lambda k: -(by_class[k][0] - by_class_chance[k][0])):
                    value, count = by_class[name]
                    chance_value = by_class_chance[name][0]
                    row = {
                        "encoder": encoder_key, "stream": stream, "corpus": corpus,
                        "class": name, "n": count,
                        "p_at_1": round(by_class_p1[name][0], 4),
                        "p_at_5": round(value, 4),
                        "random_p_at_5": round(chance_value, 4),
                        "lift": round(value - chance_value, 4),
                    }
                    line = (f"  {name:20}{count:5d}{by_class_p1[name][0]:7.3f}{value:7.3f}"
                            f"{chance_value:8.3f}{value - chance_value:+7.3f}")
                    if by_class_ss:
                        row["p_at_5_same_sensor"] = round(by_class_ss[name][0], 4)
                        row["lift_same_sensor"] = round(
                            by_class_ss[name][0] - by_class_chance_ss[name][0], 4)
                        line += f"{by_class_ss[name][0]:17.3f}"
                    print(line)
                    per_class_rows.append(row)

                if corpus == "clean":
                    for label, keys in strata.items():
                        grouped, grouped_chance = group_mean(p5, keys), group_mean(chance, keys)
                        print(f"  -- P@5 by {label} --")
                        for key in sorted(grouped):
                            value, count = grouped[key]
                            print(f"     {key:16}{count:6d}{value:8.3f}"
                                  f"{grouped_chance[key][0]:8.3f}"
                                  f"{value - grouped_chance[key][0]:+8.3f}")
                            stratified_rows.append({
                                "encoder": encoder_key, "stream": stream,
                                "stratification": label, "bucket": key, "n": count,
                                "p_at_5": round(value, 4),
                                "random_p_at_5": round(grouped_chance[key][0], 4),
                                "lift": round(value - grouped_chance[key][0], 4),
                            })
                else:
                    # 2 -- where localisation error starts to bite.
                    grouped = group_mean(p5, iou_bands)
                    grouped_sim = group_mean(scores[:, 0], iou_bands)
                    print("  -- P@5 by IoU with truth --")
                    for key in sorted(grouped):
                        value, count = grouped[key]
                        print(f"     {key:16}{count:6d}{value:8.3f}"
                              f"   sim@1 {grouped_sim[key][0]:.3f}")
                        iou_rows.append({
                            "encoder": encoder_key, "stream": stream, "iou_band": key,
                            "n": count, "p_at_5": round(value, 4),
                            "mean_sim_at_1": round(grouped_sim[key][0], 4),
                        })

                # 8.5 -- which classes retrieval conflates.
                wrong = Counter()
                for query_index, row in enumerate(neighbours):
                    for neighbour in row:
                        if memory["class_ids"][neighbour] != query["class_ids"][query_index]:
                            wrong[(query["class_names"][query_index],
                                   memory["class_names"][neighbour])] += 1
                for (source, target), count in wrong.most_common(10):
                    confusion_rows.append({
                        "encoder": encoder_key, "stream": stream, "corpus": corpus,
                        "query_class": source, "neighbour_class": target, "count": count,
                        "share_of_top5": round(count / (TOP_K * len(query["ids"])), 5),
                    })

                if corpus == "clean":
                    object_sim = scores[:, 0]

            # 1 -- novelty. Background crops query the same memory.
            bg = queries["background"]
            _, bg_scores, bg_eligible = retrieve(
                embeddings["background", stream], bg["image_codes"],
                db, memory["image_codes"], TOP_K,
            )
            bg_sim = bg_scores[:, 0]
            area = auroc(object_sim, bg_sim)
            threshold, cost = rejection_threshold(object_sim, bg_sim, 0.95)
            print(f"\n  -- novelty ({encoder_key}/{stream}) --\n"
                  f"     sim@1 object     mean {object_sim.mean():.3f} "
                  f"sd {object_sim.std():.3f}  p05 {np.quantile(object_sim, 0.05):.3f}\n"
                  f"     sim@1 background mean {bg_sim.mean():.3f} "
                  f"sd {bg_sim.std():.3f}  p95 {np.quantile(bg_sim, 0.95):.3f}\n"
                  f"     AUROC {area:.3f}   threshold {threshold:.3f} rejects 95% of "
                  f"background and {100 * cost:.1f}% of real objects")
            novelty_rows.append({
                "encoder": encoder_key, "stream": stream, "source": "all",
                "n_object": len(object_sim), "n_background": len(bg_sim),
                "object_sim_mean": round(float(object_sim.mean()), 4),
                "object_sim_p05": round(float(np.quantile(object_sim, 0.05)), 4),
                "background_sim_mean": round(float(bg_sim.mean()), 4),
                "background_sim_p95": round(float(np.quantile(bg_sim, 0.95)), 4),
                "auroc": round(area, 4),
                "threshold_95pct_bg_reject": round(threshold, 4),
                "object_reject_cost": round(cost, 4),
                "mean_excluded": round(float(bg_eligible.mean()), 1),
            })
            for source in sorted(set(bg_sources.tolist())):
                subset = bg_sim[bg_sources == source]
                area = auroc(object_sim, subset)
                threshold, cost = rejection_threshold(object_sim, subset, 0.95)
                print(f"     {source:16} n={len(subset):4d}  sim@1 mean {subset.mean():.3f}  "
                      f"AUROC {area:.3f}  cost {100 * cost:.1f}%")
                novelty_rows.append({
                    "encoder": encoder_key, "stream": stream, "source": source,
                    "n_object": len(object_sim), "n_background": len(subset),
                    "object_sim_mean": round(float(object_sim.mean()), 4),
                    "object_sim_p05": round(float(np.quantile(object_sim, 0.05)), 4),
                    "background_sim_mean": round(float(subset.mean()), 4),
                    "background_sim_p95": round(float(np.quantile(subset, 0.95)), 4),
                    "auroc": round(area, 4),
                    "threshold_95pct_bg_reject": round(threshold, 4),
                    "object_reject_cost": round(cost, 4),
                    "mean_excluded": "",
                })

        if not args.no_figures and "local" in args.streams:
            umap_panel(
                embeddings["clean", "local"], memory["class_names"],
                report_root / f"umap_{encoder_key}.png",
                f"{encoder_key} -- object stream, clean GT crops", SEED,
            )

    if not args.no_figures:
        box_area_hist(clean, report_root / "box_area_hist.png")

    write_csv(per_class_rows, report_root / "purity_per_class.csv")
    write_csv(stratified_rows, report_root / "purity_by_size.csv")
    write_csv(sensor_rows, report_root / "purity_by_sensor.csv")
    write_csv(iou_rows, report_root / "purity_by_iou.csv")
    write_csv(novelty_rows, report_root / "novelty.csv")
    write_csv(correlation_rows, report_root / "obj_context_correlation.csv")
    write_csv(confusion_rows, report_root / "confusion_pairs.csv")
    print(f"\nWrote reports to {report_root}")


if __name__ == "__main__":
    main()

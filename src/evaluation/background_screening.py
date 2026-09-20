from __future__ import annotations

import csv
import json
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from PIL import Image

from src.detection.config import save_resolved_config
from src.evaluation.assistance import (
    _manifest_split_lookup,
    _read_csv,
    _resolve_image_path,
    resolve_inference_directory,
)
from src.evaluation.config import resolve_experiment_path
from src.vlm.nim import NvidiaNimClient, build_screening_prompt, create_overlapping_tiles


def load_background_screening_cases(
    inference_directory: Path,
    *,
    image_root: Path | None,
    splits: Iterable[str],
    maximum_per_label: int | None,
    seed: int,
    split_lookup: dict[Path, str] | None = None,
) -> list[dict[str, Any]]:
    rows = _read_csv(inference_directory.resolve() / "inference_images.csv")
    wanted_splits = set(splits)
    grouped = {"background": [], "missed_objects": []}
    for row in rows:
        if int(row["prediction_count"]) != 0:
            continue
        image_path = _resolve_image_path(row["source_image_path"], image_root)
        source_split = (split_lookup or {}).get(
            image_path.resolve(), row["source_split"]
        )
        if source_split not in wanted_splits:
            continue
        truth_count = int(row["ground_truth_count"])
        label = "background" if truth_count == 0 else "missed_objects"
        grouped[label].append({
            "case_id": f"zero_detection::{row['image_id']}",
            "image_id": row["image_id"],
            "image_path": image_path,
            "source_split": source_split,
            "ground_truth_count": truth_count,
            "ground_truth_label": label,
        })

    rng = random.Random(seed)
    selected = []
    for candidates in grouped.values():
        candidates.sort(key=lambda row: row["case_id"])
        rng.shuffle(candidates)
        selected.extend(
            candidates if maximum_per_label is None else candidates[:maximum_per_label]
        )
    if not selected:
        raise ValueError("No zero-detection images matched the requested splits")
    return sorted(selected, key=lambda row: row["case_id"])


def _latest_records(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    return {row["case_id"]: row for row in rows}


def write_background_screening_metrics(
    records: Iterable[dict[str, Any]], output_directory: Path
) -> list[dict[str, Any]]:
    completed = [row for row in records if row.get("run_status") == "completed"]
    groups = {
        "positive": [row for row in completed if row["ground_truth_label"] == "missed_objects"],
        "background": [row for row in completed if row["ground_truth_label"] == "background"],
        "all": completed,
    }

    def metric(name, group, predicate):
        rows = groups[group]
        numerator = sum(predicate(row) for row in rows)
        return {
            "metric": name,
            "value": numerator / len(rows) if rows else None,
            "numerator": numerator,
            "denominator": len(rows),
        }

    metrics = [
        metric("object_presence_recall", "positive", lambda r: r["assessment"]["decision"] != "likely_background"),
        metric("background_specificity", "background", lambda r: r["assessment"]["decision"] == "likely_background"),
        metric("false_clear_rate", "positive", lambda r: r["assessment"]["decision"] == "likely_background"),
        metric("human_review_rate", "all", lambda r: r["assessment"]["decision"] != "likely_background"),
    ]
    (output_directory / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    with (output_directory / "metrics.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=metrics[0])
        writer.writeheader()
        writer.writerows(metrics)
    return metrics


def run_background_screening_experiment(config: dict[str, Any], project_root: Path) -> Path:
    experiment = config["experiment"]
    output = (
        resolve_experiment_path(experiment["output_root"], project_root)
        / experiment["name"]
    ).resolve()
    output.mkdir(parents=True, exist_ok=True)
    results_path = output / "image_results.jsonl"
    save_resolved_config(config, output / "resolved_config.yaml")

    cases_config = config["cases"]
    image_root = resolve_experiment_path(cases_config["image_root"], project_root)
    split_lookup = _manifest_split_lookup(cases_config, project_root)
    inference_directory = resolve_inference_directory(cases_config, project_root, split_lookup)
    cases = load_background_screening_cases(
        inference_directory,
        image_root=image_root,
        splits=cases_config["splits"],
        maximum_per_label=cases_config.get("maximum_per_label"),
        seed=int(cases_config["seed"]),
        split_lookup=split_lookup,
    )
    completed_ids = {
        case_id
        for case_id, row in _latest_records(results_path).items()
        if row.get("run_status") == "completed"
    }
    vlm = config["vlm"]
    client = NvidiaNimClient(
        model=vlm["model"],
        cache_root=resolve_experiment_path(vlm["cache_root"], project_root),
        base_url=vlm["base_url"],
        api_key_variable=vlm["api_key_variable"],
        timeout=float(vlm["timeout"]),
        requests_per_minute=float(vlm.get("requests_per_minute", 30)),
    )
    tiling = config["tiling"]
    with results_path.open("a", encoding="utf-8", buffering=1) as file:
        for index, case in enumerate(cases, start=1):
            if case["case_id"] in completed_ids:
                continue
            with Image.open(case["image_path"]) as opened:
                image = opened.convert("RGB")
            tiles = create_overlapping_tiles(
                image,
                int(tiling["tile_size"]),
                float(tiling["overlap_fraction"]),
            )
            if len(tiles) > int(tiling["maximum_tiles"]):
                raise ValueError(f"{case['image_id']} creates too many tiles: {len(tiles)}")
            started = time.perf_counter()
            assessment = client.screen_image(
                image=image, tiles=tiles, classes=config["classes"]
            )
            record = {
                **case,
                "image_path": str(case["image_path"]),
                "assessment": {
                    "decision": assessment.decision,
                    "possible_classes": list(assessment.possible_classes),
                    "confidence": assessment.confidence,
                    "observations": assessment.observations,
                    "uncertainty": assessment.uncertainty,
                    "suspicious_tiles": list(assessment.suspicious_tiles),
                    "prompt": build_screening_prompt(
                        config["classes"], [tile_id for tile_id, _ in tiles]
                    ),
                    "raw_response": assessment.raw_response,
                    "latency_seconds": time.perf_counter() - started,
                },
                "run_status": "completed",
            }
            file.write(json.dumps(record) + "\n")
            file.flush()
            print(f"[{index}/{len(cases)}] {case['case_id']}: completed")

    latest = _latest_records(results_path)
    metrics = write_background_screening_metrics(latest.values(), output)
    summary = {
        "selected_images": len(cases),
        "completed_images": len(latest),
        "ground_truth_label_counts": dict(
            Counter(row["ground_truth_label"] for row in cases)
        ),
        "undefined_metrics": [row["metric"] for row in metrics if row["value"] is None],
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return output

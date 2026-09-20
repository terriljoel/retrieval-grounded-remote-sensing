from __future__ import annotations

import csv

from PIL import Image

from src.evaluation.background_screening import (
    load_background_screening_cases,
    write_background_screening_metrics,
)


def _write_images(path, rows):
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=(
                "image_id", "source_image_path", "source_split",
                "prediction_count", "ground_truth_count",
            ),
        )
        writer.writeheader()
        writer.writerows(rows)


def test_load_screening_cases_selects_only_zero_detection_images(tmp_path):
    rows = []
    for image_id, predictions, truth in (
        ("background", 0, 0),
        ("missed", 0, 2),
        ("detected", 1, 1),
    ):
        image_path = tmp_path / f"{image_id}.jpg"
        Image.new("RGB", (64, 64), "gray").save(image_path)
        rows.append({
            "image_id": image_id,
            "source_image_path": image_path,
            "source_split": "test",
            "prediction_count": predictions,
            "ground_truth_count": truth,
        })
    _write_images(tmp_path / "inference_images.csv", rows)

    cases = load_background_screening_cases(
        tmp_path,
        image_root=None,
        splits=["test"],
        maximum_per_label=None,
        seed=42,
    )

    assert {case["image_id"] for case in cases} == {"background", "missed"}
    assert {case["ground_truth_label"] for case in cases} == {
        "background", "missed_objects"
    }


def test_metrics_leave_positive_metrics_undefined_without_positive_cases(tmp_path):
    records = [{
        "case_id": "zero_detection::background",
        "run_status": "completed",
        "ground_truth_label": "background",
        "assessment": {"decision": "likely_background"},
    }]

    metrics = write_background_screening_metrics(records, tmp_path)
    indexed = {row["metric"]: row for row in metrics}

    assert indexed["background_specificity"]["value"] == 1.0
    assert indexed["object_presence_recall"]["value"] is None
    assert indexed["false_clear_rate"]["value"] is None

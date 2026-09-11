from __future__ import annotations

import csv
import json

from PIL import Image
import pytest

import src.evaluation.assistance as assistance_module
from src.evaluation.assistance import (
    assessment_is_correct,
    load_assistance_cases,
    write_assistance_metrics,
)


def _write_csv(path, fieldnames, rows):
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_load_cases_aligns_exported_tables_and_excludes_false_negatives(tmp_path):
    image_path = tmp_path / "query.jpg"
    Image.new("RGB", (100, 80), "gray").save(image_path)
    _write_csv(
        tmp_path / "inference_images.csv",
        ("image_id", "source_image_path", "source_split"),
        [{"image_id": "image_1", "source_image_path": image_path, "source_split": "test"}],
    )
    _write_csv(
        tmp_path / "detections.csv",
        (
            "detection_id", "image_id", "predicted_class_id",
            "predicted_class_name", "confidence", "xmin", "ymin", "xmax", "ymax",
        ),
        [{
            "detection_id": "det_1", "image_id": "image_1",
            "predicted_class_id": 0, "predicted_class_name": "airplane",
            "confidence": 0.9, "xmin": 1, "ymin": 2, "xmax": 30, "ymax": 40,
        }],
    )
    _write_csv(
        tmp_path / "ground_truth.csv",
        ("ground_truth_id", "image_id", "class_id", "class_name"),
        [{"ground_truth_id": "gt_1", "image_id": "image_1", "class_id": 0, "class_name": "airplane"}],
    )
    _write_csv(
        tmp_path / "prediction_comparison.csv",
        ("comparison_id", "image_id", "detection_id", "ground_truth_id", "iou", "status"),
        [
            {"comparison_id": "cmp_1", "image_id": "image_1", "detection_id": "det_1", "ground_truth_id": "gt_1", "iou": 0.8, "status": "true_positive"},
            {"comparison_id": "cmp_2", "image_id": "image_1", "detection_id": "", "ground_truth_id": "gt_1", "iou": "", "status": "false_negative"},
        ],
    )

    cases, false_negatives = load_assistance_cases(
        tmp_path,
        image_root=None,
        splits=["test"],
        statuses=["true_positive"],
        maximum_per_status=None,
        seed=42,
    )

    assert false_negatives == 1
    assert len(cases) == 1
    assert cases[0].ground_truth_class_name == "airplane"
    assert assessment_is_correct(cases[0], {
        "decision": "accept", "suggested_class": "airplane"
    })


def test_metrics_keep_variants_separate(tmp_path):
    records = [{
        "case_id": "cmp_1",
        "run_status": "completed",
        "ground_truth_status": "false_positive",
        "ground_truth_class_name": None,
        "retrieval": {"object_evidence": [], "latency_seconds": 0.1},
        "query_only_vlm": {
            "correct": True,
            "decision": "human_review",
            "unsupported_evidence_ids": [],
            "latency_seconds": 0.2,
        },
        "retrieval_grounded_vlm": {
            "correct": False,
            "decision": "accept",
            "unsupported_evidence_ids": [],
            "latency_seconds": 0.3,
        },
        "configured_policy": {"recommendation": "human_review"},
    }]

    metrics = write_assistance_metrics(records, tmp_path)
    indexed = {(row["variant"], row["metric"]): row for row in metrics}

    assert indexed[("query_only_vlm", "verification_accuracy")]["value"] == 1.0
    assert indexed[("retrieval_grounded_vlm", "verification_accuracy")]["value"] == 0.0
    assert indexed[("retrieval_grounded_vlm", "false_accept_rate")]["value"] == 1.0
    assert json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))


def test_experiment_failure_is_written_to_local_log(monkeypatch, tmp_path):
    config = {
        "experiment": {"name": "traceable", "output_root": str(tmp_path)},
        "cases": {
            "inference_directory": "/missing/export",
            "image_root": "/missing/images",
            "splits": ["val"],
            "statuses": ["false_positive"],
            "maximum_per_status": 1,
        },
        "execution": {"variants": ["detector_only"]},
    }

    def fail(_config, _project_root):
        raise ValueError("diagnostic failure")

    monkeypatch.setattr(assistance_module, "_run_assistance_experiment", fail)
    with pytest.raises(ValueError, match="diagnostic failure"):
        assistance_module.run_assistance_experiment(config, tmp_path)

    log = (tmp_path / "traceable" / "experiment.log").read_text(encoding="utf-8")
    assert "inference_directory=/missing/export" in log
    assert "splits=['val']" in log
    assert "RUN FAILED" in log
    assert "ValueError: diagnostic failure" in log

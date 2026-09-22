from __future__ import annotations

import csv
import importlib.metadata
import json
import platform
import random
import time
import traceback
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from PIL import Image

from src.annotation.models import Box, DetectionSuggestion
from src.annotation.policy import evaluate_acceptance_policy
from src.annotation.service import retrieve_detection_evidence
from src.data.manifests import load_split_manifest
from src.detection.config import save_resolved_config
from src.evaluation.config import resolve_experiment_path
from src.retrieval.lancedb_store import LanceDbEvidenceStore, find_latest_database
from src.retrieval.models import EvidenceRecord
from src.retrieval.remoteclip import RemoteClipEncoder
from src.vlm.nim import NvidiaNimClient, VlmAssessment, build_prompt, grounded_montage


@dataclass(frozen=True)
class AssistanceCase:
    case_id: str
    image_id: str
    image_path: Path
    source_split: str
    detection: DetectionSuggestion
    ground_truth_id: str | None
    ground_truth_class_id: int | None
    ground_truth_class_name: str | None
    iou: float | None
    status: str


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Required experiment table not found: {path}")
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


INFERENCE_TABLES = (
    "inference_images.csv",
    "detections.csv",
    "ground_truth.csv",
    "prediction_comparison.csv",
)


def resolve_inference_directory(
    cases_config: dict[str, Any],
    project_root: Path,
    split_lookup: dict[Path, str] | None = None,
) -> Path:
    configured = cases_config["inference_directory"]
    if configured != "latest":
        return resolve_experiment_path(configured, project_root)

    root = resolve_experiment_path(cases_config["inference_root"], project_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Inference export root not found: {root}")
    requested_splits = set(cases_config["splits"])
    image_root = (
        resolve_experiment_path(cases_config["image_root"], project_root)
        if cases_config.get("image_root") else None
    )
    effective_lookup = split_lookup or _manifest_split_lookup(
        cases_config, project_root
    )
    compatible: list[Path] = []
    inspected: dict[str, list[str]] = {}
    for image_table in root.rglob("inference_images.csv"):
        directory = image_table.parent.resolve()
        if not all((directory / name).is_file() for name in INFERENCE_TABLES):
            continue
        image_rows = _read_csv(image_table)
        available_splits = set()
        for row in image_rows:
            effective_split = row["source_split"]
            if effective_lookup:
                try:
                    image_path = _resolve_image_path(
                        row["source_image_path"], image_root
                    )
                    effective_split = effective_lookup.get(
                        image_path.resolve(), effective_split
                    )
                except (FileNotFoundError, ValueError):
                    pass
            available_splits.add(effective_split)
        inspected[str(directory)] = sorted(available_splits)
        if requested_splits <= available_splits:
            compatible.append(directory)
    if not compatible:
        raise ValueError(
            f"No complete inference export under {root} contains requested "
            f"splits {sorted(requested_splits)}. Inspected exports: {inspected}"
        )
    return max(
        compatible,
        key=lambda directory: max(
            (directory / name).stat().st_mtime for name in INFERENCE_TABLES
        ),
    )


def _manifest_split_lookup(
    cases_config: dict[str, Any], project_root: Path
) -> dict[Path, str]:
    if not cases_config.get("manifest"):
        return {}
    splits = load_split_manifest(
        resolve_experiment_path(cases_config["manifest"], project_root),
        resolve_experiment_path(
            cases_config["manifest_dataset_root"], project_root
        ),
    )
    return {
        item.image_path.resolve(): split
        for split, items in splits.items()
        for item in items
    }


def _resolve_image_path(stored_path: str, image_root: Path | None) -> Path:
    stored = Path(stored_path).expanduser()
    if stored.is_file():
        return stored.resolve()
    if image_root is None:
        raise FileNotFoundError(f"Query image not found: {stored}")
    candidates = []
    if not stored.is_absolute():
        candidates.append(image_root / stored)
    if image_root.name in stored.parts:
        root_index = stored.parts.index(image_root.name)
        candidates.append(image_root.joinpath(*stored.parts[root_index + 1:]))
    candidates.append(image_root / stored.name)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    matches = list(image_root.rglob(stored.name))
    if len(matches) == 1:
        return matches[0].resolve()
    if not matches:
        raise FileNotFoundError(
            f"Query image {stored.name!r} was not found under {image_root}"
        )
    raise ValueError(
        f"Query image name {stored.name!r} is ambiguous under {image_root}"
    )


def load_assistance_cases(
    inference_directory: Path,
    *,
    image_root: Path | None,
    splits: Iterable[str],
    statuses: Iterable[str],
    maximum_per_status: int | None,
    seed: int,
    split_lookup: dict[Path, str] | None = None,
    minimum_detector_confidence: float = 0.0,
) -> tuple[list[AssistanceCase], int]:
    inference_directory = inference_directory.resolve()
    image_rows = _read_csv(inference_directory / "inference_images.csv")
    detection_rows = _read_csv(inference_directory / "detections.csv")
    ground_truth_rows = _read_csv(inference_directory / "ground_truth.csv")
    comparison_rows = _read_csv(inference_directory / "prediction_comparison.csv")

    images = {row["image_id"]: row for row in image_rows}
    image_paths = {
        image_id: _resolve_image_path(row["source_image_path"], image_root)
        for image_id, row in images.items()
    }
    effective_splits = {
        image_id: (split_lookup or {}).get(
            image_paths[image_id].resolve(), row["source_split"]
        )
        for image_id, row in images.items()
    }
    detections = {row["detection_id"]: row for row in detection_rows}
    ground_truths = {row["ground_truth_id"]: row for row in ground_truth_rows}
    wanted_splits = set(splits)
    wanted_statuses = set(statuses)
    false_negative_count = sum(
        row["status"] == "false_negative"
        and effective_splits[row["image_id"]] in wanted_splits
        for row in comparison_rows
    )
    grouped: dict[str, list[AssistanceCase]] = defaultdict(list)

    for comparison in comparison_rows:
        if comparison["status"] not in wanted_statuses or not comparison["detection_id"]:
            continue
        image_row = images[comparison["image_id"]]
        if effective_splits[comparison["image_id"]] not in wanted_splits:
            continue
        detection_row = detections[comparison["detection_id"]]
        if float(detection_row["confidence"]) < minimum_detector_confidence:
            continue
        ground_truth_id = comparison["ground_truth_id"] or None
        ground_truth = ground_truths.get(ground_truth_id or "")
        detection = DetectionSuggestion(
            detection_id=detection_row["detection_id"],
            class_id=int(detection_row["predicted_class_id"]),
            class_name=detection_row["predicted_class_name"],
            confidence=float(detection_row["confidence"]),
            box=Box(
                float(detection_row["xmin"]),
                float(detection_row["ymin"]),
                float(detection_row["xmax"]),
                float(detection_row["ymax"]),
            ),
        )
        grouped[comparison["status"]].append(AssistanceCase(
            case_id=comparison["comparison_id"],
            image_id=comparison["image_id"],
            image_path=image_paths[comparison["image_id"]],
            source_split=effective_splits[comparison["image_id"]],
            detection=detection,
            ground_truth_id=ground_truth_id,
            ground_truth_class_id=(
                int(ground_truth["class_id"]) if ground_truth else None
            ),
            ground_truth_class_name=(ground_truth["class_name"] if ground_truth else None),
            iou=float(comparison["iou"]) if comparison["iou"] else None,
            status=comparison["status"],
        ))

    rng = random.Random(seed)
    selected: list[AssistanceCase] = []
    for status in sorted(wanted_statuses):
        candidates = sorted(grouped[status], key=lambda item: item.case_id)
        rng.shuffle(candidates)
        if maximum_per_status is not None:
            candidates = candidates[:maximum_per_status]
        selected.extend(candidates)
    if not selected:
        split_counts = Counter(effective_splits.values())
        status_counts = Counter(row["status"] for row in comparison_rows)
        raise ValueError(
            "No eligible per-detection cases remain after applying split and "
            f"status filters at detector confidence >= "
            f"{minimum_detector_confidence:.3f}. Available source splits: "
            f"{dict(split_counts)}. "
            f"Available comparison statuses: {dict(status_counts)}."
        )
    return sorted(selected, key=lambda item: item.case_id), false_negative_count


def _evidence_dict(record: EvidenceRecord) -> dict[str, Any]:
    return {
        "embedding_id": record.embedding_id,
        "image_id": record.image_id,
        "image_key": record.image_key,
        "source_path": str(record.source_path),
        "split": record.split,
        "class_id": record.class_id,
        "class_name": record.class_name,
        "crop_type": record.crop_type,
        "box": record.box.as_list(),
        "cosine_similarity": record.cosine_similarity,
    }


def _assessment_dict(
    assessment: VlmAssessment,
    *,
    prompt: str,
    latency_seconds: float,
    supplied_evidence_count: int,
) -> dict[str, Any]:
    allowed_ids = {f"E{index}" for index in range(1, supplied_evidence_count + 1)}
    unsupported = sorted(set(assessment.evidence_ids) - allowed_ids)
    return {
        "decision": assessment.decision,
        "suggested_class": assessment.suggested_class,
        "confidence": assessment.confidence,
        "observations": assessment.observations,
        "uncertainty": assessment.uncertainty,
        "evidence_ids": list(assessment.evidence_ids),
        "unsupported_evidence_ids": unsupported,
        "raw_response": assessment.raw_response,
        "prompt": prompt,
        "latency_seconds": latency_seconds,
    }


def assessment_is_correct(case: AssistanceCase, result: dict[str, Any]) -> bool:
    decision = result.get("decision")
    suggestion = result.get("suggested_class")
    if case.status == "true_positive":
        return decision == "accept" and suggestion in {None, case.detection.class_name}
    if case.status == "false_positive":
        return decision == "human_review"
    return decision == "correct" and suggestion == case.ground_truth_class_name


def _latest_records(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    latest: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            latest[record["case_id"]] = record
    return latest


def _record_has_variants(record: dict[str, Any], variants: set[str]) -> bool:
    required_fields = {
        "detector_retrieval": "retrieval",
        "query_only_vlm": "query_only_vlm",
        "retrieval_grounded_vlm": "retrieval_grounded_vlm",
        "configured_policy": "configured_policy",
    }
    return all(
        variant == "detector_only" or required_fields[variant] in record
        for variant in variants
    )


def _metric(
    rows: list[dict[str, Any]], variant: str, name: str, predicate
) -> dict[str, Any]:
    denominator = len(rows)
    numerator = sum(bool(predicate(row)) for row in rows)
    return {
        "variant": variant,
        "metric": name,
        "value": numerator / denominator if denominator else None,
        "numerator": numerator,
        "denominator": denominator,
    }


def _mean_metric(
    rows: list[dict[str, Any]], variant: str, name: str, value
) -> dict[str, Any]:
    values = [float(value(row)) for row in rows]
    return {
        "variant": variant,
        "metric": name,
        "value": sum(values) / len(values) if values else None,
        "numerator": sum(values) if values else 0,
        "denominator": len(values),
    }


def _decision_metrics(
    rows: list[dict[str, Any]], variant: str, accepted
) -> list[dict[str, Any]]:
    accepted_rows = [row for row in rows if accepted(row)]
    incorrect_rows = [
        row for row in rows
        if row["ground_truth_status"] != "true_positive"
    ]
    return [
        _metric(
            rows,
            variant,
            "decision_accuracy",
            lambda row: accepted(row)
            == (row["ground_truth_status"] == "true_positive"),
        ),
        _metric(rows, variant, "auto_accept_coverage", accepted),
        _metric(
            accepted_rows,
            variant,
            "auto_accept_precision",
            lambda row: row["ground_truth_status"] == "true_positive",
        ),
        _metric(incorrect_rows, variant, "false_accept_rate", accepted),
    ]


def _macro_decision_f1(rows: list[dict[str, Any]], key: str) -> float | None:
    if not rows:
        return None
    expected_by_status = {
        "true_positive": "accept",
        "false_positive": "human_review",
        "class_error": "correct",
    }
    scores = []
    for label in ("accept", "human_review", "correct"):
        true_positive = sum(
            row[key]["decision"] == label
            and expected_by_status[row["ground_truth_status"]] == label
            for row in rows
        )
        false_positive = sum(
            row[key]["decision"] == label
            and expected_by_status[row["ground_truth_status"]] != label
            for row in rows
        )
        false_negative = sum(
            row[key]["decision"] != label
            and expected_by_status[row["ground_truth_status"]] == label
            for row in rows
        )
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(2 * true_positive / denominator if denominator else 0.0)
    return sum(scores) / len(scores)


def write_assistance_metrics(
    records: Iterable[dict[str, Any]], output_directory: Path
) -> list[dict[str, Any]]:
    completed = [row for row in records if row.get("run_status") == "completed"]
    matched = [row for row in completed if row.get("ground_truth_class_name")]
    policy_rows = [row for row in completed if row.get("configured_policy")]
    metrics = _decision_metrics(
        policy_rows,
        "detector_only",
        lambda row: row["configured_policy"]["detector_passed"],
    )
    metrics.extend(_decision_metrics(
        policy_rows,
        "detector_retrieval",
        lambda row: (
            row["configured_policy"]["detector_passed"]
            and row["configured_policy"]["retrieval_passed"]
        ),
    ))
    retrieval_rows = [row for row in matched if row.get("retrieval")]
    metrics.append(_metric(
        retrieval_rows,
        "detector_retrieval",
        "top1_class_accuracy",
        lambda row: row["retrieval"]["object_evidence"]
        and row["retrieval"]["object_evidence"][0]["class_name"]
        == row["ground_truth_class_name"],
    ))
    metrics.append(_mean_metric(
        retrieval_rows,
        "detector_retrieval",
        "precision_at_k",
        lambda row: sum(
            item["class_name"] == row["ground_truth_class_name"]
            for item in row["retrieval"]["object_evidence"]
        ) / len(row["retrieval"]["object_evidence"])
        if row["retrieval"]["object_evidence"] else 0.0,
    ))
    metrics.append(_metric(
        retrieval_rows,
        "detector_retrieval",
        "hit_at_k",
        lambda row: any(
            item["class_name"] == row["ground_truth_class_name"]
            for item in row["retrieval"]["object_evidence"]
        ),
    ))
    retrieval_all = [row for row in completed if row.get("retrieval")]
    metrics.append(_mean_metric(
        retrieval_all,
        "detector_retrieval",
        "mean_latency_seconds",
        lambda row: row["retrieval"]["latency_seconds"],
    ))
    for key, variant in (
        ("query_only_vlm", "query_only_vlm"),
        ("retrieval_grounded_vlm", "retrieval_grounded_vlm"),
    ):
        rows = [row for row in completed if row.get(key)]
        metrics.extend(_decision_metrics(
            rows,
            variant,
            lambda row, k=key: row[k]["decision"] == "accept",
        ))
        metrics.append(_metric(
            rows, variant, "verification_accuracy", lambda row, k=key: row[k]["correct"]
        ))
        metrics.append({
            "variant": variant,
            "metric": "macro_decision_f1",
            "value": _macro_decision_f1(rows, key),
            "numerator": "",
            "denominator": len(rows),
        })
        class_errors = [
            row for row in rows if row["ground_truth_status"] == "class_error"
        ]
        metrics.append(_metric(
            class_errors,
            variant,
            "class_correction_accuracy",
            lambda row, k=key: row[k]["correct"],
        ))
        metrics.append(_metric(
            rows,
            variant,
            "unsupported_evidence_reference_rate",
            lambda row, k=key: bool(row[k]["unsupported_evidence_ids"]),
        ))
        metrics.append(_mean_metric(
            rows,
            variant,
            "mean_latency_seconds",
            lambda row, k=key: row[k]["latency_seconds"],
        ))
    metrics.extend(_decision_metrics(
        policy_rows,
        "configured_policy",
        lambda row: row["configured_policy"]["recommendation"] == "accept",
    ))

    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    with (output_directory / "metrics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.DictWriter(
            file, fieldnames=("variant", "metric", "value", "numerator", "denominator")
        )
        writer.writeheader()
        writer.writerows(metrics)
    return metrics


def _database_path(config: dict[str, Any], project_root: Path) -> Path:
    retrieval = config["retrieval"]
    if retrieval.get("database_path"):
        return resolve_experiment_path(retrieval["database_path"], project_root)
    artifact_root = resolve_experiment_path(retrieval["artifact_root"], project_root)
    if retrieval.get("artifact", "latest") != "latest":
        return (artifact_root / retrieval["artifact"] / "lancedb").resolve()
    return find_latest_database(artifact_root)


def experiment_output_directory(
    config: dict[str, Any], project_root: Path
) -> Path:
    experiment = config["experiment"]
    return (
        resolve_experiment_path(experiment["output_root"], project_root.resolve())
        / experiment["name"]
    ).resolve()


def _append_experiment_log(output_directory: Path, message: str) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat()
    with (output_directory / "experiment.log").open(
        "a", encoding="utf-8", buffering=1
    ) as log:
        log.write(f"[{timestamp}] {message.rstrip()}\n")


def _run_assistance_experiment(
    config: dict[str, Any], project_root: Path
) -> Path:
    project_root = project_root.resolve()
    output_directory = experiment_output_directory(config, project_root)
    output_directory.mkdir(parents=True, exist_ok=True)
    results_path = output_directory / "case_results.jsonl"
    execution = config["execution"]
    resume = bool(execution.get("resume", True))
    if results_path.is_file() and not resume:
        raise FileExistsError(
            f"Experiment results already exist and resume=false: {results_path}"
        )
    save_resolved_config(config, output_directory / "resolved_config.yaml")
    package_versions = {}
    for package in ("lancedb", "open-clip-torch", "Pillow", "torch"):
        try:
            package_versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            package_versions[package] = None
    (output_directory / "runtime.json").write_text(
        json.dumps({
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": package_versions,
        }, indent=2) + "\n",
        encoding="utf-8",
    )

    cases_config = config["cases"]
    image_root = (
        resolve_experiment_path(cases_config["image_root"], project_root)
        if cases_config.get("image_root") else None
    )
    split_lookup = _manifest_split_lookup(cases_config, project_root)
    inference_directory = resolve_inference_directory(
        cases_config, project_root, split_lookup
    )
    _append_experiment_log(
        output_directory, f"resolved_inference_directory={inference_directory}"
    )
    cases, excluded_false_negatives = load_assistance_cases(
        inference_directory,
        image_root=image_root,
        splits=cases_config["splits"],
        statuses=cases_config["statuses"],
        maximum_per_status=(
            int(cases_config["maximum_per_status"])
            if cases_config.get("maximum_per_status") is not None else None
        ),
        seed=int(cases_config["seed"]),
        split_lookup=split_lookup,
        minimum_detector_confidence=float(
            cases_config.get("minimum_detector_confidence", 0.0)
        ),
    )
    existing = _latest_records(results_path)
    completed_ids = {
        case_id for case_id, row in existing.items()
        if row.get("run_status") == "completed"
        and _record_has_variants(row, set(execution["variants"]))
    }

    variants = set(execution["variants"])
    needs_retrieval = bool(variants - {"detector_only", "query_only_vlm"})
    embedding_config = config["embedding"]
    retrieval_config = config["retrieval"]
    encoder = store = None
    if needs_retrieval:
        encoder = RemoteClipEncoder(
            model_name=embedding_config["model_name"],
            repository=embedding_config["repository"],
            filename=embedding_config["filename"],
            cache_dir=resolve_experiment_path(embedding_config["cache_dir"], project_root),
            device=embedding_config.get("device"),
            expected_dimension=int(embedding_config["dimension"]),
        )
        store = LanceDbEvidenceStore(
            database_path=_database_path(config, project_root),
            table_name=retrieval_config["table"],
            dataset_root=resolve_experiment_path(
                retrieval_config["dataset_root"], project_root
            ),
        )

    vlm_config = config["vlm"]
    client = None
    if any("vlm" in variant for variant in variants):
        client = NvidiaNimClient(
            model=vlm_config["model"],
            cache_root=resolve_experiment_path(vlm_config["cache_root"], project_root),
            base_url=vlm_config["base_url"],
            api_key_variable=vlm_config["api_key_variable"],
            timeout=float(vlm_config["timeout"]),
            requests_per_minute=float(vlm_config.get("requests_per_minute", 30)),
        )

    montage_directory = output_directory / "montages"
    if execution.get("save_montages", True):
        montage_directory.mkdir(parents=True, exist_ok=True)
    classes = tuple(config["classes"])
    print(
        f"Selected cases: {len(cases)}; already completed: {len(completed_ids)}; "
        f"false negatives excluded from per-box VLM evaluation: {excluded_false_negatives}"
    )
    with results_path.open("a", encoding="utf-8", buffering=1) as results_file:
        for index, case in enumerate(cases, start=1):
            if case.case_id in completed_ids:
                continue
            record: dict[str, Any] = {
                "case_id": case.case_id,
                "image_id": case.image_id,
                "image_path": str(case.image_path),
                "source_split": case.source_split,
                "ground_truth_status": case.status,
                "ground_truth_id": case.ground_truth_id,
                "ground_truth_class_id": case.ground_truth_class_id,
                "ground_truth_class_name": case.ground_truth_class_name,
                "iou": case.iou,
                "detector": case.detection.to_dict(),
                "run_status": "running",
                "started_at_utc": datetime.now(timezone.utc).isoformat(),
            }
            try:
                with Image.open(case.image_path) as opened:
                    image = opened.convert("RGB")
                case.detection.box.validate(image.width, image.height)
                evidence = None
                if needs_retrieval:
                    started = time.perf_counter()
                    evidence = retrieve_detection_evidence(
                        image=image,
                        detection=case.detection,
                        encoder=encoder,
                        store=store,
                        context_margin=float(embedding_config["context_margin"]),
                        evidence_splits=tuple(retrieval_config["evidence_splits"]),
                        top_k=int(retrieval_config["top_k"]),
                        candidate_limit=int(retrieval_config["candidate_limit"]),
                        query_image_id=case.image_id,
                    )
                    record["retrieval"] = {
                        "latency_seconds": time.perf_counter() - started,
                        "object_evidence": [
                            _evidence_dict(item) for item in evidence.object_cases
                        ],
                        "context_evidence": [
                            _evidence_dict(item) for item in evidence.context_cases
                        ],
                    }

                max_evidence = int(vlm_config["max_evidence"])
                vlm_evidence = evidence.object_cases[:max_evidence] if evidence else []
                evidence_crops = [store.evidence_crop(item) for item in vlm_evidence]
                evidence_images = [store.evidence_image(item) for item in vlm_evidence]
                for variant, mode in (
                    ("query_only_vlm", "query_only"),
                    ("retrieval_grounded_vlm", "retrieval_grounded"),
                ):
                    if variant not in variants:
                        continue
                    selected_evidence = vlm_evidence if mode == "retrieval_grounded" else []
                    prompt = build_prompt(case.detection, selected_evidence, classes, mode)
                    started = time.perf_counter()
                    assessment = client.assess(
                        query_image=image,
                        query_crop=(
                            evidence.object_crop if evidence else image.crop(case.detection.box.as_list())
                        ),
                        detection=case.detection,
                        classes=classes,
                        mode=mode,
                        evidence=vlm_evidence,
                        evidence_images=evidence_images,
                        evidence_crops=evidence_crops,
                    )
                    result = _assessment_dict(
                        assessment,
                        prompt=prompt,
                        latency_seconds=time.perf_counter() - started,
                        supplied_evidence_count=len(selected_evidence),
                    )
                    result["correct"] = assessment_is_correct(case, result)
                    record[variant] = result
                    if execution.get("save_montages", True):
                        montage = grounded_montage(
                            image,
                            evidence.object_crop if evidence else image.crop(case.detection.box.as_list()),
                            case.detection,
                            selected_evidence,
                            evidence_images if selected_evidence else [],
                            evidence_crops if selected_evidence else [],
                        )
                        montage.save(
                            montage_directory / f"{case.case_id}_{mode}.jpg",
                            quality=90,
                        )
                if "configured_policy" in variants:
                    grounded = record["retrieval_grounded_vlm"]
                    assessment = VlmAssessment(
                        decision=grounded["decision"],
                        suggested_class=grounded["suggested_class"],
                        confidence=grounded["confidence"],
                        observations=grounded["observations"],
                        uncertainty=grounded["uncertainty"],
                        evidence_ids=tuple(grounded["evidence_ids"]),
                        raw_response=grounded["raw_response"],
                    )
                    record["configured_policy"] = evaluate_acceptance_policy(
                        detection=case.detection,
                        evidence=vlm_evidence,
                        assessment=assessment,
                        settings=config["decision_policy"],
                    ).to_dict()
                record["run_status"] = "completed"
            except Exception as error:
                record["run_status"] = "failed"
                record["error"] = f"{type(error).__name__}: {error}"
                record["traceback"] = traceback.format_exc()
                _append_experiment_log(
                    output_directory,
                    f"CASE FAILED {case.case_id}\n{record['traceback']}",
                )
            record["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
            results_file.write(json.dumps(record) + "\n")
            results_file.flush()
            print(f"[{index}/{len(cases)}] {case.case_id}: {record['run_status']}")

    latest = _latest_records(results_path)
    write_assistance_metrics(latest.values(), output_directory)
    summary = {
        "selected_cases": len(cases),
        "completed_cases": sum(
            row.get("run_status") == "completed" for row in latest.values()
        ),
        "failed_cases": sum(row.get("run_status") == "failed" for row in latest.values()),
        "excluded_false_negatives": excluded_false_negatives,
        "status_counts": dict(Counter(case.status for case in cases)),
    }
    (output_directory / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    if summary["failed_cases"]:
        raise RuntimeError(
            f"{summary['failed_cases']} experiment cases failed. Completed cases "
            "were preserved; rerun with resume=true after correcting the error."
        )
    return output_directory


def run_assistance_experiment(
    config: dict[str, Any], project_root: Path
) -> Path:
    """Run the experiment and always persist a diagnostic execution log."""
    output_directory = experiment_output_directory(config, project_root)
    cases = config["cases"]
    execution = config["execution"]
    _append_experiment_log(
        output_directory,
        "RUN START\n"
        f"config={config.get('_config_path', '<in-memory>')}\n"
        f"inference_directory={cases.get('inference_directory')}\n"
        f"image_root={cases.get('image_root')}\n"
        f"splits={cases.get('splits')}\n"
        f"statuses={cases.get('statuses')}\n"
        f"maximum_per_status={cases.get('maximum_per_status')}\n"
        f"minimum_detector_confidence="
        f"{cases.get('minimum_detector_confidence', 0.0)}\n"
        f"variants={execution.get('variants')}",
    )
    try:
        result = _run_assistance_experiment(config, project_root)
    except Exception:
        _append_experiment_log(output_directory, f"RUN FAILED\n{traceback.format_exc()}")
        raise
    _append_experiment_log(output_directory, "RUN SUCCEEDED")
    return result

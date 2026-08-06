from __future__ import annotations

import csv
from pathlib import Path
import re
from typing import Any

from src.detection.config import resolve_path, save_resolved_config
from src.detection.inference import (
    DetectionRecord,
    InferenceImageRecord,
    discover_inference_images,
    write_inference_tables,
)
from src.detection.reporting import collect_runtime_metadata, write_json


def _dataset_yaml(config: dict[str, Any], project_root: Path) -> Path:
    path = resolve_path(
        Path(config["export"]["output_root"]) / "dataset.yaml", project_root
    )
    if not path.is_file():
        raise FileNotFoundError(
            f"Prepared dataset YAML not found: {path}. Run prepare_dataset.py first."
        )
    return path


def _load_yolo(weights: str | Path):
    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise RuntimeError(
            "Ultralytics is not installed. Run: python -m pip install -e ."
        ) from error
    return YOLO(str(weights))


def _ultralytics_save_dir(model: Any, results: Any, stage: str) -> Path:
    """Return the run directory across supported Ultralytics result variants."""
    save_dir = getattr(results, "save_dir", None)
    if save_dir is None:
        runner = getattr(model, "validator" if stage == "evaluation" else "predictor", None)
        save_dir = getattr(runner, "save_dir", None)
    if save_dir is None:
        raise RuntimeError(f"Ultralytics did not report a {stage} output directory")
    return Path(save_dir).resolve()


def train_detector(config: dict[str, Any], project_root: Path) -> Path:
    if config["detector"]["backend"] != "ultralytics":
        raise ValueError("train_detector currently supports the ultralytics backend")

    dataset_yaml = _dataset_yaml(config, project_root)
    output_root = resolve_path(config["outputs"]["training_root"], project_root)
    training = config["training"]
    model = _load_yolo(config["model"]["weights"])

    results = model.train(
        data=str(dataset_yaml),
        epochs=training["epochs"],
        imgsz=training["imgsz"],
        batch=training["batch"],
        device=training["device"],
        workers=training["workers"],
        seed=config["experiment"]["seed"],
        deterministic=training["deterministic"],
        optimizer=training["optimizer"],
        warmup_epochs=training["warmup_epochs"],
        cos_lr=training["cos_lr"],
        lrf=training["lrf"],
        patience=training["patience"],
        pretrained=training["pretrained"],
        save=training["save"],
        save_period=training["save_period"],
        plots=training["plots"],
        verbose=training["verbose"],
        project=str(output_root),
        name=config["experiment"]["name"],
        exist_ok=training["exist_ok"],
    )

    run_directory = Path(results.save_dir).resolve()
    best_checkpoint = run_directory / "weights" / "best.pt"
    if not best_checkpoint.is_file():
        raise RuntimeError(f"Training completed without best.pt: {run_directory}")
    save_resolved_config(config, run_directory / "resolved_config.yaml")
    metadata = collect_runtime_metadata(project_root)
    metadata.update({
        "stage": "training",
        "dataset_yaml": str(dataset_yaml),
        "best_checkpoint": str(best_checkpoint),
    })
    write_json(metadata, run_directory / "training_metadata.json")
    return best_checkpoint


def evaluate_detector(
    config: dict[str, Any],
    project_root: Path,
    checkpoint: Path,
) -> Path:
    checkpoint = checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    dataset_yaml = _dataset_yaml(config, project_root)
    evaluation = config["evaluation"]
    split = evaluation["split"]
    output_root = resolve_path(config["outputs"]["evaluation_root"], project_root)
    run_name = f"{config['experiment']['name']}_{split}"
    model = _load_yolo(checkpoint)
    metrics = model.val(
        data=str(dataset_yaml),
        split=split,
        imgsz=evaluation["imgsz"],
        batch=evaluation["batch"],
        device=evaluation["device"],
        workers=evaluation["workers"],
        plots=evaluation["plots"],
        verbose=evaluation["verbose"],
        project=str(output_root),
        name=run_name,
        exist_ok=evaluation["exist_ok"],
    )

    run_directory = _ultralytics_save_dir(model, metrics, "evaluation")
    overall = {
        "split": split,
        "checkpoint": str(checkpoint),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "mAP50": float(metrics.box.map50),
        "mAP50_95": float(metrics.box.map),
        "speed_ms_per_image": {
            key: float(value) for key, value in metrics.speed.items()
        },
    }
    write_json(overall, run_directory / "overall_metrics.json")

    per_class_rows = []
    for result_index, class_id in enumerate(metrics.box.ap_class_index):
        class_id = int(class_id)
        precision, recall, ap50, ap50_95 = metrics.class_result(result_index)
        instances = (
            int(metrics.nt_per_class[class_id])
            if metrics.nt_per_class is not None
            else None
        )
        per_class_rows.append({
            "class_id": class_id,
            "class_name": metrics.names[class_id],
            "instances": instances,
            "precision": float(precision),
            "recall": float(recall),
            "AP50": float(ap50),
            "AP50_95": float(ap50_95),
        })

    per_class_path = run_directory / "per_class_metrics.csv"
    fieldnames = [
        "class_id",
        "class_name",
        "instances",
        "precision",
        "recall",
        "AP50",
        "AP50_95",
    ]
    with per_class_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_class_rows)

    save_resolved_config(config, run_directory / "resolved_config.yaml")
    metadata = collect_runtime_metadata(project_root)
    metadata.update({"stage": "evaluation", **overall})
    write_json(metadata, run_directory / "evaluation_metadata.json")
    return run_directory


def _safe_run_component(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")
    if not cleaned:
        raise ValueError(f"Value cannot form a safe output name: {value!r}")
    return cleaned


def export_predictions(
    config: dict[str, Any],
    project_root: Path,
    checkpoint: Path,
    source: Path,
    dataset_id: str,
    source_split: str,
) -> Path:
    """Export structured predictions for an arbitrary local image source."""
    if config["detector"]["backend"] != "ultralytics":
        raise ValueError(
            "export_predictions currently supports the ultralytics backend"
        )
    dataset_id = dataset_id.strip()
    source_split = source_split.strip().lower()
    if not dataset_id:
        raise ValueError("dataset_id must not be empty")
    if not source_split:
        raise ValueError("source_split must not be empty")

    checkpoint = checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    source = resolve_path(source, project_root)
    source_images = discover_inference_images(source)

    inference = config["inference"]
    output_root = resolve_path(config["outputs"]["root"], project_root)
    safe_dataset_id = _safe_run_component(dataset_id)
    safe_source_split = _safe_run_component(source_split)
    run_name = (
        f"{config['run']['name']}_{safe_dataset_id}_{safe_source_split}"
    )
    model = _load_yolo(checkpoint)
    results = model.predict(
        source=[str(path) for path in source_images],
        imgsz=inference["imgsz"],
        batch=inference["batch"],
        conf=inference["confidence"],
        iou=inference["iou"],
        device=inference["device"],
        save=inference["save_rendered_images"],
        save_txt=inference["save_yolo_labels"],
        save_conf=inference["save_confidence"],
        verbose=inference["verbose"],
        project=str(output_root),
        name=run_name,
        exist_ok=inference["exist_ok"],
    )
    if len(results) != len(source_images):
        raise RuntimeError(
            f"Ultralytics returned {len(results)} results for "
            f"{len(source_images)} images"
        )

    image_records: list[InferenceImageRecord] = []
    detection_records: list[DetectionRecord] = []
    source_indices = {
        path.resolve(): index for index, path in enumerate(source_images)
    }
    returned_sources: set[Path] = set()
    for result in results:
        source_path = Path(result.path).resolve()
        if source_path not in source_indices:
            raise RuntimeError(f"Ultralytics returned an unknown source: {source_path}")
        if source_path in returned_sources:
            raise RuntimeError(f"Ultralytics returned a source twice: {source_path}")
        returned_sources.add(source_path)
        source_index = source_indices[source_path]
        image_id = f"{safe_dataset_id}_{source_index:06d}_{source_path.stem}"
        image_height, image_width = map(int, result.orig_shape)
        boxes = result.boxes
        coordinates = boxes.xyxy.cpu().tolist()
        class_ids = boxes.cls.cpu().tolist()
        confidences = boxes.conf.cpu().tolist()
        if not (len(coordinates) == len(class_ids) == len(confidences)):
            raise RuntimeError(f"Incomplete Ultralytics result for {source_path}")

        image_records.append(InferenceImageRecord(
            image_id=image_id,
            dataset_id=dataset_id,
            source_image_path=str(source_path),
            source_split=source_split,
            image_width=image_width,
            image_height=image_height,
            prediction_count=len(coordinates),
        ))
        for index, (box, class_value, confidence) in enumerate(
            zip(coordinates, class_ids, confidences, strict=True)
        ):
            xmin, ymin, xmax, ymax = map(float, box)
            xmin = min(max(xmin, 0.0), image_width)
            xmax = min(max(xmax, 0.0), image_width)
            ymin = min(max(ymin, 0.0), image_height)
            ymax = min(max(ymax, 0.0), image_height)
            class_id = int(class_value)
            class_name = result.names[class_id]
            detection_records.append(DetectionRecord(
                detection_id=f"{image_id}_det_{index:04d}",
                image_id=image_id,
                predicted_class_id=class_id,
                predicted_class_name=str(class_name),
                confidence=float(confidence),
                xmin=xmin,
                ymin=ymin,
                xmax=xmax,
                ymax=ymax,
            ))

    run_directory = _ultralytics_save_dir(model, results[-1], "prediction")
    summary = write_inference_tables(
        run_directory, image_records, detection_records
    )
    save_resolved_config(config, run_directory / "resolved_config.yaml")
    metadata = collect_runtime_metadata(project_root)
    metadata.update({
        "stage": "inference_export",
        "dataset_id": dataset_id,
        "source_split": source_split,
        "checkpoint": str(checkpoint),
        "source": str(source),
        "confidence": inference["confidence"],
        "iou": inference["iou"],
        **summary,
    })
    write_json(metadata, run_directory / "inference_metadata.json")
    return run_directory


def predict_images(
    config: dict[str, Any],
    project_root: Path,
    checkpoint: Path,
    source: str,
) -> Path:
    checkpoint = checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    prediction = config["prediction"]
    output_root = resolve_path(config["outputs"]["prediction_root"], project_root)
    model = _load_yolo(checkpoint)
    model.predict(
        source=source,
        imgsz=prediction["imgsz"],
        conf=prediction["confidence"],
        iou=prediction["iou"],
        device=prediction["device"],
        save=prediction["save_images"],
        save_txt=prediction["save_labels"],
        save_conf=prediction["save_confidence"],
        project=str(output_root),
        name=f"{config['experiment']['name']}_predict",
        exist_ok=False,
    )
    run_directory = _ultralytics_save_dir(model, None, "prediction")
    metadata = collect_runtime_metadata(project_root)
    metadata.update({
        "stage": "prediction",
        "checkpoint": str(checkpoint),
        "source": source,
    })
    write_json(metadata, run_directory / "prediction_metadata.json")
    return run_directory

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from src.detection.config import resolve_path, save_resolved_config
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

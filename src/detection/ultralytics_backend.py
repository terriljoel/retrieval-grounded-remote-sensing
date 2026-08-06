from __future__ import annotations

import csv
from pathlib import Path
import re
from typing import Any

from src.data.manifests import load_split_manifest
from src.data.parsers.nwpu import parse_nwpu_annotation
from src.detection.config import resolve_path, save_resolved_config
from src.detection.inference import (
    ComparisonRecord,
    DetectionRecord,
    GroundTruthRecord,
    InferenceImageRecord,
    compare_predictions_to_ground_truth,
    discover_inference_images,
    parse_yolo_ground_truth,
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


def _ground_truth_annotation_path(
    image_path: Path,
    source: Path,
    ground_truth_root: Path,
) -> Path | None:
    """Match an image to a label by relative path, then by filename stem."""
    if ground_truth_root.is_file():
        return ground_truth_root if source.is_file() else None

    candidates: list[Path] = []
    if source.is_dir():
        relative_image = image_path.relative_to(source)
        candidates.append(ground_truth_root / relative_image.with_suffix(".txt"))
    candidates.append(ground_truth_root / f"{image_path.stem}.txt")
    return next((path.resolve() for path in candidates if path.is_file()), None)


def _load_ground_truth_boxes(
    annotation_path: Path | None,
    annotation_format: str,
    image_width: int,
    image_height: int,
) -> list[tuple[int, float, float, float, float]]:
    if annotation_path is None:
        return []
    if annotation_format == "yolo":
        return parse_yolo_ground_truth(
            annotation_path, image_width, image_height
        )
    if annotation_format == "nwpu":
        boxes = []
        for box in parse_nwpu_annotation(annotation_path):
            if box.class_id not in range(1, 11):
                raise ValueError(
                    f"NWPU class ID must be 1-10 in {annotation_path}, "
                    f"got {box.class_id}"
                )
            if not (
                0 <= box.xmin < box.xmax <= image_width
                and 0 <= box.ymin < box.ymax <= image_height
            ):
                raise ValueError(
                    f"NWPU box is outside {image_width}x{image_height}: {box}"
                )
            boxes.append((
                box.class_id - 1,
                float(box.xmin),
                float(box.ymin),
                float(box.xmax),
                float(box.ymax),
            ))
        return boxes
    raise ValueError(f"Unsupported ground-truth format: {annotation_format}")


def export_predictions(
    config: dict[str, Any],
    project_root: Path,
) -> Path:
    """Export structured predictions for an arbitrary local image source."""
    if config["detector"]["backend"] != "ultralytics":
        raise ValueError(
            "export_predictions currently supports the ultralytics backend"
        )
    detector = config["detector"]
    source_config = config["source"]
    dataset_id = str(source_config["dataset_id"]).strip()
    source_split = str(source_config.get("split") or "external").strip().lower()
    allow_test = source_config.get("allow_test", False)
    checkpoint = resolve_path(detector["checkpoint"], project_root)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    source = resolve_path(source_config["path"], project_root)
    source_images = discover_inference_images(source)

    manifest_path: Path | None = None
    manifest_lookup: dict[Path, tuple[str, Any]] = {}
    manifest_value = source_config.get("manifest")
    if manifest_value:
        manifest_path = resolve_path(manifest_value, project_root)
        resolved_dataset_root = resolve_path(source_config["dataset_root"], project_root)
        manifest_splits = load_split_manifest(manifest_path, resolved_dataset_root)
        for split_name, items in manifest_splits.items():
            for item in items:
                manifest_lookup[item.image_path.resolve()] = (split_name, item)

    matched_manifest_images = sum(
        image.resolve() in manifest_lookup for image in source_images
    )
    source_splits = {
        manifest_lookup[image.resolve()][0]
        for image in source_images
        if image.resolve() in manifest_lookup
    }
    if (source_split == "test" or "test" in source_splits) and not allow_test:
        raise ValueError(
            "Test images require source.allow_test=true in the inference config"
        )

    ground_truth_config = config.get("ground_truth")
    compare_ground_truth = ground_truth_config is not None
    ground_truth_root: Path | None = None
    annotation_format: str | None = None
    if ground_truth_config is not None:
        ground_truth_root = resolve_path(ground_truth_config["path"], project_root)
        if not ground_truth_root.exists():
            raise FileNotFoundError(
                f"Ground-truth path not found: {ground_truth_root}"
            )
        if ground_truth_root.is_file() and len(source_images) != 1:
            raise ValueError(
                "A ground-truth file can only be used with one source image"
            )
        annotation_format = ground_truth_config["format"]

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
    ground_truth_records: list[GroundTruthRecord] = []
    matched_ground_truth_files = 0
    for source_index, (source_path, result) in enumerate(
        zip(source_images, results, strict=True)
    ):
        source_path = source_path.resolve()
        image_id = f"{safe_dataset_id}_{source_index:06d}_{source_path.stem}"
        image_height, image_width = map(int, result.orig_shape)
        manifest_entry = manifest_lookup.get(source_path)
        image_split = manifest_entry[0] if manifest_entry else source_split
        ground_truth_boxes: list[tuple[int, float, float, float, float]] = []
        if compare_ground_truth:
            annotation_path = _ground_truth_annotation_path(
                source_path, source, ground_truth_root
            )
            if annotation_path is not None:
                matched_ground_truth_files += 1
            ground_truth_boxes = _load_ground_truth_boxes(
                annotation_path,
                annotation_format,
                image_width,
                image_height,
            )
            for ground_truth_index, ground_truth_box in enumerate(ground_truth_boxes):
                class_id, xmin, ymin, xmax, ymax = ground_truth_box
                if class_id not in result.names:
                    raise ValueError(
                        f"Ground-truth class {class_id} is not in model classes"
                    )
                ground_truth_records.append(GroundTruthRecord(
                    ground_truth_id=(
                        f"{image_id}_gt_{ground_truth_index:04d}"
                    ),
                    image_id=image_id,
                    class_id=class_id,
                    class_name=str(result.names[class_id]),
                    xmin=xmin,
                    ymin=ymin,
                    xmax=xmax,
                    ymax=ymax,
                ))
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
            source_split=image_split,
            image_width=image_width,
            image_height=image_height,
            prediction_count=len(coordinates),
            ground_truth_count=(
                len(ground_truth_boxes) if compare_ground_truth else None
            ),
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
    comparison_records: list[ComparisonRecord] | None = None
    if compare_ground_truth:
        comparison_records = compare_predictions_to_ground_truth(
            detection_records,
            ground_truth_records,
            iou_threshold=ground_truth_config.get("iou_threshold", 0.5),
        )
    summary = write_inference_tables(
        run_directory,
        image_records,
        detection_records,
        ground_truths=ground_truth_records if compare_ground_truth else None,
        comparisons=comparison_records,
    )
    save_resolved_config(config, run_directory / "resolved_config.yaml")
    metadata = collect_runtime_metadata(project_root)
    metadata.update({
        "stage": "inference_export",
        "dataset_id": dataset_id,
        "source_split": source_split,
        "checkpoint": str(checkpoint),
        "source": str(source),
        "manifest": str(manifest_path) if manifest_path else None,
        "manifest_matched_images": matched_manifest_images,
        "manifest_unmatched_images": len(source_images) - matched_manifest_images,
        "compare_ground_truth": compare_ground_truth,
        "ground_truth_format": annotation_format,
        "ground_truth_path": str(ground_truth_root) if ground_truth_root else None,
        "ground_truth_matched_files": matched_ground_truth_files,
        "ground_truth_unmatched_images": (
            len(source_images) - matched_ground_truth_files
            if compare_ground_truth else None
        ),
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

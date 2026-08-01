from __future__ import annotations

import csv
from pathlib import Path

from src.data.models import DatasetItem
from src.data.splitting import DatasetSplits, validate_dataset_splits


def _parse_boolean(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(f"Invalid boolean value in manifest: {value!r}")


def load_split_manifest(
    manifest_path: Path,
    dataset_root: Path,
) -> DatasetSplits:
    """Reconstruct fixed dataset splits from a portable CSV manifest."""
    manifest_path = manifest_path.resolve()
    dataset_root = dataset_root.resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Split manifest not found: {manifest_path}")

    splits: DatasetSplits = {"train": [], "val": [], "test": []}
    with manifest_path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        required = {"image_path", "annotation_path", "is_background", "split"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Manifest is missing columns: {sorted(missing)}")

        for row_number, row in enumerate(reader, start=2):
            split = row["split"]
            if split not in splits:
                raise ValueError(f"{manifest_path}:{row_number}: invalid split {split!r}")
            image_path = (dataset_root / row["image_path"]).resolve()
            annotation_value = row["annotation_path"].strip()
            annotation_path = (
                (dataset_root / annotation_value).resolve()
                if annotation_value else None
            )
            if not image_path.is_file():
                raise FileNotFoundError(f"Manifest image not found: {image_path}")
            if annotation_path is not None and not annotation_path.is_file():
                raise FileNotFoundError(
                    f"Manifest annotation not found: {annotation_path}"
                )
            splits[split].append(DatasetItem(
                image_path=image_path,
                annotation_path=annotation_path,
                is_background=_parse_boolean(row["is_background"]),
            ))

    validate_dataset_splits(splits)
    return splits

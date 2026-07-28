# src/data/splitting.py

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from sklearn.model_selection import train_test_split

from src.data.models import DatasetItem
from iterstrat.ml_stratifiers import (
    MultilabelStratifiedShuffleSplit,
)
import numpy as np

from src.data.parsers.nwpu import (
    parse_nwpu_annotation,
)


def create_multilabel_targets(
    items: list[DatasetItem],
    number_of_classes: int = 10,
) -> np.ndarray:
    """Create class-presence targets plus one background target."""
    targets = np.zeros(
        (len(items), number_of_classes + 1),
        dtype=np.int8,
    )

    background_index = number_of_classes

    for item_index, item in enumerate(items):
        if item.is_background:
            targets[item_index, background_index] = 1
            continue

        if item.annotation_path is None:
            raise ValueError(
                f"Positive image has no annotation: "
                f"{item.image_path}"
            )

        boxes = parse_nwpu_annotation(
            item.annotation_path
        )

        for box in boxes:
            zero_based_class_id = box.class_id - 1

            if zero_based_class_id not in range(
                number_of_classes
            ):
                raise ValueError(
                    f"Unexpected class ID: {box.class_id}"
                )

            targets[
                item_index,
                zero_based_class_id,
            ] = 1

    return targets


@dataclass(frozen=True)
class SplitConfig:
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    seed: int = 42

    def validate(self) -> None:
        ratios = (
            self.train_ratio,
            self.val_ratio,
            self.test_ratio,
        )

        if any(ratio <= 0 for ratio in ratios):
            raise ValueError("All split ratios must be positive.")

        if abs(sum(ratios) - 1.0) > 1e-9:
            raise ValueError(
                "Train, validation, and test ratios must sum to 1.0."
            )


DatasetSplits = dict[str, list[DatasetItem]]


def create_dataset_splits(
    items: list[DatasetItem],
    config: SplitConfig = SplitConfig(),
) -> DatasetSplits:
    """Create deterministic multilabel-stratified splits."""
    config.validate()

    if not items:
        raise ValueError(
            "Cannot split an empty dataset."
        )

    image_paths = [
        item.image_path.resolve()
        for item in items
    ]

    if len(image_paths) != len(set(image_paths)):
        raise ValueError(
            "Duplicate image paths were provided."
        )

    targets = create_multilabel_targets(items)

    item_indices = np.arange(len(items))

    remaining_ratio = (
        config.val_ratio
        + config.test_ratio
    )

    train_splitter = (
        MultilabelStratifiedShuffleSplit(
            n_splits=1,
            test_size=remaining_ratio,
            random_state=config.seed,
        )
    )

    train_indices, remaining_indices = next(
        train_splitter.split(
            item_indices,
            targets,
        )
    )

    remaining_targets = targets[
        remaining_indices
    ]

    relative_test_ratio = (
        config.test_ratio
        / remaining_ratio
    )

    validation_splitter = (
        MultilabelStratifiedShuffleSplit(
            n_splits=1,
            test_size=relative_test_ratio,
            random_state=config.seed,
        )
    )

    validation_relative_indices, test_relative_indices = next(
        validation_splitter.split(
            remaining_indices,
            remaining_targets,
        )
    )

    validation_indices = remaining_indices[
        validation_relative_indices
    ]

    test_indices = remaining_indices[
        test_relative_indices
    ]

    splits = {
        "train": [
            items[index]
            for index in train_indices
        ],
        "val": [
            items[index]
            for index in validation_indices
        ],
        "test": [
            items[index]
            for index in test_indices
        ],
    }

    validate_dataset_splits(
        splits,
        expected_total=len(items),
        require_backgrounds=True,
    )

    return splits


def validate_dataset_splits(
    splits: DatasetSplits,
    expected_total: int | None = None,
    require_backgrounds: bool = True,
) -> None:
    """Validate split names, sizes, overlap, and backgrounds."""
    expected_names = {"train", "val", "test"}

    if set(splits) != expected_names:
        raise ValueError(
            f"Expected splits {sorted(expected_names)}, "
            f"got {sorted(splits)}."
        )

    split_paths: dict[str, set[Path]] = {}

    for split_name, split_items in splits.items():
        if not split_items:
            raise ValueError(f"Split is empty: {split_name}")

        paths = {
            item.image_path.resolve()
            for item in split_items
        }

        if len(paths) != len(split_items):
            raise ValueError(
                f"Duplicate image paths inside {split_name} split."
            )

        if require_backgrounds and not any(
            item.is_background
            for item in split_items
        ):
            raise ValueError(
                f"No background images in {split_name} split."
            )

        split_paths[split_name] = paths

    comparisons = (
        ("train", "val"),
        ("train", "test"),
        ("val", "test"),
    )

    for first_name, second_name in comparisons:
        overlap = (
            split_paths[first_name]
            & split_paths[second_name]
        )

        if overlap:
            raise ValueError(
                f"{first_name} and {second_name} overlap by "
                f"{len(overlap)} images."
            )

    actual_total = sum(
        len(split_items)
        for split_items in splits.values()
    )

    if (
        expected_total is not None
        and actual_total != expected_total
    ):
        raise ValueError(
            f"Expected {expected_total} images, "
            f"found {actual_total}."
        )


def save_split_manifest(
    splits: DatasetSplits,
    output_path: Path,
    dataset_root: Path,
    config: SplitConfig = SplitConfig(),
    overwrite: bool = False,
) -> Path:
    """Save split assignments using paths relative to the dataset root."""
    config.validate()
    validate_dataset_splits(splits)

    output_path = output_path.resolve()
    dataset_root = dataset_root.resolve()

    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Manifest already exists: {output_path}. "
            "Pass overwrite=True to replace it."
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    records = []

    for split_name, split_items in splits.items():
        for item in split_items:
            image_path = item.image_path.resolve()

            try:
                relative_image_path = image_path.relative_to(
                    dataset_root
                )
            except ValueError as error:
                raise ValueError(
                    f"Image is outside the dataset root: {image_path}"
                ) from error

            relative_annotation_path = ""

            if item.annotation_path is not None:
                annotation_path = item.annotation_path.resolve()

                try:
                    relative_annotation_path = str(
                        annotation_path.relative_to(dataset_root)
                    )
                except ValueError as error:
                    raise ValueError(
                        "Annotation is outside the dataset root: "
                        f"{annotation_path}"
                    ) from error

            records.append({
                "image_id": image_path.stem,
                "image_path": str(relative_image_path),
                "annotation_path": relative_annotation_path,
                "is_background": item.is_background,
                "split": split_name,
                "seed": config.seed,
                "train_ratio": config.train_ratio,
                "val_ratio": config.val_ratio,
                "test_ratio": config.test_ratio,
            })

    records.sort(
        key=lambda record: (
            record["split"],
            record["image_id"],
        )
    )

    fieldnames = [
        "image_id",
        "image_path",
        "annotation_path",
        "is_background",
        "split",
        "seed",
        "train_ratio",
        "val_ratio",
        "test_ratio",
    ]

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(records)

    return output_path
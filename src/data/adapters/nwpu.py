from pathlib import Path

from src.data.models import DatasetItem


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
}


def discover_nwpu_items(
    dataset_root: Path,
) -> list[DatasetItem]:
    positive_dir = next(
        (
            path
            for path in dataset_root.rglob("*")
            if path.is_dir()
            and path.name.lower() == "positive image set"
        ),
        None,
    )

    negative_dir = next(
        (
            path
            for path in dataset_root.rglob("*")
            if path.is_dir()
            and path.name.lower() == "negative image set"
        ),
        None,
    )

    annotation_dir = next(
        (
            path
            for path in dataset_root.rglob("*")
            if path.is_dir()
            and path.name.lower() == "ground truth"
        ),
        None,
    )

    if positive_dir is None:
        raise FileNotFoundError("Positive image directory not found")

    if negative_dir is None:
        raise FileNotFoundError("Negative image directory not found")

    if annotation_dir is None:
        raise FileNotFoundError("Ground-truth directory not found")

    positive_images = sorted(
        path
        for path in positive_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    negative_images = sorted(
        path
        for path in negative_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    annotations = {
        path.stem: path
        for path in annotation_dir.glob("*.txt")
        if path.stem.isdigit()
    }

    items = []

    for image_path in positive_images:
        items.append(
            DatasetItem(
                image_path=image_path,
                annotation_path=annotations.get(image_path.stem),
                is_background=False,
            )
        )

    for image_path in negative_images:
        items.append(
            DatasetItem(
                image_path=image_path,
                annotation_path=None,
                is_background=True,
            )
        )

    return items
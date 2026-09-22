from pathlib import Path
from xml.etree import ElementTree

from src.data.models import BoundingBox


def parse_pascal_voc_annotation(
    annotation_path: Path,
    class_mapping: dict[str, int],
) -> list[BoundingBox]:
    """Parse mapped objects from a Pascal VOC XML annotation."""
    try:
        root = ElementTree.parse(annotation_path).getroot()
    except ElementTree.ParseError as error:
        raise ValueError(f"Invalid Pascal VOC XML: {annotation_path}") from error

    normalized_mapping = {
        " ".join(name.strip().lower().split()): class_id
        for name, class_id in class_mapping.items()
    }
    boxes = []

    for object_number, object_node in enumerate(root.findall("object"), start=1):
        raw_class_name = object_node.findtext("name") or ""
        class_name = " ".join(raw_class_name.strip().lower().split())
        if class_name not in normalized_mapping:
            continue

        box_node = object_node.find("bndbox")
        location = f"{annotation_path}, object {object_number}"
        if box_node is None:
            raise ValueError(f"{location}: missing bndbox")

        try:
            xmin, ymin, xmax, ymax = (
                int(float(box_node.findtext(coordinate, "")))
                for coordinate in ("xmin", "ymin", "xmax", "ymax")
            )
        except ValueError as error:
            raise ValueError(f"{location}: invalid bounding-box coordinates") from error

        if xmax <= xmin or ymax <= ymin:
            raise ValueError(f"{location}: empty bounding box")

        boxes.append(
            BoundingBox(
                xmin=xmin,
                ymin=ymin,
                xmax=xmax,
                ymax=ymax,
                class_id=normalized_mapping[class_name],
            )
        )

    return boxes

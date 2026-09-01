"""HRRSD as a second memory partition: VOC XML parsing and scope loading.

HRRSD shares ten class names with NWPU and adds three NWPU does not have --
crossroad, parking lot, T junction. Those three are the natural open-set
complement, so class names are normalised to NWPU's underscore form here
rather than downstream: "storage tank" -> "storage_tank". Without that the
shared classes would not join and every HRRSD class would look novel.

**RemoteCLIP was trained on HRRSD and not on NWPU.** Any retrieval number
measured on HRRSD is inflated by that and must be reported separately -- the
`dataset` column on the case memory exists so a combined memory can always be
split back apart.

Crops go through `src.retrieval.crops.extract_crops` with the identical
geometry NWPU uses; only the annotation parser and the class map differ.
"""

from __future__ import annotations

import csv
from pathlib import Path
from xml.etree import ElementTree

from src.data.models import BoundingBox

# The ten NWPU classes carry NWPU's names so the two memories join.
NWPU_SHARED = {
    "airplane", "ship", "storage_tank", "baseball_diamond", "tennis_court",
    "basketball_court", "ground_track_field", "harbor", "bridge", "vehicle",
}


def normalise(name: str) -> str:
    return name.strip().lower().replace(" ", "_")


def load_class_map(archive_root: Path) -> dict[int, str]:
    """class_id -> normalised name, read from the dataset's own Classes.csv."""
    mapping: dict[int, str] = {}
    with (archive_root / "Classes.csv").open(encoding="utf-8", newline="") as file:
        for row in csv.reader(file):
            if len(row) == 2:
                mapping[int(row[1])] = normalise(row[0])
    if not mapping:
        raise ValueError(f"No classes parsed from {archive_root / 'Classes.csv'}")
    return mapping


def open_set_classes(class_map: dict[int, str]) -> list[str]:
    """HRRSD classes NWPU has no examples of."""
    return sorted(set(class_map.values()) - NWPU_SHARED)


def voc_parser(class_map: dict[int, str]):
    """Build a parser with the signature `extract_crops` expects.

    Boxes whose class is not in the map are dropped rather than raising: the
    map comes from the dataset's own Classes.csv, so an unknown name means a
    malformed annotation, and one bad file should not stop a 4,000-image build.
    """
    by_name = {name: class_id for class_id, name in class_map.items()}

    def parse(annotation_path: Path) -> list[BoundingBox]:
        root = ElementTree.parse(annotation_path).getroot()
        boxes = []
        for obj in root.findall("object"):
            name = normalise(obj.findtext("name") or "")
            if name not in by_name:
                continue
            corner = obj.find("bndbox")
            if corner is None:
                continue
            boxes.append(BoundingBox(
                xmin=int(float(corner.findtext("xmin"))),
                ymin=int(float(corner.findtext("ymin"))),
                xmax=int(float(corner.findtext("xmax"))),
                ymax=int(float(corner.findtext("ymax"))),
                class_id=by_name[name],
            ))
        return boxes

    return parse


def load_scope(archive_root: Path, subset_file: Path, split: str = "train") -> list[dict]:
    """Scope records for the committed subset IDs, sorted by image_uid.

    image_uid is prefixed `hrrsd_` so a combined memory cannot collide with
    NWPU's uids -- both datasets use bare zero-padded integers.
    """
    ids = sorted({line.strip() for line in subset_file.read_text().splitlines() if line.strip()})
    scope = []
    for image_id in ids:
        image_path = archive_root / "JpegImages" / f"{image_id}.jpg"
        annotation_path = archive_root / "Annotations" / f"{image_id}.xml"
        if not image_path.is_file() or not annotation_path.is_file():
            raise FileNotFoundError(f"HRRSD {image_id}: missing image or annotation")
        scope.append({
            "image_uid": f"hrrsd_{image_id}",
            "split": split,
            "image_path": image_path,
            "annotation_path": annotation_path,
        })
    return scope


def _self_check() -> None:
    import tempfile

    xml = """<annotation>
      <object><name>storage tank</name>
        <bndbox><xmin>10</xmin><ymin>20</ymin><xmax>30</xmax><ymax>40</ymax></bndbox></object>
      <object><name>T junction</name>
        <bndbox><xmin>1</xmin><ymin>2</ymin><xmax>3.9</xmax><ymax>4</ymax></bndbox></object>
      <object><name>not_a_real_class</name>
        <bndbox><xmin>0</xmin><ymin>0</ymin><xmax>1</xmax><ymax>1</ymax></bndbox></object>
    </annotation>"""

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "Classes.csv").write_text(
            "basketball court,0\nstorage tank,5\nT junction,11\nparking lot,10\ncrossroad,3\n"
            "airplane,9\nship,6\nvehicle,7\nharbor,4\nbridge,2\ntennis court,8\n"
            "baseball diamond,1\nground track field,12\n"
        )
        class_map = load_class_map(root)
        assert class_map[5] == "storage_tank" and class_map[11] == "t_junction", class_map
        # The three HRRSD-only classes are exactly the open-set complement.
        assert open_set_classes(class_map) == ["crossroad", "parking_lot", "t_junction"]

        path = root / "a.xml"
        path.write_text(xml)
        boxes = voc_parser(class_map)(path)
        # Two known classes parsed, the unknown one dropped rather than raising.
        assert len(boxes) == 2, boxes
        assert boxes[0] == BoundingBox(10, 20, 30, 40, 5), boxes[0]
        # Float coordinates in VOC XML must not crash int().
        assert boxes[1].xmax == 3, boxes[1]

        # load_scope refuses to silently skip a missing file.
        (root / "JpegImages").mkdir()
        (root / "Annotations").mkdir()
        subset = root / "subset.txt"
        subset.write_text("00001\n")
        try:
            load_scope(root, subset)
        except FileNotFoundError:
            pass
        else:
            raise AssertionError("accepted a subset id with no image on disk")

        (root / "JpegImages" / "00001.jpg").write_bytes(b"")
        (root / "Annotations" / "00001.xml").write_text(xml)
        scope = load_scope(root, subset)
        assert scope[0]["image_uid"] == "hrrsd_00001", scope[0]

    print("hrrsd self-check OK")


if __name__ == "__main__":
    _self_check()

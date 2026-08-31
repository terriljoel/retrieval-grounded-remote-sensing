"""Join the detector export back to `image_uid` and the project's split.

The export keys everything on

    image_id = f"{dataset_id}_{source_index:06d}_{source_path.stem}"

e.g. `new_remote_sensing_dataset_000003_004`. The stem alone is **not** a key:
`positive image set/004.jpg` and `negative image set/004.jpg` both stem to
`004`, which is the 150-collision problem the fixed split exists to avoid. So
the id must be resolved back to a path before it means anything.

Two routes, in order of preference:

1. **`inference_images.csv`** carries `source_image_path` per `image_id`. That
   is a direct lookup and needs nothing else. Use it whenever the file is
   present -- it survives a reordered source directory, which route 2 does not.
2. **`source_index`**, reconstructed by re-running the export's own image
   discovery (`sorted(rglob)` over the source root). Deterministic and cheap --
   no GPU, no model -- but it is positional, so it is only valid if the source
   directory has not changed since the export ran. `verify()` is what says
   whether that held.

**The split comes from the project's fixed split CSV, never from the export and
never from any manifest.** At least four similarly-named seed-42 CSVs exist on
this machine (`nwpu_vhr10_split_seed42.csv`, `nwpu_vhr10_multilabel_split_seed42.csv`,
the `_fixed` one, and the shared-resources manifest). The manifest's split
disagrees with the fixed split on **335 of 800 images**, and mislabels **97 of
the 118 test images** as train or val. Picking the wrong one is a silent
contamination, not an error.

So the split file is **not a parameter**. `load_split()` takes the project root
and builds the one legal path itself, then checks the file's identity by
content (`image_uid` column, 562/120/118) before returning. There is no
argument through which a different split file can be supplied, and
`load_paths()` refuses any CSV carrying a `split` column so a manifest cannot
be fed in through the path-reading door either. Documentation would not have
stopped this; the missing parameter does.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
ID_PATTERN = re.compile(r"^(?P<dataset>.+)_(?P<index>\d{6})_(?P<stem>.+)$")


def parse_image_id(image_id: str) -> tuple[str, int, str]:
    """`..._000003_004` -> (dataset_id, 3, '004').

    The dataset id may itself contain underscores and digits, so the six-digit
    index is anchored from the right, not found by splitting.
    """
    match = ID_PATTERN.match(image_id)
    if not match:
        raise ValueError(f"Not an export image_id: {image_id!r}")
    return match["dataset"], int(match["index"]), match["stem"]


def image_uid(path: Path | str) -> str:
    """Path -> the project's `image_uid`.

    Positive/negative is read from the directory, which is the only thing that
    disambiguates two files with the same stem.
    """
    text = str(path).replace("\\", "/")
    stem = text.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    lowered = text.lower()
    if "negative" in lowered:
        return f"nwpu_background_{stem}"
    if "positive" in lowered:
        return f"nwpu_positive_{stem}"
    raise ValueError(f"Cannot tell positive from negative in path: {path!r}")


def discover(source: Path) -> list[Path]:
    """The export's image ordering, re-derived. Mirrors
    `src.detection.inference.discover_inference_images` -- imported rather than
    copied would be better, but that module belongs to the detection half and
    pulls in ultralytics, which is not a dependency of the retrieval half."""
    return sorted(
        path.resolve()
        for path in source.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def uid_by_index(source: Path) -> dict[int, str]:
    return {index: image_uid(path) for index, path in enumerate(discover(source))}


def uid_from_images_csv(images_csv: Path) -> dict[str, str]:
    """`image_id` -> `image_uid`, straight from the export's own image table."""
    with images_csv.open(encoding="utf-8", newline="") as file:
        return {
            row["image_id"]: image_uid(row["source_image_path"])
            for row in csv.DictReader(file)
        }


SPLIT_CSV = Path("configs/splits/nwpu_vhr10_multilabel_split_seed42_fixed.csv")
EXPECTED_COUNTS = {"train": 562, "val": 120, "test": 118}


def load_split(root: Path) -> dict[str, str]:
    """`image_uid` -> split. The only authority, and deliberately unparameterised.

    Takes the project root, not a file: there is no argument here through which
    a lookalike seed-42 CSV can be passed. The content check is the second
    lock -- a file swapped in at the right path still has to carry `image_uid`
    and split to 562/120/118 or this raises.
    """
    path = root / SPLIT_CSV
    if not path.is_file():
        raise FileNotFoundError(f"Fixed split not found: {path}")
    with path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        fields = set(reader.fieldnames or [])
        if "image_uid" not in fields:
            raise ValueError(
                f"{path} has no image_uid column -- this is not the fixed split. "
                "Keying on image_id instead collides on 150 images."
            )
        splits = {row["image_uid"]: row["split"] for row in reader}

    counts = {name: sum(1 for value in splits.values() if value == name)
              for name in EXPECTED_COUNTS}
    if counts != EXPECTED_COUNTS:
        raise ValueError(
            f"{path} splits {counts}, expected {EXPECTED_COUNTS}. A different "
            "seed-42 draw has been substituted at the canonical path; refusing "
            "to continue. Nothing downstream is safe with the wrong split."
        )
    return splits


def load_paths(csv_path: Path, path_column: str = "image_path") -> dict[str, str]:
    """`image_uid` -> path, from a manifest. **Paths only, never the split.**

    Raises on any CSV carrying a `split` column. A manifest that has one is a
    rival split file, and the whole point of routing path lookups through here
    is that its split must never be read -- so the safe move is to refuse the
    file rather than to quietly ignore the column and trust the next caller.
    """
    with csv_path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        fields = list(reader.fieldnames or [])
        if "split" in fields:
            raise ValueError(
                f"{csv_path} carries a 'split' column. Split comes from "
                f"{SPLIT_CSV} and nowhere else -- see load_split(). If you only "
                "want this file's paths, strip the column first, deliberately."
            )
        if path_column not in fields:
            raise ValueError(f"{csv_path} has no {path_column!r} column")
        return {image_uid(row[path_column]): row[path_column] for row in reader}


def verify(source: Path, root: Path, dataset_id: str = "new_remote_sensing_dataset") -> dict:
    """Round-trip every discovered image through an export id and back.

    Checks the three things that would make the join wrong rather than merely
    absent: ids must be unique, uids must be unique (the collision), and every
    uid must exist in the fixed split.
    """
    paths = discover(source)
    splits = load_split(root)

    ids, uids, unknown, collisions = [], [], [], {}
    for index, path in enumerate(paths):
        identifier = f"{dataset_id}_{index:06d}_{path.stem}"
        _, parsed_index, parsed_stem = parse_image_id(identifier)
        assert parsed_index == index and parsed_stem == path.stem, identifier
        uid = image_uid(path)
        ids.append(identifier)
        uids.append(uid)
        collisions.setdefault(path.stem, set()).add(uid)
        if uid not in splits:
            unknown.append(uid)

    ambiguous = {stem: sorted(found) for stem, found in collisions.items() if len(found) > 1}
    return {
        "images": len(paths),
        "unique_ids": len(set(ids)),
        "unique_uids": len(set(uids)),
        "unknown_uids": unknown,
        "stems_colliding": ambiguous,
        "by_split": {
            name: sum(1 for uid in uids if splits.get(uid) == name)
            for name in ("train", "val", "test")
        },
    }


def compare_manifest_split(manifest_csv: Path, root: Path) -> dict:
    """How far a rival manifest's split drifts from the fixed split.

    Not a diagnostic for the manifest's quality -- both are valid seed-42 draws
    of the same 800 images, they are simply *different* draws. The point is
    that they cannot be mixed.

    This is the one place that reads a foreign `split` column, and it reads it
    with raw csv rather than through `load_paths` on purpose: measuring the
    disagreement is exactly the job, so the guard would be in the way. Nothing
    it returns feeds the pipeline -- it is a report.
    """
    fixed = load_split(root)
    with manifest_csv.open(encoding="utf-8", newline="") as file:
        manifest = {image_uid(row["image_path"]): row["split"] for row in csv.DictReader(file)}

    disagreements: dict[tuple[str, str], int] = {}
    for uid, split in manifest.items():
        if uid in fixed and fixed[uid] != split:
            disagreements[fixed[uid], split] = disagreements.get((fixed[uid], split), 0) + 1
    return {
        "rows": len(manifest),
        "uids_match": set(manifest) == set(fixed),
        "agree": sum(1 for uid, split in manifest.items() if fixed.get(uid) == split),
        "disagree": sum(disagreements.values()),
        "disagreements": disagreements,
        # The dangerous one: fixed-test images the manifest would hand over as
        # train or val, i.e. straight into the case memory.
        "fixed_test_leaked": sum(
            count for (fixed_split, _), count in disagreements.items() if fixed_split == "test"
        ),
    }


def _self_check() -> None:
    assert parse_image_id("new_remote_sensing_dataset_000003_004") == (
        "new_remote_sensing_dataset", 3, "004")
    # A dataset id containing digits and underscores must not break the anchor.
    assert parse_image_id("ds_v2_000042_img_7") == ("ds_v2", 42, "img_7")
    for bad in ("no_index_here", "ds_0003_004"):
        try:
            parse_image_id(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted {bad!r}")

    # The collision this module exists for: same stem, different uid.
    positive = image_uid("NWPU VHR-10 dataset/positive image set/004.jpg")
    negative = image_uid(r"NWPU VHR-10 dataset\negative image set\004.jpg")
    assert positive == "nwpu_positive_004" and negative == "nwpu_background_004"
    assert positive != negative, "stem alone must never be the key"
    try:
        image_uid("some/other/place/004.jpg")
    except ValueError:
        pass
    else:
        raise AssertionError("accepted a path with no positive/negative marker")
    # --- the split guard ---------------------------------------------------
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        fake_root = Path(directory)
        (fake_root / SPLIT_CSV.parent).mkdir(parents=True)

        def write(path: Path, header: str, rows: list[str]) -> None:
            path.write_text("\n".join([header, *rows]) + "\n", encoding="utf-8")

        # A missing split file is an error, never an empty dict.
        try:
            load_split(fake_root)
        except FileNotFoundError:
            pass
        else:
            raise AssertionError("accepted a missing split file")

        # Right path, right column, WRONG draw -- this is the manifest's
        # failure mode and the content check is the only thing that sees it.
        canonical = fake_root / SPLIT_CSV
        write(canonical, "image_uid,split",
              [f"nwpu_positive_{i:03d},train" for i in range(560)]
              + [f"nwpu_positive_{i:03d},val" for i in range(560, 680)]
              + [f"nwpu_positive_{i:03d},test" for i in range(680, 800)])
        try:
            load_split(fake_root)
        except ValueError as error:
            assert "562" in str(error), error
        else:
            raise AssertionError("accepted a 560/120/120 draw at the canonical path")

        # Right path, but keyed on the colliding image_id instead of image_uid.
        write(canonical, "image_id,split", ["004,train"])
        try:
            load_split(fake_root)
        except ValueError as error:
            assert "image_uid" in str(error), error
        else:
            raise AssertionError("accepted a file with no image_uid column")

        # The correct file passes both locks.
        write(canonical, "image_uid,split",
              [f"nwpu_positive_{i:03d},train" for i in range(562)]
              + [f"nwpu_positive_{i:03d},val" for i in range(562, 682)]
              + [f"nwpu_positive_{i:03d},test" for i in range(682, 800)])
        assert len(load_split(fake_root)) == 800

        # A manifest must not get in through the path-reading door either.
        manifest = fake_root / "manifest.csv"
        write(manifest, "image_id,image_path,split",
              [r"004,NWPU VHR-10 dataset\positive image set\004.jpg,train"])
        try:
            load_paths(manifest)
        except ValueError as error:
            assert "split" in str(error), error
        else:
            raise AssertionError("read paths from a file carrying a split column")

        # Strip the column and the same file is fine -- the guard blocks the
        # rival authority, not the paths.
        write(manifest, "image_id,image_path",
              [r"004,NWPU VHR-10 dataset\positive image set\004.jpg"])
        assert load_paths(manifest) == {
            "nwpu_positive_004": r"NWPU VHR-10 dataset\positive image set\004.jpg"}

    print("export-join self-check OK")


if __name__ == "__main__":
    _self_check()

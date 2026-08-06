from pathlib import Path

from PIL import Image

from src.data.manifests import load_split_manifest
from src.data.models import DatasetItem
from src.data.splitting import SplitConfig, save_split_manifest


def _image(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (10, 10)).save(path)
    return path


def test_manifest_round_trip_preserves_backgrounds_and_unique_ids(tmp_path):
    splits = {"train": [], "val": [], "test": []}
    for split in splits:
        positive = _image(tmp_path / split / "positive" / "001.jpg")
        annotation = tmp_path / split / "annotations" / "001.txt"
        annotation.parent.mkdir(parents=True, exist_ok=True)
        annotation.write_text("(1,1),(5,5),1\n", encoding="utf-8")
        background = _image(tmp_path / split / "background" / "001.jpg")
        splits[split] = [
            DatasetItem(positive, annotation, False),
            DatasetItem(background, None, True),
        ]

    manifest = tmp_path / "manifest.csv"
    save_split_manifest(
        splits,
        manifest,
        tmp_path,
        SplitConfig(0.70, 0.15, 0.15, 42),
    )
    loaded = load_split_manifest(manifest, tmp_path)

    assert {len(items) for items in loaded.values()} == {2}
    assert all(sum(item.is_background for item in items) == 1 for items in loaded.values())
    manifest_text = manifest.read_text(encoding="utf-8")
    assert "positive_001" in manifest_text
    assert "background_001" in manifest_text

from PIL import Image

from src.data.conversion import convert_pascal_voc_to_yolo


def test_pascal_voc_to_yolo_conversion(tmp_path):
    dataset_root = tmp_path / "hrrsd"
    image_root = dataset_root / "JPEGImages"
    annotation_root = dataset_root / "Annotations"
    image_root.mkdir(parents=True)
    annotation_root.mkdir()
    Image.new("RGB", (100, 50)).save(image_root / "sample.jpg")
    (annotation_root / "sample.xml").write_text(
        """<annotation>
        <object><name>storage tank</name><bndbox>
          <xmin>10</xmin><ymin>5</ymin><xmax>50</xmax><ymax>25</ymax>
        </bndbox></object>
        <object><name>unsupported</name><bndbox>
          <xmin>1</xmin><ymin>1</ymin><xmax>2</xmax><ymax>2</ymax>
        </bndbox></object>
        </annotation>""",
        encoding="utf-8",
    )
    config = {
        "dataset": {"root": str(dataset_root)},
        "source": {
            "images": "JPEGImages",
            "annotations": "Annotations",
            "annotation_format": "pascal_voc",
            "image_extensions": [".jpg"],
        },
        "output": {
            "annotation_format": "yolo",
            "labels": "labels_nwpu",
            "overwrite": True,
        },
        "class_mapping": {"storage tank": 2},
        "conversion": {
            "clamp_boxes": True,
            "create_empty_labels": True,
            "fail_on_missing_annotation": True,
            "fail_on_invalid_box": True,
        },
    }

    summary = convert_pascal_voc_to_yolo(config)

    label = (dataset_root / "labels_nwpu" / "sample.txt").read_text().strip()
    assert label == "2 0.30000000 0.30000000 0.40000000 0.40000000"
    assert summary["images"] == 1
    assert summary["objects"] == 1
    assert summary["class_counts"] == {2: 1}

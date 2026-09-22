from __future__ import annotations

import argparse
from pathlib import Path

from src.data.conversion import convert_pascal_voc_to_yolo
from src.detection.config import load_yaml_config
from src.utils.job_logging import logged_cli


@logged_cli("rs-convert-dataset")
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert configured dataset annotations to YOLO format."
    )
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    summary = convert_pascal_voc_to_yolo(load_yaml_config(args.config))
    print("Dataset conversion completed")
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()

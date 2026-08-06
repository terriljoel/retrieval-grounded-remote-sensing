from __future__ import annotations

import argparse
from pathlib import Path

from src.detection.config import load_detection_config
from src.detection.ultralytics_backend import train_detector
from src.utils.job_logging import logged_cli
from src.utils.paths import find_project_root


@logged_cli("rs-train-detector")
def main() -> None:
    parser = argparse.ArgumentParser(description="Train a configured detector.")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    project_root = find_project_root(Path(__file__).resolve())
    config = load_detection_config(args.config)
    best_checkpoint = train_detector(config, project_root)
    print(f"Best checkpoint: {best_checkpoint}")


if __name__ == "__main__":
    main()

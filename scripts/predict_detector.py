from __future__ import annotations

import argparse
from pathlib import Path

from src.detection.config import load_detection_config
from src.detection.ultralytics_backend import predict_images
from src.utils.paths import find_project_root


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict objects in arbitrary images.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source", required=True, help="Image, directory, glob, or URL.")
    args = parser.parse_args()

    project_root = find_project_root(Path(__file__).resolve())
    config = load_detection_config(args.config)
    run_directory = predict_images(
        config, project_root, args.checkpoint, args.source
    )
    print(f"Prediction outputs: {run_directory}")


if __name__ == "__main__":
    main()

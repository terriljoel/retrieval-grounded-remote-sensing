from __future__ import annotations

import argparse
from pathlib import Path

from src.detection.inference_config import load_inference_config
from src.detection.ultralytics_backend import export_predictions
from src.utils.job_logging import logged_cli
from src.utils.paths import find_project_root


@logged_cli("rs-export-predictions")
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export structured detector predictions for an image source."
    )
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    project_root = find_project_root(Path(__file__).resolve())
    config = load_inference_config(args.config)
    run_directory = export_predictions(config, project_root)
    print(f"Inference export: {run_directory}")


if __name__ == "__main__":
    main()

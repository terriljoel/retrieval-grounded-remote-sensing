from __future__ import annotations

import argparse
from pathlib import Path

from src.detection.config import load_detection_config
from src.detection.ultralytics_backend import evaluate_detector
from src.utils.paths import find_project_root


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a detector checkpoint.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--confirm-test",
        action="store_true",
        help="Required when evaluation.split=test.",
    )
    args = parser.parse_args()

    project_root = find_project_root(Path(__file__).resolve())
    config = load_detection_config(args.config)
    split = config["evaluation"]["split"]
    if split == "test" and not args.confirm_test:
        parser.error(
            "evaluation.split=test requires --confirm-test to protect the held-out set"
        )

    run_directory = evaluate_detector(config, project_root, args.checkpoint)
    print(f"Evaluation outputs: {run_directory}")


if __name__ == "__main__":
    main()

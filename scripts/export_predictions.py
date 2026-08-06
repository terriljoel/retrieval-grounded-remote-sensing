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
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="One image or a directory containing images.",
    )
    parser.add_argument(
        "--dataset-id",
        required=True,
        help="Stable provenance name for the source dataset.",
    )
    parser.add_argument(
        "--source-split",
        help="Partition provenance, for example val, test, or external.",
    )
    parser.add_argument(
        "--confirm-test",
        action="store_true",
        help="Required when exporting held-out test predictions.",
    )
    args = parser.parse_args()

    project_root = find_project_root(Path(__file__).resolve())
    config = load_inference_config(args.config)
    source_split = (
        args.source_split or config["inference"]["default_source_split"]
    ).strip().lower()
    if source_split == "test" and not args.confirm_test:
        parser.error(
            "source-split=test requires --confirm-test to protect the held-out set"
        )

    run_directory = export_predictions(
        config,
        project_root,
        args.checkpoint,
        args.source,
        args.dataset_id,
        source_split,
    )
    print(f"Inference export: {run_directory}")


if __name__ == "__main__":
    main()

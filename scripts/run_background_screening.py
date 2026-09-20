from __future__ import annotations

import argparse
from pathlib import Path

from src.evaluation.background_screening import run_background_screening_experiment
from src.detection.config import load_yaml_config
from src.utils.job_logging import logged_cli
from src.utils.paths import find_project_root


@logged_cli("rs-run-background-screening")
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Screen zero-detection images for possible missed target objects."
    )
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    project_root = find_project_root(Path(__file__).resolve())
    config = load_yaml_config(args.config)
    output = run_background_screening_experiment(config, project_root)
    print(f"Background-screening experiment: {output}")


if __name__ == "__main__":
    main()

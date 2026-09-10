from __future__ import annotations

import argparse
from pathlib import Path

from src.evaluation.assistance import run_assistance_experiment
from src.evaluation.config import load_assistance_experiment_config
from src.utils.job_logging import logged_cli
from src.utils.paths import find_project_root


@logged_cli("rs-run-assistance-experiment")
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a resumable detector/retrieval/VLM comparison."
    )
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    project_root = find_project_root(Path(__file__).resolve())
    config = load_assistance_experiment_config(args.config)
    output = run_assistance_experiment(config, project_root)
    print(f"Assistance experiment: {output}")


if __name__ == "__main__":
    main()


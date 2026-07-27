from pathlib import Path
from typing import Any

import yaml


def load_dataset_config(
    config_name: str,
    project_root: Path,
) -> dict[str, Any]:
    config_path = (
        project_root
        / "configs"
        / "datasets"
        / f"{config_name}.yaml"
    )

    if not config_path.is_file():
        raise FileNotFoundError(
            f"Dataset configuration not found: {config_path}"
        )

    with config_path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)

    required_sections = {"dataset", "classes", "extensions"}
    missing = required_sections - config.keys()

    if missing:
        raise ValueError(
            f"Missing configuration sections: {sorted(missing)}"
        )

    return config
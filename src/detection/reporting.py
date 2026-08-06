from __future__ import annotations

from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any


def collect_runtime_metadata(project_root: Path) -> dict[str, Any]:
    packages = {}
    for package in ("ultralytics", "torch", "numpy", "Pillow", "PyYAML"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None

    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None

    cuda = {"available": False, "device": None}
    try:
        import torch

        cuda["available"] = torch.cuda.is_available()
        if cuda["available"]:
            cuda["device"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass

    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": commit,
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "cuda": cuda,
        "job_id": os.environ.get("RS_JOB_ID"),
        "job_directory": os.environ.get("RS_JOB_DIRECTORY"),
    }


def write_json(data: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return path

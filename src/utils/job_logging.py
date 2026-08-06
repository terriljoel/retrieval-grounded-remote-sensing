from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from functools import wraps
import json
import os
from pathlib import Path
import platform
import sys
import time
import traceback
from typing import Any, Callable, TextIO, TypeVar
from uuid import uuid4


T = TypeVar("T")


class _Tee:
    def __init__(self, primary: TextIO, log_file: TextIO):
        self.primary = primary
        self.log_file = log_file

    def write(self, text: str) -> int:
        primary_result = self.primary.write(text)
        self.log_file.write(text)
        return primary_result

    def flush(self) -> None:
        self.primary.flush()
        self.log_file.flush()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.primary, name)


def _job_log_root() -> Path | None:
    explicit = os.environ.get("JOB_LOG_ROOT")
    if explicit:
        return Path(explicit).expanduser().resolve()
    experiment_root = os.environ.get("EXPERIMENT_OUTPUT_ROOT")
    if experiment_root:
        return (Path(experiment_root).expanduser() / "job_logs").resolve()
    return None


def _write_metadata(path: Path, metadata: dict[str, Any]) -> None:
    path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def run_logged_job(command_name: str, function: Callable[[], T]) -> T:
    root = _job_log_root()
    if root is None:
        print(
            "[job] Logging disabled because JOB_LOG_ROOT and "
            "EXPERIMENT_OUTPUT_ROOT are not set.",
            file=sys.stderr,
        )
        return function()

    started_at = datetime.now(timezone.utc)
    started_timer = time.monotonic()
    timestamp = started_at.strftime("%Y%m%dT%H%M%SZ")
    job_id = f"{timestamp}_{uuid4().hex[:8]}"
    job_directory = root / command_name / job_id
    job_directory.mkdir(parents=True, exist_ok=False)
    log_path = job_directory / "run.log"
    metadata_path = job_directory / "job.json"
    metadata: dict[str, Any] = {
        "job_id": job_id,
        "command_name": command_name,
        "command": [command_name, *sys.argv[1:]],
        "status": "running",
        "exit_code": None,
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": None,
        "duration_seconds": None,
        "working_directory": str(Path.cwd()),
        "python": sys.version,
        "platform": platform.platform(),
        "log_path": str(log_path),
    }
    _write_metadata(metadata_path, metadata)

    previous_job_id = os.environ.get("RS_JOB_ID")
    previous_job_directory = os.environ.get("RS_JOB_DIRECTORY")
    os.environ["RS_JOB_ID"] = job_id
    os.environ["RS_JOB_DIRECTORY"] = str(job_directory)

    def finish(status: str, exit_code: int) -> None:
        finished_at = datetime.now(timezone.utc)
        metadata.update({
            "status": status,
            "exit_code": exit_code,
            "finished_at_utc": finished_at.isoformat(),
            "duration_seconds": round(time.monotonic() - started_timer, 6),
        })
        _write_metadata(metadata_path, metadata)
        print(f"[job] status={status} log={log_path}")

    try:
        with log_path.open("w", encoding="utf-8", buffering=1) as log_file:
            stdout = _Tee(sys.stdout, log_file)
            stderr = _Tee(sys.stderr, log_file)
            with redirect_stdout(stdout), redirect_stderr(stderr):
                print(f"[job] id={job_id}")
                print(f"[job] directory={job_directory}")
                try:
                    result = function()
                except SystemExit as error:
                    exit_code = error.code if isinstance(error.code, int) else 1
                    finish("succeeded" if exit_code == 0 else "failed", exit_code)
                    raise
                except KeyboardInterrupt:
                    print("[job] interrupted by user", file=sys.stderr)
                    finish("interrupted", 130)
                    raise
                except Exception:
                    traceback.print_exc()
                    finish("failed", 1)
                    raise SystemExit(1) from None
                else:
                    finish("succeeded", 0)
                    return result
    finally:
        if previous_job_id is None:
            os.environ.pop("RS_JOB_ID", None)
        else:
            os.environ["RS_JOB_ID"] = previous_job_id
        if previous_job_directory is None:
            os.environ.pop("RS_JOB_DIRECTORY", None)
        else:
            os.environ["RS_JOB_DIRECTORY"] = previous_job_directory


def logged_cli(command_name: str) -> Callable[[Callable[[], T]], Callable[[], T]]:
    def decorator(function: Callable[[], T]) -> Callable[[], T]:
        @wraps(function)
        def wrapper() -> T:
            return run_logged_job(command_name, function)

        return wrapper

    return decorator

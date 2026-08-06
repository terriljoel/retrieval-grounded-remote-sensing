import json
import sys

import pytest

from src.utils.job_logging import run_logged_job


def _only_job_directory(root, command_name):
    directories = list((root / command_name).iterdir())
    assert len(directories) == 1
    return directories[0]


def test_run_logged_job_tees_output_and_records_success(
    monkeypatch, tmp_path, capsys
):
    monkeypatch.setenv("JOB_LOG_ROOT", str(tmp_path))
    monkeypatch.setattr(sys, "argv", ["rs-example", "--value", "42"])

    result = run_logged_job("rs-example", lambda: print("job output"))

    assert result is None
    job_directory = _only_job_directory(tmp_path, "rs-example")
    metadata = json.loads(
        (job_directory / "job.json").read_text(encoding="utf-8")
    )
    log = (job_directory / "run.log").read_text(encoding="utf-8")
    assert metadata["status"] == "succeeded"
    assert metadata["exit_code"] == 0
    assert metadata["command"] == ["rs-example", "--value", "42"]
    assert "job output" in log
    assert "job output" in capsys.readouterr().out
    assert "RS_JOB_ID" not in __import__("os").environ


def test_run_logged_job_records_failure_and_traceback(monkeypatch, tmp_path):
    monkeypatch.setenv("JOB_LOG_ROOT", str(tmp_path))
    monkeypatch.setattr(sys, "argv", ["rs-failing"])

    def fail():
        print("before failure")
        raise ValueError("deliberate failure")

    with pytest.raises(SystemExit, match="1"):
        run_logged_job("rs-failing", fail)

    job_directory = _only_job_directory(tmp_path, "rs-failing")
    metadata = json.loads(
        (job_directory / "job.json").read_text(encoding="utf-8")
    )
    log = (job_directory / "run.log").read_text(encoding="utf-8")
    assert metadata["status"] == "failed"
    assert metadata["exit_code"] == 1
    assert "before failure" in log
    assert "ValueError: deliberate failure" in log

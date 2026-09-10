from __future__ import annotations

import copy

import pytest

from src.annotation.models import NWPU_CLASSES
from src.evaluation.config import validate_assistance_experiment_config


def valid_config():
    return {
        "experiment": {},
        "cases": {
            "inference_directory": "export",
            "statuses": ["true_positive", "false_positive", "class_error"],
            "maximum_per_status": 5,
        },
        "classes": list(NWPU_CLASSES),
        "embedding": {},
        "retrieval": {
            "artifact_root": "embeddings", "evidence_splits": ["train"], "top_k": 5
        },
        "vlm": {"enabled": True, "max_evidence": 3},
        "decision_policy": {},
        "execution": {
            "variants": ["retrieval_grounded_vlm", "configured_policy"],
        },
    }


def test_valid_comparison_config():
    validate_assistance_experiment_config(valid_config())


def test_policy_requires_grounded_variant():
    config = copy.deepcopy(valid_config())
    config["execution"]["variants"] = ["configured_policy"]
    with pytest.raises(ValueError, match="requires retrieval_grounded_vlm"):
        validate_assistance_experiment_config(config)

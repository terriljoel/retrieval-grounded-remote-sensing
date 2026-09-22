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


def test_request_rate_must_be_positive():
    config = copy.deepcopy(valid_config())
    config["vlm"]["requests_per_minute"] = 0
    with pytest.raises(ValueError, match="requests_per_minute must be positive"):
        validate_assistance_experiment_config(config)


def test_low_confidence_rescue_requires_query_only_vlm():
    config = copy.deepcopy(valid_config())
    config["decision_policy"]["low_confidence_rescue"] = {
        "enabled": True,
        "detector_min_confidence": 0.2,
        "detector_max_confidence": 0.5,
        "retrieval_min_cosine_similarity": 0.92,
        "retrieval_min_supporting_neighbors": 3,
        "query_vlm_min_confidence": 0.9,
        "grounded_vlm_min_confidence": 0.9,
    }
    with pytest.raises(ValueError, match="requires configured_policy and query_only_vlm"):
        validate_assistance_experiment_config(config)

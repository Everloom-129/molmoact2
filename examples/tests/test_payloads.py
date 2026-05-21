"""Wire-format tests for the embodiment payload extractors.

These confirm that `_predict_from_payload` in each server file:

  * reads the right field names from the request body
  * forwards them to `policy.predict(...)` with the correct kwargs
  * honours num_steps defaulting
  * ignores any client-supplied enable_cuda_graph (sticky deploy-time flag)

We load each server module via importlib so we don't need to add their
directories to sys.path globally. Each module's `BasePolicy.__init__` is
never invoked — we hand the extractor a hand-rolled `_StubPolicy`.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from types import ModuleType
from typing import Any

import numpy as np
import pytest


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _load_server_module(path: str, name: str) -> ModuleType:
    """Import a server file without colliding with sibling-module names."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class _StubPolicy:
    """Captures predict() args; no model, no GPU."""

    def __init__(self, embodiment: str) -> None:
        self.EMBODIMENT = embodiment
        self.STATE_DIM = 8 if embodiment == "droid" else 14
        self.default_cuda_graph = False
        self.calls: list[dict[str, Any]] = []

    def predict(self, **kwargs: Any) -> np.ndarray:
        self.calls.append(kwargs)
        # Shape mirrors the embodiment's action dim; only used by callers that
        # care, which the extractor tests don't.
        return np.zeros((10, self.STATE_DIM), dtype=np.float32)


# ---------------------------------------------------------------------------
# DROID
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def droid_mod() -> ModuleType:
    return _load_server_module(
        os.path.join(REPO_ROOT, "examples", "droid", "host_server_droid.py"),
        "_test_host_server_droid",
    )


def test_droid_payload_extracts_expected_fields(droid_mod):
    stub = _StubPolicy("droid")
    payload = {
        "external_cam": np.zeros((4, 4, 3), dtype=np.uint8),
        "wrist_cam": np.full((4, 4, 3), 7, dtype=np.uint8),
        "instruction": "stack the blocks",
        "state": np.arange(8, dtype=np.float32),
    }
    droid_mod._predict_from_payload(stub, payload)
    assert len(stub.calls) == 1
    call = stub.calls[0]
    assert call["instruction"] == "stack the blocks"
    assert call["state"].shape == (8,)
    assert call["num_steps"] == droid_mod.DEFAULT_NUM_STEPS
    # wrist_cam content survived round-trip (smoke check).
    assert int(call["wrist_cam"][0, 0, 0]) == 7
    # CUDA-graph flag follows the policy default, not any client field.
    assert call["enable_cuda_graph"] is False


def test_droid_payload_honours_num_steps_override(droid_mod):
    stub = _StubPolicy("droid")
    payload = {
        "external_cam": np.zeros((4, 4, 3), dtype=np.uint8),
        "wrist_cam": np.zeros((4, 4, 3), dtype=np.uint8),
        "instruction": "x",
        "state": np.zeros(8, dtype=np.float32),
        "num_steps": 4,
    }
    droid_mod._predict_from_payload(stub, payload)
    assert stub.calls[0]["num_steps"] == 4


def test_droid_payload_ignores_client_cuda_graph(droid_mod):
    """Sticky-flag policy: per-request override is dropped on purpose."""
    stub = _StubPolicy("droid")
    stub.default_cuda_graph = False
    payload = {
        "external_cam": np.zeros((4, 4, 3), dtype=np.uint8),
        "wrist_cam": np.zeros((4, 4, 3), dtype=np.uint8),
        "instruction": "x",
        "state": np.zeros(8, dtype=np.float32),
        "enable_cuda_graph": True,  # should be ignored
    }
    droid_mod._predict_from_payload(stub, payload)
    assert stub.calls[0]["enable_cuda_graph"] is False


def test_droid_payload_missing_field_raises_keyerror(droid_mod):
    stub = _StubPolicy("droid")
    payload = {"instruction": "x"}  # no cameras, no state
    with pytest.raises(KeyError):
        droid_mod._predict_from_payload(stub, payload)


# ---------------------------------------------------------------------------
# YAM
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def yam_mod() -> ModuleType:
    return _load_server_module(
        os.path.join(REPO_ROOT, "examples", "yam", "host_server_yam.py"),
        "_test_host_server_yam",
    )


def test_yam_payload_extracts_three_cameras_in_order(yam_mod):
    stub = _StubPolicy("yam")
    payload = {
        "top_cam": np.full((4, 4, 3), 1, dtype=np.uint8),
        "left_cam": np.full((4, 4, 3), 2, dtype=np.uint8),
        "right_cam": np.full((4, 4, 3), 3, dtype=np.uint8),
        "instruction": "hand the cup",
        "state": np.arange(14, dtype=np.float32),
    }
    yam_mod._predict_from_payload(stub, payload)
    assert len(stub.calls) == 1
    call = stub.calls[0]
    assert int(call["top_cam"][0, 0, 0]) == 1
    assert int(call["left_cam"][0, 0, 0]) == 2
    assert int(call["right_cam"][0, 0, 0]) == 3
    assert call["state"].shape == (14,)
    assert call["instruction"] == "hand the cup"
    assert call["num_steps"] == yam_mod.DEFAULT_NUM_STEPS
    assert call["enable_cuda_graph"] is False


def test_yam_payload_honours_num_steps_override(yam_mod):
    stub = _StubPolicy("yam")
    payload = {
        "top_cam": np.zeros((4, 4, 3), dtype=np.uint8),
        "left_cam": np.zeros((4, 4, 3), dtype=np.uint8),
        "right_cam": np.zeros((4, 4, 3), dtype=np.uint8),
        "instruction": "x",
        "state": np.zeros(14, dtype=np.float32),
        "num_steps": 6,
    }
    yam_mod._predict_from_payload(stub, payload)
    assert stub.calls[0]["num_steps"] == 6


# ---------------------------------------------------------------------------
# Subclass contracts (`BasePolicy` constants set correctly)
# ---------------------------------------------------------------------------

def test_droid_policy_constants(droid_mod):
    P = droid_mod.DroidPolicy
    assert P.EMBODIMENT == "droid"
    assert P.STATE_DIM == 8
    assert P.MODE_KWARG == "action_mode"
    assert P.NORM_TAG == "franka_droid"


def test_yam_policy_constants(yam_mod):
    P = yam_mod.YamPolicy
    assert P.EMBODIMENT == "yam"
    assert P.STATE_DIM == 14
    assert P.MODE_KWARG == "inference_action_mode"
    assert P.NORM_TAG == "yam_dual_molmoact2"

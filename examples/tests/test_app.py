"""Integration tests for the shared FastAPI app factory.

We never call `BasePolicy.__init__` (which would snapshot_download and load
the real model). Instead we hand `build_app` a `FakePolicy` that satisfies the
attribute contract, plus a `predict_from_payload` callable that just returns a
canned ndarray. That lets us exercise:

  * the GET /act health response shape (revision, norm_tag, device, dtype)
  * /healthz
  * /metrics text exposition contains our counter names
  * POST /act returns json_numpy-decoded actions + dt_ms
  * POST /act error paths (bad json, missing fields, bad state shape) each
    bump the right counter
  * POST /act dispatches to a thread, so /healthz stays responsive while
    inference is "running" (regression test for the async fix)
"""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace
from typing import Any

import json_numpy
import numpy as np
import pytest
from httpx import ASGITransport, AsyncClient

from _common.server import build_app

json_numpy.patch()


class FakePolicy:
    """Minimal duck-type of BasePolicy that build_app needs.

    No model, no GPU, no snapshot dir — just the attributes the route handlers
    read.
    """

    EMBODIMENT = "fake"
    REPO_ID = "test/fake"
    NORM_TAG = "fake_norm"
    STATE_DIM = 4

    def __init__(self) -> None:
        self.revision = "fakerevsha"
        self.device = "cpu"
        self.default_cuda_graph = False
        self.model = SimpleNamespace(dtype="torch.bfloat16")


def _build_test_app(predict_from_payload):
    policy = FakePolicy()
    return build_app(
        policy=policy,
        title="Fake server",
        health_extra={"num_cameras": 2, "state_dim": policy.STATE_DIM},
        predict_from_payload=predict_from_payload,
        default_num_steps=10,
    )


def _make_client(app) -> AsyncClient:
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


# ---------------------------------------------------------------------------
# Health / liveness / metrics surface
# ---------------------------------------------------------------------------

async def test_get_act_returns_health_with_revision():
    app = _build_test_app(lambda payload: np.zeros((10, 8), dtype=np.float32))
    async with _make_client(app) as client:
        r = await client.get("/act")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert body["repo_id"] == "test/fake"
        assert body["revision"] == "fakerevsha"
        assert body["norm_tag"] == "fake_norm"
        assert body["device"] == "cpu"
        assert body["num_cameras"] == 2
        assert body["state_dim"] == 4


async def test_healthz_is_trivial():
    app = _build_test_app(lambda payload: np.zeros((10, 8), dtype=np.float32))
    async with _make_client(app) as client:
        r = await client.get("/healthz")
        assert r.status_code == 200
        assert r.json() == {"status": "ok"}


async def test_metrics_exposes_our_counter_names():
    app = _build_test_app(lambda payload: np.zeros((10, 8), dtype=np.float32))
    async with _make_client(app) as client:
        r = await client.get("/metrics")
        assert r.status_code == 200
        text = r.text
        assert "molmoact2_inference_requests_total" in text
        assert "molmoact2_inference_errors_total" in text
        assert "molmoact2_inference_latency_seconds" in text


# ---------------------------------------------------------------------------
# POST /act — happy path
# ---------------------------------------------------------------------------

async def test_post_act_returns_actions_and_dt_ms():
    captured: dict[str, Any] = {}

    def predict(payload):
        captured.update(payload)
        return np.arange(80, dtype=np.float32).reshape(10, 8)

    app = _build_test_app(predict)
    payload = {
        "external_cam": np.zeros((4, 4, 3), dtype=np.uint8),
        "wrist_cam": np.zeros((4, 4, 3), dtype=np.uint8),
        "instruction": "pick up the cube",
        "state": np.zeros(8, dtype=np.float32),
    }
    body = json_numpy.dumps(payload)

    async with _make_client(app) as client:
        r = await client.post("/act", content=body)
        assert r.status_code == 200
        decoded = json_numpy.loads(r.text)
        assert decoded["actions"].shape == (10, 8)
        assert decoded["actions"].dtype == np.float32
        assert isinstance(decoded["dt_ms"], float)
        assert decoded["dt_ms"] >= 0.0

    # The predict callable saw exactly the fields we sent (cameras + state
    # are ndarrays after json_numpy round-trip, so check shape rather than
    # identity).
    assert captured["instruction"] == "pick up the cube"
    assert captured["state"].shape == (8,)


# ---------------------------------------------------------------------------
# POST /act — error paths
# ---------------------------------------------------------------------------

async def test_post_act_bad_json_returns_400():
    app = _build_test_app(lambda payload: np.zeros((10, 8), dtype=np.float32))
    async with _make_client(app) as client:
        r = await client.post("/act", content=b"not json at all")
        assert r.status_code == 400
        assert "decode" in r.json()["error"].lower()


async def test_post_act_missing_field_returns_400():
    def predict(payload):
        # mimic real extractor: KeyError on missing required field
        return payload["external_cam"]  # noqa: F841

    app = _build_test_app(predict)
    body = json_numpy.dumps({"instruction": "x"})
    async with _make_client(app) as client:
        r = await client.post("/act", content=body)
        assert r.status_code == 400
        assert "missing required field" in r.json()["error"]


async def test_post_act_bad_shape_returns_400():
    def predict(payload):
        raise ValueError("state must be shape (8,), got (3,)")

    app = _build_test_app(predict)
    body = json_numpy.dumps({"x": 1})
    async with _make_client(app) as client:
        r = await client.post("/act", content=body)
        assert r.status_code == 400
        assert "state must be shape" in r.json()["error"]


async def test_post_act_internal_error_returns_500():
    def predict(payload):
        raise RuntimeError("boom")

    app = _build_test_app(predict)
    body = json_numpy.dumps({"x": 1})
    async with _make_client(app) as client:
        r = await client.post("/act", content=body)
        assert r.status_code == 500
        assert "inference failed" in r.json()["error"]


# ---------------------------------------------------------------------------
# Async-dispatch regression: /healthz must respond while /act is busy.
# ---------------------------------------------------------------------------

async def test_act_does_not_block_event_loop():
    """Regression test for the async-dispatch fix.

    If POST /act ran inference on the event loop, then GET /healthz would have
    to wait the full predict duration. We sleep 400 ms inside predict and
    expect /healthz to come back well under that — under 100 ms is plenty,
    even with ASGI overhead.
    """

    def slow_predict(payload):
        time.sleep(0.4)
        return np.zeros((10, 8), dtype=np.float32)

    app = _build_test_app(slow_predict)
    body = json_numpy.dumps(
        {
            "external_cam": np.zeros((4, 4, 3), dtype=np.uint8),
            "wrist_cam": np.zeros((4, 4, 3), dtype=np.uint8),
            "instruction": "x",
            "state": np.zeros(8, dtype=np.float32),
        }
    )

    async with _make_client(app) as client:
        post_task = asyncio.create_task(client.post("/act", content=body))
        # Give the POST a moment to enter the threadpool.
        await asyncio.sleep(0.05)

        t0 = time.perf_counter()
        r = await client.get("/healthz")
        healthz_ms = (time.perf_counter() - t0) * 1000.0
        assert r.status_code == 200
        # /healthz should be fast; if predict blocks the loop we'd see ~350 ms.
        assert healthz_ms < 150.0, f"/healthz took {healthz_ms:.1f}ms; loop blocked?"

        post_resp = await post_task
        assert post_resp.status_code == 200


# ---------------------------------------------------------------------------
# Metrics counters move on real traffic.
# ---------------------------------------------------------------------------

@pytest.fixture
def fresh_metric_values():
    """Snapshot the labeled counter values before a test so we can assert
    deltas without depending on prior test runs (metrics are module-global)."""
    from _common.server import ERRORS, REQUESTS

    def snapshot():
        return {
            "ok": REQUESTS.labels(embodiment="fake", status="ok")._value.get(),
            "bad": REQUESTS.labels(embodiment="fake", status="bad_request")._value.get(),
            "decode_err": ERRORS.labels(embodiment="fake", kind="decode")._value.get(),
        }

    return snapshot


async def test_metrics_counters_increment(fresh_metric_values):
    before = fresh_metric_values()

    app = _build_test_app(lambda payload: np.zeros((10, 8), dtype=np.float32))
    body = json_numpy.dumps({"instruction": "x", "state": np.zeros(8, dtype=np.float32)})
    async with _make_client(app) as client:
        await client.post("/act", content=body)              # ok
        await client.post("/act", content=b"not json")        # decode err -> bad_request

    after = fresh_metric_values()
    assert after["ok"] == before["ok"] + 1
    assert after["bad"] == before["bad"] + 1
    assert after["decode_err"] == before["decode_err"] + 1

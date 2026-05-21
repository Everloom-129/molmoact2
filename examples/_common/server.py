"""Shared MolmoAct2 inference-server scaffolding.

Each `examples/<embodiment>/host_server_*.py` keeps embodiment-specific
constants (REPO_ID, NORM_TAG, STATE_DIM, camera order, the right
`*_action_mode` kwarg) and a tiny request-payload extractor, but delegates the
model load, upstream-patch dance, FastAPI route wiring, warmup, and metrics to
this module. The per-embodiment server is still the source of truth for its
own `/act` schema — we do not multiplex embodiments behind one process.

What lives here (and why):

  * `patch_modeling_for_bf16` — idempotent edits to upstream
    `modeling_molmoact2.py`; needed because the released checkpoints were not
    built with bf16 + local-snapshot loading in mind. See CLAUDE.md.
  * `to_pil` — defensive uint8 RGB conversion of incoming ndarrays.
  * `BasePolicy` — owns model + processor lifecycle, the per-instance
    `_move_inputs_to_device` cast hook, the inference lock, and the
    `predict_action` call. Subclasses set class-level constants and implement a
    thin `predict(...)` that arranges images / state and calls
    `self._run_predict_action`.
  * `build_app` — registers `/act` (GET health + POST inference), `/healthz`,
    and `/metrics`. The POST handler offloads the blocking inference call to a
    worker thread so `/healthz` and `/metrics` stay responsive mid-step.
  * Prometheus metrics — request count, error count, latency histogram, GPU
    memory gauges, all labeled by embodiment.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from typing import Any, Callable

import json_numpy
import numpy as np
import torch
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse, Response
from huggingface_hub import snapshot_download
from PIL import Image
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from transformers import AutoModelForImageTextToText, AutoProcessor


log = logging.getLogger("molmoact2.server")


# ---------------------------------------------------------------------------
# Prometheus metrics (module-level so multiple Policy instances in tests can
# share them; in production each server is its own process).
# ---------------------------------------------------------------------------

REQUESTS = Counter(
    "molmoact2_inference_requests_total",
    "Inference requests received, by embodiment and outcome.",
    labelnames=("embodiment", "status"),
)
ERRORS = Counter(
    "molmoact2_inference_errors_total",
    "Inference errors, by embodiment and error class.",
    labelnames=("embodiment", "kind"),
)
LATENCY = Histogram(
    "molmoact2_inference_latency_seconds",
    "End-to-end /act inference latency, by embodiment.",
    labelnames=("embodiment",),
    # 5 Hz robot loop ~ 200 ms budget; spread buckets across the realistic range.
    buckets=(0.02, 0.04, 0.06, 0.08, 0.10, 0.15, 0.20, 0.30, 0.50, 1.0, 2.0),
)
GPU_ALLOCATED = Gauge(
    "molmoact2_gpu_memory_allocated_bytes",
    "torch.cuda.memory_allocated, sampled on /metrics scrape.",
    labelnames=("embodiment", "device"),
)
GPU_RESERVED = Gauge(
    "molmoact2_gpu_memory_reserved_bytes",
    "torch.cuda.memory_reserved, sampled on /metrics scrape.",
    labelnames=("embodiment", "device"),
)


# ---------------------------------------------------------------------------
# Upstream patches and small helpers
# ---------------------------------------------------------------------------

def patch_modeling_for_bf16(local_dir: str) -> None:
    """Make upstream `modeling_molmoact2.py` survive bf16 inference.

    Two idempotent edits, both marked with `# patched_bf16_*` comments. The
    needles only match the DROID snapshot revision; on YAM both are already
    fixed upstream and we'll log "needle not found" — that's expected. Keeps
    the snapshot dir and the `trust_remote_code` cache copy in sync.
    """
    patches = [
        (
            "device=device,\n            dtype=torch.float32,\n            generator=generator,",
            "device=device,\n"
            "            dtype=source_tensor.dtype,  # patched_bf16_dtype\n"
            "            generator=generator,",
            "patched_bf16_dtype",
        ),
        (
            "return value.detach().cpu().numpy().astype(np.float32, copy=False)",
            "return value.detach().cpu().float().numpy().astype(np.float32, copy=False)  # patched_bf16_to_array",
            "patched_bf16_to_array",
        ),
    ]
    candidates = [os.path.join(local_dir, "modeling_molmoact2.py")]
    modules_root = os.path.expanduser(
        "~/.cache/huggingface/modules/transformers_modules"
    )
    if os.path.isdir(modules_root):
        for sub in os.listdir(modules_root):
            p = os.path.join(modules_root, sub, "modeling_molmoact2.py")
            if os.path.isfile(p):
                candidates.append(p)
    for path in candidates:
        try:
            with open(path, "r", encoding="utf-8") as f:
                src = f.read()
        except OSError:
            continue
        new_src = src
        applied: list[str] = []
        for needle, replacement, marker in patches:
            if marker in new_src:
                continue
            if needle not in new_src:
                log.warning("patch %s: needle not found in %s", marker, path)
                continue
            new_src = new_src.replace(needle, replacement, 1)
            applied.append(marker)
        if new_src != src:
            with open(path, "w", encoding="utf-8") as f:
                f.write(new_src)
            log.info("Applied patches %s in %s", applied, path)


def to_pil(arr: Any) -> Image.Image:
    if isinstance(arr, Image.Image):
        return arr.convert("RGB")
    a = np.asarray(arr)
    if a.ndim != 3 or a.shape[2] != 3:
        raise ValueError(f"image must be HxWx3, got shape {a.shape}")
    if a.dtype != np.uint8:
        a = np.clip(a, 0, 255).astype(np.uint8)
    return Image.fromarray(a, mode="RGB")


def error_response(status: int, message: str) -> Response:
    body = json_numpy.dumps({"error": message})
    return Response(content=body, status_code=status, media_type="application/json")


# ---------------------------------------------------------------------------
# BasePolicy
# ---------------------------------------------------------------------------

class BasePolicy:
    """Shared lifecycle for a MolmoAct2 inference policy.

    Subclasses set class-level constants and implement `predict(...)` that
    accepts the embodiment-specific payload and calls `_run_predict_action`.

    Subclass contract:
      * `EMBODIMENT`     — short label, used as a Prometheus label.
      * `REPO_ID`        — HF repo id to snapshot-download.
      * `NORM_TAG`       — passed to `predict_action`.
      * `STATE_DIM`      — expected flat state shape.
      * `MODE_KWARG`     — name of the predict_action mode kwarg this checkpoint
                           uses ("action_mode" on DROID; "inference_action_mode"
                           on YAM). Each server must use its checkpoint's name;
                           copying the call site verbatim across embodiments
                           will TypeError at warmup.
    """

    EMBODIMENT: str = "unknown"
    REPO_ID: str = ""
    NORM_TAG: str = ""
    STATE_DIM: int = 0
    MODE_KWARG: str = "action_mode"

    def __init__(
        self,
        repo_id: str | None,
        device: str,
        dtype: torch.dtype,
        enable_cuda_graph: bool = False,
    ) -> None:
        self.default_cuda_graph = enable_cuda_graph
        repo = repo_id or self.REPO_ID

        # `predict_action` reads `norm_stats.json` from `config._name_or_path`,
        # so we must load from the local snapshot dir, not a bare repo id.
        # snapshot_download is a no-op when files are cached.
        local_dir = snapshot_download(repo_id=repo)
        log.info("Resolved snapshot dir: %s", local_dir)
        self.local_dir = local_dir
        self.revision = os.path.basename(local_dir)

        patch_modeling_for_bf16(local_dir)

        log.info("Loading processor")
        # tokenizer_config.json ships `extra_special_tokens` as a list, but
        # transformers >=4.46 expects a dict. The model code only uses these
        # via `convert_tokens_to_ids`, so an empty dict is safe.
        self.processor = AutoProcessor.from_pretrained(
            local_dir, trust_remote_code=True, extra_special_tokens={}
        )

        log.info("Loading model (dtype=%s, device=%s)", dtype, device)
        self.model = (
            AutoModelForImageTextToText.from_pretrained(
                local_dir,
                trust_remote_code=True,
                torch_dtype=dtype,
            )
            .to(device)
            .eval()
        )
        self.device = device

        # Upstream `_move_inputs_to_device` only moves tensors; it does not cast
        # floats to the model dtype, so processor-produced fp32 pixel_values
        # trip `mat1 and mat2 must have the same dtype` under bf16 weights.
        target_dtype = next(self.model.parameters()).dtype

        def _move_and_cast(
            inputs: Any, dev: Any, _target: torch.dtype = target_dtype
        ) -> dict[str, Any]:
            out: dict[str, Any] = {}
            for key, value in inputs.items():
                if torch.is_tensor(value):
                    value = value.to(dev)
                    if value.is_floating_point() and value.dtype != _target:
                        value = value.to(_target)
                out[key] = value
            return out

        self.model._move_inputs_to_device = _move_and_cast

        # CUDA graphs in the action expert are not safe under concurrent calls.
        self._lock = threading.Lock()

    @torch.inference_mode()
    def _run_predict_action(
        self,
        images: list[Image.Image],
        instruction: str,
        state: np.ndarray,
        num_steps: int,
        enable_cuda_graph: bool,
    ) -> np.ndarray:
        state_f32 = np.asarray(state, dtype=np.float32).reshape(-1)
        if state_f32.shape != (self.STATE_DIM,):
            raise ValueError(
                f"state must be shape ({self.STATE_DIM},), got {state_f32.shape}"
            )

        kwargs = {self.MODE_KWARG: "continuous"}
        with self._lock:
            out = self.model.predict_action(
                processor=self.processor,
                images=images,
                task=instruction,
                state=state_f32,
                norm_tag=self.NORM_TAG,
                enable_depth_reasoning=False,
                num_steps=num_steps,
                normalize_language=True,
                enable_cuda_graph=enable_cuda_graph,
                **kwargs,
            )
        raw = out.actions
        if torch.is_tensor(raw):
            raw = raw.detach().to(dtype=torch.float32, device="cpu").numpy()
        actions = np.asarray(raw, dtype=np.float32)
        if actions.ndim == 3 and actions.shape[0] == 1:
            actions = actions[0]
        return actions


# ---------------------------------------------------------------------------
# FastAPI app factory
# ---------------------------------------------------------------------------

# A handler that maps a decoded payload dict to (actions, num_steps_used).
# We keep the embodiment-specific payload extraction in each server file and
# pass it in here as a callable.
PredictHandler = Callable[[dict[str, Any]], np.ndarray]


def _refresh_gpu_gauges(embodiment: str, device: str) -> None:
    if not torch.cuda.is_available():
        return
    try:
        dev_idx = torch.device(device).index if "cuda" in device else 0
        GPU_ALLOCATED.labels(embodiment=embodiment, device=device).set(
            float(torch.cuda.memory_allocated(dev_idx))
        )
        GPU_RESERVED.labels(embodiment=embodiment, device=device).set(
            float(torch.cuda.memory_reserved(dev_idx))
        )
    except Exception:  # noqa: BLE001
        log.debug("failed to sample CUDA memory gauges", exc_info=True)


def build_app(
    policy: BasePolicy,
    title: str,
    health_extra: dict[str, Any],
    predict_from_payload: PredictHandler,
    default_num_steps: int,
) -> FastAPI:
    """Wire up /act (GET+POST), /healthz, /metrics.

    `predict_from_payload(payload) -> actions` is the embodiment's adapter:
    it pulls out cameras/state/instruction in the right shape, then calls
    `policy._run_predict_action(...)`.
    """
    app = FastAPI(title=title, version="0.2.0")
    embodiment = policy.EMBODIMENT

    @app.get("/act")
    async def health() -> JSONResponse:
        return JSONResponse(
            {
                "status": "ok",
                "repo_id": policy.REPO_ID,
                "revision": policy.revision,
                "norm_tag": policy.NORM_TAG,
                "device": policy.device,
                "dtype": str(policy.model.dtype),
                **health_extra,
            }
        )

    @app.get("/healthz")
    async def healthz() -> JSONResponse:
        return JSONResponse({"status": "ok"})

    @app.get("/metrics")
    async def metrics() -> PlainTextResponse:
        _refresh_gpu_gauges(embodiment, policy.device)
        return PlainTextResponse(
            content=generate_latest().decode("utf-8"),
            media_type=CONTENT_TYPE_LATEST,
        )

    @app.post("/act")
    async def act(request: Request) -> Response:
        raw = await request.body()
        try:
            payload = json_numpy.loads(raw.decode("utf-8"))
        except Exception as e:  # noqa: BLE001
            ERRORS.labels(embodiment=embodiment, kind="decode").inc()
            REQUESTS.labels(embodiment=embodiment, status="bad_request").inc()
            return error_response(400, f"failed to decode json_numpy body: {e}")

        t0 = time.perf_counter()
        try:
            # predict_from_payload blocks on inference. Push it to the
            # threadpool so the event loop (and /healthz, /metrics) stays
            # responsive while the GPU is busy.
            actions = await asyncio.to_thread(predict_from_payload, payload)
        except KeyError as e:
            ERRORS.labels(embodiment=embodiment, kind="missing_field").inc()
            REQUESTS.labels(embodiment=embodiment, status="bad_request").inc()
            return error_response(400, f"missing required field: {e}")
        except ValueError as e:
            ERRORS.labels(embodiment=embodiment, kind="bad_shape").inc()
            REQUESTS.labels(embodiment=embodiment, status="bad_request").inc()
            return error_response(400, str(e))
        except Exception as e:  # noqa: BLE001
            log.exception("inference failed")
            ERRORS.labels(embodiment=embodiment, kind="inference").inc()
            REQUESTS.labels(embodiment=embodiment, status="error").inc()
            return error_response(500, f"inference failed: {e}")

        dt_s = time.perf_counter() - t0
        LATENCY.labels(embodiment=embodiment).observe(dt_s)
        REQUESTS.labels(embodiment=embodiment, status="ok").inc()

        body = json_numpy.dumps({"actions": actions, "dt_ms": dt_s * 1000.0})
        return Response(content=body, media_type="application/json")

    return app


def run_warmup(
    policy: BasePolicy,
    dummy_images: list[np.ndarray],
    dummy_instruction: str,
    default_num_steps: int,
    build_predict: Callable[[BasePolicy, list[np.ndarray], str, int, bool], None],
) -> None:
    """Run one inference on dummy frames so the first real request is fast.

    `build_predict(policy, dummy_images, instruction, num_steps, cuda_graph)`
    is supplied by each server because it knows how to map images to its
    embodiment's `predict()` signature.
    """
    log.info(
        "Warming up model with dummy frame(s) (cuda_graph=%s) ...",
        policy.default_cuda_graph,
    )
    t0 = time.perf_counter()
    try:
        build_predict(
            policy,
            dummy_images,
            dummy_instruction,
            default_num_steps,
            policy.default_cuda_graph,
        )
    except Exception:  # noqa: BLE001
        log.exception("warmup inference failed (server will still start)")
        return
    log.info("Warmup OK (%.1f ms)", (time.perf_counter() - t0) * 1000.0)

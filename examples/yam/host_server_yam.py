"""MolmoAct2-BimanualYAM inference server.

Mirrors the DROID server but for the bimanual YAM checkpoint:

  * 3 cameras in fixed order [top, left, right]
  * raw robot state is shape (14,)  (per-arm 7-D, two arms)
  * norm_tag = "yam_dual_molmoact2"

Wire protocol:

    GET  /act        -> health check, returns {"status": "ok", "revision": ..., ...}
    GET  /healthz    -> liveness
    GET  /metrics    -> Prometheus exposition
    POST /act        -> action inference
        request body  (json_numpy):
            {
              "top_cam":     ndarray(H, W, 3) uint8 RGB,
              "left_cam":    ndarray(H, W, 3) uint8 RGB,
              "right_cam":   ndarray(H, W, 3) uint8 RGB,
              "instruction": str,
              "state":       ndarray(14,) float32,
              "timestamp":   float (optional),
              "num_steps":   int   (optional, default 10),
            }
        response body (json_numpy):
            {"actions": ndarray(N, D) float32, "dt_ms": float}

Run:

    uv run python examples/yam/host_server_yam.py --host 0.0.0.0 --port 8202
"""

from __future__ import annotations

import os

# Enable hf-transfer BEFORE huggingface_hub is imported: its constants module
# captures this env var at import time and never re-reads it, so setting it
# later (e.g. inside main()) is a silent no-op and downloads fall back to the
# slow single-stream path.
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

import argparse
import logging
import sys
from typing import Any

import json_numpy
import numpy as np
import torch

json_numpy.patch()

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _common.server import (  # noqa: E402
    BasePolicy,
    build_app,
    run_warmup,
    to_pil,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("molmoact2.yam.server")


REPO_ID = "allenai/MolmoAct2-BimanualYAM"
NORM_TAG = "yam_dual_molmoact2"
STATE_DIM = 14
NUM_CAMERAS = 3
DEFAULT_NUM_STEPS = 10


class YamPolicy(BasePolicy):
    EMBODIMENT = "yam"
    REPO_ID = REPO_ID
    NORM_TAG = NORM_TAG
    STATE_DIM = STATE_DIM
    # YAM checkpoint expects `inference_action_mode` (no default upstream).
    MODE_KWARG = "inference_action_mode"

    def predict(
        self,
        top_cam: np.ndarray,
        left_cam: np.ndarray,
        right_cam: np.ndarray,
        instruction: str,
        state: np.ndarray,
        num_steps: int = DEFAULT_NUM_STEPS,
        enable_cuda_graph: bool = False,
    ) -> np.ndarray:
        # Camera order must match training: [top, left, right].
        images = [to_pil(top_cam), to_pil(left_cam), to_pil(right_cam)]
        return self._run_predict_action(
            images=images,
            instruction=instruction,
            state=state,
            num_steps=num_steps,
            enable_cuda_graph=enable_cuda_graph,
        )


def _predict_from_payload(policy: YamPolicy, payload: dict[str, Any]) -> np.ndarray:
    top_cam = payload["top_cam"]
    left_cam = payload["left_cam"]
    right_cam = payload["right_cam"]
    instruction = str(payload["instruction"])
    state = payload["state"]
    num_steps = int(payload.get("num_steps", DEFAULT_NUM_STEPS))
    return policy.predict(
        top_cam=top_cam,
        left_cam=left_cam,
        right_cam=right_cam,
        instruction=instruction,
        state=state,
        num_steps=num_steps,
        enable_cuda_graph=policy.default_cuda_graph,
    )


def _warmup_call(
    policy: BasePolicy,
    dummy_images: list[np.ndarray],
    instruction: str,
    num_steps: int,
    cuda_graph: bool,
) -> None:
    assert isinstance(policy, YamPolicy)
    policy.predict(
        top_cam=dummy_images[0],
        left_cam=dummy_images[1],
        right_cam=dummy_images[2],
        instruction=instruction,
        state=np.zeros(policy.STATE_DIM, dtype=np.float32),
        num_steps=num_steps,
        enable_cuda_graph=cuda_graph,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MolmoAct2-BimanualYAM inference server")
    p.add_argument("--host", default="0.0.0.0", help="bind address (default: 0.0.0.0)")
    p.add_argument("--port", type=int, default=8202, help="bind port (default: 8202)")
    p.add_argument("--repo-id", default=REPO_ID, help=f"HF repo id (default: {REPO_ID})")
    p.add_argument("--device", default="cuda:0", help="torch device (default: cuda:0)")
    p.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="model dtype (default: bfloat16; fp32 needs ~26 GB)",
    )
    p.add_argument("--no-warmup", action="store_true", help="skip warmup pass")
    p.add_argument(
        "--cuda-graph",
        action="store_true",
        help="enable CUDA graph capture for action expert (faster but ~2 GB more VRAM)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]

    policy = YamPolicy(
        repo_id=args.repo_id,
        device=args.device,
        dtype=dtype,
        enable_cuda_graph=args.cuda_graph,
    )
    if not args.no_warmup:
        dummy = np.zeros((180, 320, 3), dtype=np.uint8)
        run_warmup(
            policy=policy,
            dummy_images=[dummy, dummy, dummy],
            dummy_instruction="warmup",
            default_num_steps=DEFAULT_NUM_STEPS,
            build_predict=_warmup_call,
        )

    app = build_app(
        policy=policy,
        title="MolmoAct2-BimanualYAM server",
        health_extra={"num_cameras": NUM_CAMERAS, "state_dim": policy.STATE_DIM},
        predict_from_payload=lambda payload: _predict_from_payload(policy, payload),
        default_num_steps=DEFAULT_NUM_STEPS,
    )

    import uvicorn

    log.info("Listening on %s:%d", args.host, args.port)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()

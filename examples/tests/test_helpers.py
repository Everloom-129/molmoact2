"""Pure-function tests for helpers in `_common/server.py`.

No FastAPI, no model load. Just `to_pil`, `error_response`, and
`patch_modeling_for_bf16`.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest
from PIL import Image

from _common.server import error_response, patch_modeling_for_bf16, to_pil


# ---------------------------------------------------------------------------
# to_pil
# ---------------------------------------------------------------------------

def test_to_pil_uint8_passthrough():
    arr = np.zeros((4, 5, 3), dtype=np.uint8)
    arr[0, 0] = (10, 20, 30)
    img = to_pil(arr)
    assert isinstance(img, Image.Image)
    assert img.mode == "RGB"
    assert img.size == (5, 4)
    assert img.getpixel((0, 0)) == (10, 20, 30)


def test_to_pil_clips_and_casts_float():
    # Float inputs out of [0, 255] should clip and cast cleanly rather than
    # silently producing a garbage uint8 wrap-around.
    arr = np.array(
        [[[-50.0, 128.0, 999.0]]],
        dtype=np.float32,
    )
    img = to_pil(arr)
    assert np.array(img)[0, 0].tolist() == [0, 128, 255]


def test_to_pil_rejects_non_hxwx3():
    with pytest.raises(ValueError):
        to_pil(np.zeros((4, 5), dtype=np.uint8))
    with pytest.raises(ValueError):
        to_pil(np.zeros((4, 5, 4), dtype=np.uint8))


def test_to_pil_accepts_pil_image_in():
    src = Image.new("RGBA", (2, 2), (255, 0, 0, 128))
    out = to_pil(src)
    assert out.mode == "RGB"


# ---------------------------------------------------------------------------
# error_response
# ---------------------------------------------------------------------------

def test_error_response_carries_status_and_message():
    resp = error_response(418, "i'm a teapot")
    assert resp.status_code == 418
    assert resp.media_type == "application/json"
    payload = json.loads(resp.body.decode("utf-8"))
    assert payload == {"error": "i'm a teapot"}


# ---------------------------------------------------------------------------
# patch_modeling_for_bf16
# ---------------------------------------------------------------------------

# A minimal stand-in for the upstream `modeling_molmoact2.py` that contains
# both needles. Whitespace and surrounding context must match exactly because
# the patcher does string `.replace`.
_NEEDLE_DTYPE = (
    "device=device,\n            dtype=torch.float32,\n            generator=generator,"
)
_NEEDLE_TO_ARRAY = (
    "return value.detach().cpu().numpy().astype(np.float32, copy=False)"
)

_SAMPLE_MODELING = f"""# fake modeling file for tests
def make_traj():
    {_NEEDLE_DTYPE}
    return 1

def _to_array(value):
    {_NEEDLE_TO_ARRAY}
"""


def test_patch_modeling_applies_both_needles(tmp_path, monkeypatch):
    # Point the patcher's cache-scan at a tmp HF modules root so the test
    # doesn't depend on (or write to) the user's real ~/.cache.
    monkeypatch.setenv("HOME", str(tmp_path))

    local_dir = tmp_path / "snap"
    local_dir.mkdir()
    target = local_dir / "modeling_molmoact2.py"
    target.write_text(_SAMPLE_MODELING)

    patch_modeling_for_bf16(str(local_dir))
    patched = target.read_text()
    assert "patched_bf16_dtype" in patched
    assert "patched_bf16_to_array" in patched
    # Source needles should be gone (replaced).
    assert "dtype=torch.float32,\n            generator=generator," not in patched


def test_patch_modeling_is_idempotent(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    local_dir = tmp_path / "snap"
    local_dir.mkdir()
    target = local_dir / "modeling_molmoact2.py"
    target.write_text(_SAMPLE_MODELING)

    patch_modeling_for_bf16(str(local_dir))
    first = target.read_text()
    patch_modeling_for_bf16(str(local_dir))
    second = target.read_text()
    assert first == second  # second pass is a no-op


def test_patch_modeling_handles_missing_needles_gracefully(tmp_path, monkeypatch):
    # YAM checkpoints don't have either needle; the patcher should just log
    # and leave the file alone, not raise.
    monkeypatch.setenv("HOME", str(tmp_path))
    local_dir = tmp_path / "snap"
    local_dir.mkdir()
    target = local_dir / "modeling_molmoact2.py"
    target.write_text("# upstream already fixed\n")

    patch_modeling_for_bf16(str(local_dir))
    assert target.read_text() == "# upstream already fixed\n"

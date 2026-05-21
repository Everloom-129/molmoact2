# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repo scope

This is the MolmoAct2 release repo. The tracked code under this working directory is:

- `examples/_common/server.py` — shared scaffolding (bf16 patches, FastAPI app factory with async-dispatched `/act`, `/healthz`, `/metrics`, `BasePolicy`, warmup helper, Prometheus metrics). Each embodiment server subclasses `BasePolicy` and supplies a payload extractor.
- `examples/droid/host_server_droid.py` — FastAPI inference server for `allenai/MolmoAct2-DROID` (2 cams, 8-D state, `norm_tag="franka_droid"`, default port 8000).
- `examples/yam/host_server_yam.py` — same shape, but for `allenai/MolmoAct2-BimanualYAM` (3 cams `[top, left, right]`, 14-D state, `norm_tag="yam_dual_molmoact2"`, default port 8202).

Untracked but kept locally:

- `logs/inference_script.py` — DROID Polymetis client bridge that talks to `host_server_droid.py` from the NUC driving a Franka. Not tracked in this branch; treat it as the reference wire-format consumer for DROID.
- `README copy.md` — local snapshot of the pre-cloud-merge README, kept for diff purposes.

The rest of MolmoAct2 (training, fine-tuning, eval) is "coming soon" per the README. The `lerobot/` directory is a submodule (`allenai/lerobot-molmoact2`); evaluation work (e.g. LIBERO replication) happens in there and uses its own README/tooling.

## Environment

- Python is pinned to 3.11 via `.python-version`; `uv` manages everything.
- Torch wheels come from the CUDA-12.1 PyTorch index (configured in `pyproject.toml` under `[tool.uv.sources]`). Don't relax these pins casually — the model loading code path was validated against torch 2.5.1 / transformers 4.57.x.
- After pulling new commits: `uv sync`. After cloning: also `git submodule update --init --recursive`.

## Common commands

```bash
uv sync                                                                    # install/refresh .venv
uv run python examples/droid/host_server_droid.py --host 0.0.0.0           # DROID server, default port 8000 (bf16)
uv run python examples/yam/host_server_yam.py --host 0.0.0.0               # YAM   server, default port 8202 (bf16)
uv run python examples/droid/host_server_droid.py --dtype float32 --cuda-graph  # full precision + CUDA graphs
uv run hf download allenai/MolmoAct2-DROID                                 # pre-cache DROID (~22 GB)
uv run hf download allenai/MolmoAct2-BimanualYAM                           # pre-cache YAM   (~21 GB)
curl http://<host>:8000/act                                                # DROID health
curl http://<host>:8202/act                                                # YAM   health
curl http://<host>:8000/metrics                                            # Prometheus exposition
```

Useful server flags: `--dtype {bfloat16,float16,float32}` (default bf16; fp32 needs ~96 GB VRAM), `--device cuda:0`, `--cuda-graph` (~2× faster action expert, +~2 GB VRAM, not safe under concurrent calls), `--no-warmup`.

There is no test suite or linter wired up in this repo.

## Wire protocol (`/act`)

Both directions are `json_numpy`-encoded (each server calls `json_numpy.patch()` at import, which monkey-patches the stdlib `json` module so ndarrays round-trip). Each server has its own schema — the endpoint path is the same but the payload shape differs by embodiment.

DROID (`examples/droid/host_server_droid.py`):

- Request: `external_cam` (H,W,3 uint8 RGB), `wrist_cam` (H,W,3 uint8 RGB), `instruction` (str), `state` (8,) float32 = `[q1..q7, gripper]`. Optional: `num_steps` (default 10), `enable_cuda_graph`, `timestamp`.
- Response: `actions` (N, 8) float32 absolute joint positions + gripper, `dt_ms` float.

YAM (`examples/yam/host_server_yam.py`):

- Request: `top_cam`, `left_cam`, `right_cam` (each H,W,3 uint8 RGB — order matters and must match training), `instruction` (str), `state` (14,) float32, plus the same optional fields.
- Response: `actions` (N, D) float32, `dt_ms` float. Action shape is driven by `norm_stats.json`; don't hardcode it.

`logs/inference_script.py` is the DROID reference client (untracked). There's no YAM bridge in this repo; if you write one, mirror the YAM schema above. The bridge is hand-rolled HTTP (`requests.post`), not auto-generated, so server-schema changes need matching client edits.

## Server architecture — non-obvious bits

The MolmoAct2 checkpoints were not released with `bfloat16` or local-snapshot loading in mind. The shared scaffolding in `examples/_common/server.py` applies the same set of upstream workarounds; future changes need to preserve these or both servers will silently break:

1. **Snapshot-dir loading.** The model's `predict_action` reads `norm_stats.json` from `config._name_or_path`. Loading by repo id leaves that as a non-path string and crashes at inference time. `BasePolicy.__init__` always resolves `snapshot_download(repo_id)` and loads from the local directory, and stashes the snapshot SHA on `policy.revision` so it can be surfaced on the `/act` health response.

2. **bf16 patches to upstream `modeling_molmoact2.py`.** `patch_modeling_for_bf16` rewrites the cached `modeling_molmoact2.py` (both in the snapshot dir and in `~/.cache/huggingface/modules/transformers_modules/*/`, which is the copy `trust_remote_code` actually imports) at startup. Two edits, both idempotent and marked with `# patched_bf16_*` comments:
   - Flow-matching trajectory dtype: hardcoded `torch.float32` → `source_tensor.dtype`.
   - `_to_array`: cast to fp32 before `.numpy()` because numpy has no bf16 dtype.

   These re-apply on every server start, so re-downloading the checkpoint won't permanently break things — but if you bump `transformers` or the upstream repo restructures `modeling_molmoact2.py`, the textual needles can stop matching and you'll see "needle not found" warnings.

3. **`tokenizer_config.json` ships `extra_special_tokens` as a list.** transformers ≥4.46 expects a dict and crashes with `'list' object has no attribute 'keys'`. `AutoProcessor.from_pretrained(..., extra_special_tokens={})` overrides it; the model code only looks these up via `convert_tokens_to_ids` so the empty dict is safe.

4. **Per-instance `_move_inputs_to_device` override.** Upstream moves tensors to the device but doesn't cast floats to the model dtype. With bf16 weights the processor's fp32 `pixel_values` then trips `mat1 and mat2 must have the same dtype`. `BasePolicy.__init__` replaces the bound method with a version that casts floating-point tensors to the model dtype after the device move.

5. **Coarse lock around `predict_action`.** Robot clients poll at ~5 Hz and the action-expert CUDA graphs are not safe under concurrent calls — `BasePolicy._lock` serializes inference even when multiple HTTP clients are connected.

6. **POST `/act` dispatches inference to a worker thread.** The FastAPI route is `async def` but calls `policy.predict(...)` through `asyncio.to_thread`, so a ~80 ms inference does not block the event loop — `/healthz` and `/metrics` stay responsive while the GPU is busy. The internal `_lock` still serializes the GPU.

7. **CUDA-graph capture is a startup-only decision.** The `--cuda-graph` flag is captured into `policy.default_cuda_graph` and the per-request `enable_cuda_graph` field is ignored on purpose. Mid-episode capture toggles cost a VRAM allocation and a latency spike.

8. **Prometheus metrics live in `_common/server.py`.** Counters and a latency histogram, all labeled by embodiment, plus GPU memory gauges that are sampled lazily on every `/metrics` scrape. Each embodiment process is its own metric source — point Prometheus at every host:port pair.

## Conventions for changes

- Each `examples/<embodiment>/host_server_*.py` is the source of truth for its embodiment's `/act` schema, `NORM_TAG`, and state/camera count. New deployments (SO-100/101, LIBERO) should clone the same template into a new sibling directory, subclass `BasePolicy`, and override those constants — don't try to multiplex embodiments behind one server, and don't put embodiment-specific schema knowledge into `_common/server.py`.
- `predict_action`'s mode kwarg differs across checkpoints: the DROID revision uses `action_mode="continuous"` (defaulted), the YAM revision uses `inference_action_mode="continuous"` (required, no default). Set `MODE_KWARG` on the subclass — copying it verbatim between embodiments will TypeError at warmup.
- Don't edit the cached `modeling_molmoact2.py` directly — extend `patch_modeling_for_bf16` in `_common/server.py` so the change survives a cache rebuild. The two textual needles only match the DROID snapshot revision: on DROID, `patched_bf16_dtype` no longer matches but `patched_bf16_to_array` still applies; on YAM, both are already fixed upstream and warn "needle not found" — bf16 inference works regardless. Keep the patch code in place so re-downloads of older snapshots still work.
- `logs/inference_script.py` is the DROID Polymetis client bridge (untracked in this branch). It depends on `pyzed.sl`, `cv2`, `requests`, `json_numpy`, and optionally `pandas` for Excel logging — don't introduce server-only deps into it, and don't assume future contributors will have the file.

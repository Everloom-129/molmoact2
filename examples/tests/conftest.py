"""Test-suite setup.

These tests intentionally avoid touching the network or the GPU — no
`snapshot_download`, no model load, no `transformers` weights. We exercise
only the bits we own: the FastAPI app factory, payload extraction, helper
functions, and metric labels. The shared module still pulls in `torch` and
`transformers` at import time; that's fine in the dev venv but the tests
themselves should never call `BasePolicy.__init__`.
"""

from __future__ import annotations

import os
import sys

# Make `_common` importable as a top-level package so test files can
# `from _common.server import ...` without each one repeating the path dance
# that the embodiment server scripts do.
EXAMPLES_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if EXAMPLES_DIR not in sys.path:
    sys.path.insert(0, EXAMPLES_DIR)

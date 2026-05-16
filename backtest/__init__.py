"""Bootstrap: add the C++ pybind11 build dir to sys.path so `import clob_py`
works when running backtest scripts directly (not just via pytest)."""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
for _build_dir in (
    _REPO_ROOT / "build" / "Release" / "bindings",
    _REPO_ROOT / "build" / "Debug" / "bindings",
):
    if _build_dir.exists() and any(_build_dir.glob("clob_py*.so")):
        sp = str(_build_dir)
        if sp not in sys.path:
            sys.path.insert(0, sp)
        break

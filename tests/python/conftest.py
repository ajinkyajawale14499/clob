"""Pytest conftest — wires the build/Release/bindings .so into sys.path.

The pybind11 module `clob_py` is built by CMake (see bindings/CMakeLists.txt).
Until W15 packages it via scikit-build-core for `pip install -e .`, tests
add the build dir to sys.path manually.

The Release build is used (not Debug) because ASan-instrumented .so files
fail to load via dlopen from a non-ASan Python interpreter.
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).parents[2]
_BINDINGS_DIRS = [
    _REPO_ROOT / "build" / "Release" / "bindings",
    _REPO_ROOT / "build" / "Debug" / "bindings",  # fallback if Release not built
]

for d in _BINDINGS_DIRS:
    if d.exists() and any(d.glob("clob_py*.so")):
        sys.path.insert(0, str(d))
        break

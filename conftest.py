"""Pytest bootstrap: ensure the ``src`` layout package is importable.

The project uses a ``src/`` layout (see ``pyproject.toml``). When the package
is not installed (e.g. ``pip install -e .`` was skipped), this adds ``src`` to
``sys.path`` so ``import loyalty_engine`` works during the test run.
"""
from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src"
if SRC.is_dir() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

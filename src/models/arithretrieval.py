"""Backward-compatible alias for the WideQuant model."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.widequant import WideQuant

__all__ = ["WideQuant"]


if __name__ == "__main__":
    print("arithretrieval.py compatibility alias loaded: WideQuant")

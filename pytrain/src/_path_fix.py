"""Auto-imported by pytrain modules to ensure root config.py is findable."""
import sys
from pathlib import Path

_REPO_ROOT   = str(Path(__file__).resolve().parents[2])  # repo root
_PYTRAIN_DIR = str(Path(__file__).resolve().parents[1])  # pytrain/

for _p in (_REPO_ROOT, _PYTRAIN_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

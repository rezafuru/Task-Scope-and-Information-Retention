"""Repository locations shared by the scripts and the notebook.

The package is meant to be used from a checkout (``pip install -e .``). Set
``TASKSCOPE_ROOT`` to override the inferred checkout location.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(os.environ.get("TASKSCOPE_ROOT", Path(__file__).resolve().parents[1]))
RESULTS = REPO_ROOT / "results"
ASSETS = REPO_ROOT / "assets"
FIGURES = REPO_ROOT / "figures"

#!/usr/bin/env python3
"""Pool the CIFAR-100 codec runs into candidate groups and apply the frozen selection.

On validation the rule picks, per teacher and per preservation loss, the group
with the lowest mean complete rate meeting both accuracy allowances, and writes
that choice out. On test the stored validation file is required, its hash is
recorded, and the selections it fixed are carried over unchanged.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from taskscope.cifar.analysis import analyze
from taskscope.paths import REPO_ROOT

DEFAULT_RUN = REPO_ROOT / "runs/cifar"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_RUN,
                        help="directory holding the teacher run directories")
    parser.add_argument("--split", choices=["validation", "test"], required=True)
    parser.add_argument("--selection", type=Path,
                        help="frozen validation selection, required for the test split")
    arguments = parser.parse_args()
    if arguments.split == "test" and arguments.selection is None:
        parser.error("--selection is required for the test split")
    print(analyze(arguments.root, arguments.split, arguments.selection))


if __name__ == "__main__":
    main()

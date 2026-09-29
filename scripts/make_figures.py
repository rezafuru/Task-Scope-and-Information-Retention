#!/usr/bin/env python3
"""Build every figure the paper uses, from the retained evidence.

Needs no dataset, no trained weights, and no GPU.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from taskscope.paths import FIGURES, REPO_ROOT

GENERATORS = (
    ("scripts/figures/pilot.py", ["paper_pilot_bunting_grosbeak.pdf"]),
    ("scripts/figures/task_scope_limits.py", ["paper_task_scope_limits.pdf"]),
    ("scripts/figures/block_coding.py", ["paper_block_coding_sensitivity.pdf"]),
    ("scripts/figures/nonbinary_risks.py", ["figure3_risks.pdf"]),
    ("scripts/figures/imagenet.py", ["paper_imagenet_task_scope.pdf"]),
    ("scripts/figures/cifar.py",
     ["paper_cifar_local_preservation.pdf", "paper_cifar_validation.pdf"]),
    ("scripts/figures/taskonomy_requirements.py", ["paper_taskonomy_requirements.pdf"]),
    ("scripts/figures/taskonomy_examples.py", ["paper_taskonomy_joint_examples.pdf"]),
)

# Panels the joint Taskonomy figure is composed from, written on the way to it.
INTERMEDIATE = ("paper_taskonomy_semantic_examples.pdf", "paper_taskonomy_edge_examples.pdf")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=FIGURES)
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=True)
    expected = []
    for script, produces in GENERATORS:
        print(f"=== {script}", flush=True)
        subprocess.run([sys.executable, str(REPO_ROOT / script), "--output", str(arguments.output)],
                       cwd=REPO_ROOT, check=True)
        expected.extend(produces)
    missing = [name for name in expected + list(INTERMEDIATE)
               if not (arguments.output / name).exists()]
    if missing:
        raise SystemExit(f"Figure generators did not produce: {missing}")
    extra = sorted(path.name for path in arguments.output.glob("*.pdf")
                   if path.name not in expected and path.name not in INTERMEDIATE)
    print(f"\n{len(expected)} paper figures in {arguments.output}, "
          f"plus {len(INTERMEDIATE)} composed panels")
    if extra:
        print(f"other PDFs present: {extra}")


if __name__ == "__main__":
    main()

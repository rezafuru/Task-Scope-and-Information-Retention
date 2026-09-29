#!/usr/bin/env python3
"""Assess fitted conditional-cost selectors on the three frozen ternary codebooks.

Reads the run directory the codes stage wrote, which holds the confirmation and
the two replications with their codebooks, assessment blocks, known-cost packets
and per-block risks. Fits one cost table per codebook from an independent
calibration draw, encodes under both observations, and adds the two controls.
Needs no dataset and runs on a CPU with ``--device cpu``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from taskscope import nonbinary_coding
from taskscope.paths import REPO_ROOT

DEFAULT_RUN = REPO_ROOT / "runs/nonbinary"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_RUN,
                        help="run directory holding the three replicate subdirectories")
    parser.add_argument("--output", type=Path, default=DEFAULT_RUN / "learned_selectors")
    parser.add_argument("--calibration-count", type=int, default=1048576)
    parser.add_argument("--control-blocks", type=int, default=32768)
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda")
    parser.add_argument("--threads", type=int, default=4)
    arguments = parser.parse_args()
    if arguments.calibration_count < 1 or arguments.control_blocks < 2:
        parser.error("Positive calibration size and at least two control blocks are required")
    nonbinary_coding.learned_selectors(
        arguments.source, arguments.output, device_name=arguments.device,
        calibration_count=arguments.calibration_count, control_blocks=arguments.control_blocks,
        threads=arguments.threads, source_file=Path(nonbinary_coding.__file__))


if __name__ == "__main__":
    main()

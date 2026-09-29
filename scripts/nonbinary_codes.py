#!/usr/bin/env python3
"""Fit ternary codebooks and confirm one of them on an independent draw.

Stages, in order:

  fit      fit codebooks at the given index widths, select a development
           mixture, and assess it on fresh blocks
  confirm  assess a frozen codebook and the declared risk requirement on an
           independent source draw, against the reduced-observation minimum

The source is synthetic, so no dataset is needed. ``fit`` is GPU work at the
default sizes but runs on a CPU with ``--device cpu``. ``confirm`` reproduces
the reported packet and risks from a stored codebook alone.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from taskscope import nonbinary_coding
from taskscope.nonbinary import CONFIRMATION_TOLERANCE
from taskscope.paths import REPO_ROOT

DEFAULT_RUN = REPO_ROOT / "runs/nonbinary"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["fit", "confirm"])
    parser.add_argument("--output", type=Path, default=DEFAULT_RUN / "profile_n16_b13")
    parser.add_argument("--codebook", type=Path,
                        default=DEFAULT_RUN / "profile_n16_b13/full_b13_codebook.npz",
                        help="frozen codebook the confirm stage reads")
    parser.add_argument("--length", type=int, default=16)
    parser.add_argument("--bits", type=int, nargs="+", default=[13])
    parser.add_argument("--access", nargs="+", choices=["full", "reduced"],
                        default=["full", "reduced"])
    parser.add_argument("--condition", choices=["expanded", "primary", "no_confidence"],
                        default="expanded")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--train-count", type=int, default=131072)
    parser.add_argument("--dev-count", type=int, default=16384)
    parser.add_argument("--assessment-count", type=int, default=32768)
    parser.add_argument("--primary-weight", type=float, default=0.0)
    parser.add_argument("--additional-tolerance", type=float, default=CONFIRMATION_TOLERANCE)
    parser.add_argument("--safety-margin", type=float, default=0.001)
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda")
    parser.add_argument("--threads", type=int, default=4)
    arguments = parser.parse_args()
    if arguments.length < 1 or min(arguments.train_count, arguments.dev_count,
                                   arguments.assessment_count) < 2 or arguments.safety_margin < 0:
        parser.error("Positive dimensions and at least two blocks per split are required")

    source_file = Path(nonbinary_coding.__file__)
    if arguments.stage == "fit":
        nonbinary_coding.fit(
            arguments.output, device_name=arguments.device, length=arguments.length,
            bits=arguments.bits, access=arguments.access, condition=arguments.condition,
            seed=arguments.seed, rounds=arguments.rounds, train_count=arguments.train_count,
            dev_count=arguments.dev_count, assessment_count=arguments.assessment_count,
            primary_weight=arguments.primary_weight,
            additional_tolerance=arguments.additional_tolerance,
            safety_margin=arguments.safety_margin, threads=arguments.threads,
            source_file=source_file)
    else:
        if len(arguments.access) != 1:
            parser.error("Confirmation needs exactly one observation")
        nonbinary_coding.confirm(
            arguments.output, codebook_path=arguments.codebook, device_name=arguments.device,
            access=arguments.access[0], condition=arguments.condition, seed=arguments.seed,
            assessment_count=arguments.assessment_count, primary_weight=arguments.primary_weight,
            additional_tolerance=arguments.additional_tolerance, threads=arguments.threads,
            source_file=source_file)


if __name__ == "__main__":
    main()

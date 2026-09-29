#!/usr/bin/env python3
"""Run one stage of the CIFAR-100 preservation study.

Stages run in the order listed by ``--help``. The teacher and extraction stages
need the CIFAR-100 archive, supplied with ``--data``, and torchvision downloads
it into that directory on first use. Fitting stages need a GPU to finish in
reasonable time. Rates reported by the decoding stages are complete message
lengths, including the header, the hyperlatent, and the main payload.
"""

from __future__ import annotations

import argparse

from taskscope.cifar.pipeline import DEFAULT_EPOCHS, DEFAULT_LR, STAGES, Settings, run
from taskscope.paths import REPO_ROOT

DEFAULT_RUN = REPO_ROOT / "runs/cifar"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument("--data", help="directory holding cifar-100-python, required by the "
                                       "teacher and extraction stages")
    parser.add_argument("--output", default=str(DEFAULT_RUN))
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or mps")
    parser.add_argument("--epochs", type=int,
                        help=f"stage defaults {DEFAULT_EPOCHS}")
    parser.add_argument("--lr", type=float, help=f"stage defaults {DEFAULT_LR}")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--observation", choices=["early", "late"], default="early")
    parser.add_argument("--objective", choices=["early", "late", "family"], default="early")
    parser.add_argument("--beta", type=float, default=0.1, help="rate weight")
    parser.add_argument("--checkpoint", help="codec checkpoint the decoding stages read")
    parser.add_argument("--decoded-readouts", action="store_true")
    parser.add_argument("--initialization", help="codec checkpoint a paired continuation starts from")
    parser.add_argument("--run-tag", default="")
    parser.add_argument("--reference-output", help="run directory holding the uncoded reference")
    parser.add_argument("--readout-output", help="run directory holding the fitted readouts")
    parser.add_argument("--pool-decoded-cache", action="store_true")
    arguments = parser.parse_args()
    run(Settings(**{key: value for key, value in vars(arguments).items()}))


if __name__ == "__main__":
    main()

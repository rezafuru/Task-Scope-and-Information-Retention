#!/usr/bin/env python3
"""Fit, rescore, select, and forecast the learned block codes.

Stages, in order:

  run       fit codebooks and encoders across seeds, block lengths, index widths
  rescore   repeat validation and assessment with scores rounded to fixed precision
  analyze   select a mixture per requirement family and compare observations
  forecast  predict additional-task risks on calibration blocks, then assess

The source is synthetic, so no dataset is needed. ``run`` and ``rescore`` take
roughly ten minutes each on a GPU and considerably longer on a CPU.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from taskscope import block_coding
from taskscope.paths import REPO_ROOT

DEFAULT_RUN = REPO_ROOT / "runs/block_coding"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["run", "rescore", "analyze", "forecast", "predictions"])
    parser.add_argument("--output", type=Path, default=DEFAULT_RUN / "fitted")
    parser.add_argument("--source", type=Path, default=DEFAULT_RUN / "fitted",
                        help="run directory that rescore reads from")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or mps")
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 12, 13])
    parser.add_argument("--lengths", type=int, nargs="+", default=[12, 16])
    parser.add_argument("--laws", default="two_tau_0,two_tau_0.125,two_tau_0.5,three")
    parser.add_argument("--min-rate", type=float, default=0.30)
    parser.add_argument("--max-rate", type=float, default=0.80)
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--train-count", type=int, default=65536)
    parser.add_argument("--validation-count", type=int, default=65536)
    parser.add_argument("--assessment-count", type=int, default=131072)
    parser.add_argument("--score-steps", type=int, default=250)
    parser.add_argument("--sample-seed-offset", type=int, default=1000000,
                        help="run used 1000000, rescore used 2000000 for fresh draws")
    parser.add_argument("--score-decimals", type=int, default=5)
    parser.add_argument("--safety-margin", type=float, default=0.0015)
    parser.add_argument("--drop-paired-records", action="store_true",
                        help="skip the per-comparison record file and keep only the aggregate")
    arguments = parser.parse_args()

    if arguments.stage == "run":
        block_coding.run(
            arguments.output, device_name=arguments.device, seeds=arguments.seeds,
            lengths=arguments.lengths, laws=arguments.laws, min_rate=arguments.min_rate,
            max_rate=arguments.max_rate, rounds=arguments.rounds,
            train_count=arguments.train_count, validation_count=arguments.validation_count,
            assessment_count=arguments.assessment_count, score_steps=arguments.score_steps,
            sample_seed_offset=arguments.sample_seed_offset,
            source_file=Path(block_coding.__file__))
    elif arguments.stage == "rescore":
        block_coding.rescore(
            arguments.source, arguments.output, device_name=arguments.device,
            sample_seed_offset=arguments.sample_seed_offset,
            score_decimals=arguments.score_decimals, source_file=Path(block_coding.__file__))
    elif arguments.stage == "analyze":
        block_coding.analyze(arguments.output, safety_margin=arguments.safety_margin,
                             keep_paired_records=not arguments.drop_paired_records)
    elif arguments.stage == "forecast":
        block_coding.forecast(arguments.output, device_name=arguments.device)
    else:
        print(json.dumps(block_coding.conditions(), indent=2))


if __name__ == "__main__":
    main()

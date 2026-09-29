#!/usr/bin/env python3
"""Recompute the exact three-class rate evidence.

Writes the matched-risk comparison of the two observations at the tolerance the
finite codes are confirmed against, and reports the reduced-observation minimum
from both the closed ternary Fano form and the numerical dual. Needs no data and
no GPU, and runs in a few seconds.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from taskscope import nonbinary
from taskscope.paths import RESULTS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=RESULTS / "nonbinary")
    parser.add_argument("--tolerance", type=float, default=nonbinary.CONFIRMATION_TOLERANCE)
    parser.add_argument("--quiet", action="store_true")
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=True)

    record = nonbinary.matched_reference(arguments.tolerance)
    (arguments.output / "matched_confirmation_reference.json").write_text(
        json.dumps(record, indent=2, allow_nan=False) + "\n")
    agreement = nonbinary.bound_agreement(arguments.tolerance)
    if not arguments.quiet:
        print(json.dumps({
            "full_rate": record["full"]["rate"], "reduced_rate": record["reduced"]["rate"],
            "matched_rate_gap": record["matched_rate_gap"],
            "dual_gap_bounds": record["dual_gap_bounds"],
            "reduced_minimum": agreement,
        }, indent=2))


if __name__ == "__main__":
    main()

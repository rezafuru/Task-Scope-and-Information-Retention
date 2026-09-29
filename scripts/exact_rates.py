#!/usr/bin/env python3
"""Recompute the exact finite-family rate evidence.

Writes the rate grid, the summary quantities quoted in the main text, and the
conditional-error-profile numerical checks. Needs no data and no GPU.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from taskscope import exact, profiles
from taskscope.paths import RESULTS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=RESULTS / "exact")
    parser.add_argument("--quiet", action="store_true")
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=True)

    exact.write_rate_curves(arguments.output / "rate_curves.csv")
    summary = exact.build_summary()
    (arguments.output / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n")

    def progress(record):
        if not arguments.quiet:
            print(record["name"], record["formula_rate"], record["absolute_discrepancy"], flush=True)

    records = profiles.run_checks(progress)
    (arguments.output / "profile_numerical_checks.json").write_text(
        json.dumps(records, indent=2, allow_nan=False) + "\n")
    if not arguments.quiet:
        print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

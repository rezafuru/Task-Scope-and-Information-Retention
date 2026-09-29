#!/usr/bin/env python3
"""Rebuild the compact Taskonomy tables from the retained evaluation evidence.

Seven tables come from the released pool summaries alone: the candidate pool, the mixture
schedule in both formats, the RGB decoder contrast, the retained paired contrasts, the run
summary, and provenance. Eight more describe the focal fits per seed and per building and the
semantic class support, and those need the per-image record tree that the release omits. Point
``--records-dir`` at a copy of it to write them too.

Output goes to a scratch directory so that a verification run never overwrites the released
tables under ``results/taskonomy``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from taskscope.paths import REPO_ROOT
from taskscope.taskonomy import analysis


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "runs/taskonomy_analysis")
    parser.add_argument("--pool", type=Path, default=analysis.POOL,
                        help="directory holding the retained pool summaries and tolerances")
    parser.add_argument("--reference-validation", type=Path, default=analysis.REFERENCE_VALIDATION)
    parser.add_argument("--reference-test", type=Path, default=analysis.REFERENCE_TEST)
    parser.add_argument("--records-dir", type=Path,
                        help="copy of the per-image record tree, laid out as it was recorded")
    arguments = parser.parse_args()
    report = analysis.run(arguments.output, pool=arguments.pool,
                          reference_validation=arguments.reference_validation,
                          reference_test=arguments.reference_test,
                          records_dir=arguments.records_dir)
    print(f"Wrote {len(report['written'])} tables to {report['output']}. "
          f"Mixture test passes {report['mixture_test_passes']}/{report['mixture_cells']}.")
    if report["skipped"]:
        print("Skipped without --records-dir: " + ", ".join(report["skipped"]))


if __name__ == "__main__":
    main()

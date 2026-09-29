#!/usr/bin/env python3
"""Paired image-bootstrap reports over per-image Taskonomy records.

Two stages:

  records   one row per named candidate against a shared reference, with componentwise
            95% percentile intervals from resampling images within their building
  settings  the same measurements aggregated over the three fixed training seeds of each
            setting named by a manifest, with seed spread reported as ranges

Both stages read per-image record files that the release omits, so they run only against a
copy of the record tree. ``--records-dir`` resolves the relative source strings a manifest
carries onto that copy.
"""

from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path

from taskscope.paths import RESULTS
from taskscope.taskonomy import paired
from taskscope.taskonomy.analysis import resolve_recorded

TOLERANCES = RESULTS / "taskonomy/pool/development_tolerances.json"
VALIDATION_REFERENCE = RESULTS / "taskonomy/reference/validation_evaluation.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["records", "settings"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate", action="append", metavar="NAME=RECORDS.jsonl",
                        help="records stage: one named per-image record file, repeatable")
    parser.add_argument("--reference", type=Path, help="records stage: reference per-image records")
    parser.add_argument("--manifest", type=Path, help="settings stage: the fixed-setting manifest")
    parser.add_argument("--records-dir", type=Path,
                        help="settings stage: copy of the per-image record tree")
    parser.add_argument("--tolerances", type=Path, default=TOLERANCES)
    parser.add_argument("--validation-reference", type=Path, default=VALIDATION_REFERENCE,
                        help="fixed validation reference the requirement scale is measured against")
    parser.add_argument("--repetitions", type=int, default=paired.REPETITIONS)
    parser.add_argument("--seed", type=int, default=paired.RESAMPLING_SEED)
    arguments = parser.parse_args()

    if arguments.stage == "records":
        if not arguments.candidate or arguments.reference is None:
            parser.error("The records stage needs --reference and at least one --candidate")
        paths = dict(item.split("=", 1) for item in arguments.candidate)
        if len(paths) != len(arguments.candidate) or "reference" in paths:
            raise ValueError("Candidate names must be distinct and cannot be reference")
        result = paired.analyze(paths, arguments.reference, arguments.tolerances,
                                arguments.repetitions, arguments.seed,
                                validation_reference=arguments.validation_reference)
    else:
        if arguments.manifest is None:
            parser.error("The settings stage needs --manifest")
        resolve = (partial(resolve_recorded, arguments.records_dir)
                   if arguments.records_dir is not None else Path)
        result = paired.summarize(json.loads(arguments.manifest.read_text()),
                                  arguments.repetitions, resolve,
                                  validation_reference=arguments.validation_reference)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"Wrote {arguments.output} over {result['images']} images.")


if __name__ == "__main__":
    main()

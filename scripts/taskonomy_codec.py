#!/usr/bin/env python3
r"""Calibrate, train, and evaluate the Taskonomy task-family codec.

Modes:

  calibrate  write semantic class weights, depth statistics, and the edge target mean
  reference  train an uncompressed encoder whose latent is passed through unchanged
  codec      train encoder, entropy model, and decoders against distortion plus rate
  probe      train fresh decoders on a frozen encoder taken from --checkpoint
  evaluate   score --checkpoint on --split, with --actual for entropy-decoded messages

The reported runs use 24000 updates of 12 images. Training and entropy coding need a GPU in
practice. CompressAI supplies the entropy models and is required for every mode except
calibrate.

Example:

  python scripts/taskonomy_codec.py --mode codec --family DSE --lambda-rate 0.03 \
      --data-root /path/to/taskonomy_tiny --output runs/taskonomy/dse_r03
"""

from __future__ import annotations

import argparse
from pathlib import Path

from taskscope.taskonomy import codec


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--mode", choices=("calibrate", "reference", "codec", "probe", "evaluate"), required=True
    )
    parser.add_argument("--family", choices=codec.FAMILIES, default="DSE")
    parser.add_argument("--observation", choices=("rgb", "depth"), default="rgb")
    parser.add_argument(
        "--data-root",
        required=True,
        help="Taskonomy Tiny root holding rgb, segment_semantic, depth_zbuffer, mask_valid",
    )
    parser.add_argument("--output", required=True, help="run directory for checkpoints and logs")
    parser.add_argument(
        "--manifest",
        default=None,
        help="cache of RGB stems per building, rebuilt by scanning --data-root when absent "
        "(default: <output>/taskonomy_manifest.json)",
    )
    parser.add_argument(
        "--calibration",
        default=None,
        help="calibration file, computed when absent; pass a shared path to reuse one "
        "calibration across runs (default: <output>/calibration.json)",
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda")
    parser.add_argument("--calibration-samples", type=int, default=2400)
    parser.add_argument("--train-samples", type=int, default=24000)
    parser.add_argument("--val-samples", type=int, default=500)
    parser.add_argument("--coded-samples", type=int, default=100)
    parser.add_argument("--steps", type=int, default=24000,
                        help="updates per codec, matching the reported fits")
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--channels", type=int, default=96)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--head-channels", type=int, default=48)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--lambda-rate", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--checkpoint", help="encoder to freeze for probe, or model to evaluate")
    parser.add_argument("--resume", help="checkpoint to continue training from")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument(
        "--actual",
        action="store_true",
        help="evaluate entropy-decoded reconstructions and measured complete-packet bytes",
    )
    parser.add_argument("--amp", action="store_true")
    return parser


def main() -> None:
    arguments = build_parser().parse_args()
    Path(arguments.output).mkdir(parents=True, exist_ok=True)
    codec.dispatch(arguments)


if __name__ == "__main__":
    main()

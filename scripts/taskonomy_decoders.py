#!/usr/bin/env python3
"""Select Taskonomy decoders, evaluate decoded messages, and export fixed examples."""

from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    select = commands.add_parser("select", help="choose native or fresh decoders on validation")
    select.add_argument("--codec", type=Path, required=True, help="completed native codec run")
    select.add_argument("--probe", type=Path, required=True, help="completed width-48 DSE probe run")
    select.add_argument("--rgb-evaluation", type=Path, help="coded validation RGB-reference result directory")
    select.add_argument("--codec-evaluation", type=Path, help="same codec's coded 500-image validation directory")
    evaluate = commands.add_parser("evaluate", help="evaluate a fixed selection with actual entropy decoding")
    evaluate.add_argument("--selection", type=Path, required=True)
    rgb = commands.add_parser("rgb-reference", help="evaluate the frozen reference on decoded RGB")
    rgb.add_argument("--codec", required=True, help="RGB checkpoint relative to --checkpoint-root")
    rgb.add_argument("--reference", required=True, help="DSE reference checkpoint relative to --checkpoint-root")
    examples = commands.add_parser("examples", help="export five preselected validation examples")
    examples.add_argument("--heads", type=Path, required=True, help="d_r30, ds_r3 and de_r3 result directories")
    examples.add_argument("--fixed-metadata", type=Path, required=True, help="retained example JSON with the five fixed keys")
    for command in (select, evaluate, rgb, examples):
        command.add_argument("--checkpoint-root", type=Path, default=Path("."),
                             help="root for portable run and checkpoint identifiers")
        command.add_argument("--output", type=Path, required=True)
    for command in (evaluate, rgb, examples):
        command.add_argument("--data-root", type=Path, required=True)
        command.add_argument("--manifest", type=Path, help="dataset manifest cache, defaults to <output>/manifest.json")
        command.add_argument("--device", default="auto")
        command.add_argument("--batch-size", type=int, default=12)
        command.add_argument("--workers", type=int, default=0)
    for command in (evaluate, rgb):
        command.add_argument("--split", choices=("val", "test"), default="val")
        command.add_argument("--samples", type=int, default=500)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    from taskscope.taskonomy import codec, decoding

    try:
        if args.command == "select":
            result = decoding.select_decoders(args.codec, args.probe, args.checkpoint_root,
                                               args.rgb_evaluation, args.codec_evaluation)
            codec.write_json(args.output, result)
            return
        device = codec.resolve_device(args.device)
        if args.batch_size <= 0 or args.workers < 0 or getattr(args, "samples", 500) <= 0:
            raise ValueError("Batch size and samples must be positive, and workers nonnegative")
        args.manifest = args.manifest or args.output / "manifest.json"
        if args.command == "examples":
            decoding.export_examples(args.heads, args.fixed_metadata, args.checkpoint_root,
                                      args, args.output, device)
            return
        selection = (decoding.read_json(args.selection) if args.command == "evaluate" else
                     decoding.rgb_selection(args.codec, args.reference, args.checkpoint_root, device))
        result = decoding.evaluate_selection(selection, args.checkpoint_root, args, args.split,
                                              args.samples, args.output, device)
        if args.command == "rgb-reference":
            result.update(readout_profile="reconstructed_rgb_then_frozen_reference",
                          reference_checkpoint=args.reference,
                          reference_checkpoint_step=selection["heads"]["depth"]["validation_step"])
            codec.write_json(args.output / "evaluation.json", result)
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()

"""Adapted readouts, the frozen validation selection, and the held-out assessment.

Three stages, each run once per codec fit or once in total:

  cache   seal the training and validation splits, capture the frozen suffix's
          penultimate features alongside them, and fit the two adapted readouts
  freeze  compare every candidate against the uncompressed reference on validation,
          fix each task's decoding route, and pick the lowest mean complete rate that
          meets both accuracy requirements
  test    score one selected checkpoint on the held-out split through the route the
          selection fixed, using the readout checkpoints that selection named

Nothing here reads the test split before ``test``, and ``test`` refuses to run against a
selection file whose recorded hashes do not match what it finds.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from . import codec as base
from .data import COARSE_CLASSES, FINE_CLASSES, ImageNetImages, sha256, write_json
from .models import FeatureCodec, FrozenTeacher, SuffixReadout

READOUT_SEEDS = (17, 23)
SUFFIX_FEATURE = "frozen_suffix_penultimate_2048"
SELECTION_RULE = "lowest mean actual bpp satisfying both validation risks"
DECODER_RULE = ("For each task choose inherited or mean of the two adapted readout seeds by "
                "validation accuracy and freeze that route")


def code_suffix(model: FeatureCodec | None, teacher: FrozenTeacher, data: ImageNetImages,
                args: argparse.Namespace, device: torch.device, output: Path) -> dict:
    """Capture the frozen suffix output computed from each decoded packet."""
    cache_path = output.parent / f"{output.name}_penultimate.partial.npy"
    cache = np.lib.format.open_memmap(cache_path, mode="w+", dtype=np.float16,
                                      shape=(len(data), 2048))
    calls, offset = 0, 0

    def capture(_module: torch.nn.Module, _inputs: tuple, value: torch.Tensor) -> None:
        nonlocal calls, offset
        # The pinned code_split computes reference logits, then decoded logits, once per batch.
        if calls % 2 == 1:
            feature = value.detach().flatten(1).cpu().numpy().astype(np.float16)
            cache[offset:offset + len(feature)] = feature
            offset += len(feature)
        calls += 1

    handle = teacher.network.avgpool.register_forward_hook(capture)
    try:
        report = base.code_split(model, teacher, data, args, device, output)
    finally:
        handle.remove()
    if calls != 2 * math.ceil(len(data) / args.batch_size) or offset != len(data):
        raise RuntimeError("Pinned reference/decoded suffix call contract changed")
    cache.flush()
    cache = None  # Drop the last reference so the mapping closes before the rename.
    cache_path.rename(output / "penultimate.npy")
    write_json(output / "suffix_sealed.json",
               {"feature": SUFFIX_FEATURE,
                "penultimate_sha256": sha256(output / "penultimate.npy"),
                "base_seal_sha256": sha256(output / "sealed.json"), "images": len(data),
                "test_used": data.split == "test",
                "assessment_source_sha256": sha256(Path(__file__))})
    return report


def fit_suffix_readouts(args: argparse.Namespace, device: torch.device) -> dict:
    """Fit both adapted readouts on sealed training features, selected on validation."""
    started, caches, labels, seals = time.perf_counter(), {}, {}, {}
    for split in ("train", "validation"):
        root = args.output / split
        seal = json.loads((root / "sealed.json").read_text())
        suffix = json.loads((root / "suffix_sealed.json").read_text())
        if seal["test_used"] or suffix["test_used"] or seal["dataset"]["split"] != split:
            raise ValueError("Adaptation requires training and development examples")
        if (suffix["base_seal_sha256"] != sha256(root / "sealed.json")
                or suffix["penultimate_sha256"] != sha256(root / "penultimate.npy")):
            raise ValueError("Sealed suffix features changed")
        for name, expected in seal["files_sha256"].items():
            if sha256(root / name) != expected:
                raise ValueError("Sealed message or targets changed")
        seals[split] = {"base": seal, "suffix": suffix}
        caches[split] = np.load(root / "penultimate.npy", mmap_mode="r")
        with np.load(root / "predictions.npz", allow_pickle=False) as saved:
            labels[split] = {name: torch.tensor(saved[f"{name}_target"], device=device)
                             for name in ("fine", "coarse")}
    for key in ("checkpoint_sha256", "teacher", "provenance", "mapping_sha256"):
        if seals["train"]["base"][key] != seals["validation"]["base"][key]:
            raise ValueError("Training and validation message identities differ")
    training = torch.tensor(np.asarray(caches["train"]), device=device, dtype=torch.float32)
    validation = torch.tensor(np.asarray(caches["validation"]), device=device, dtype=torch.float32)
    mean = training.mean(0)
    scale = training.std(0, correction=0).clamp_min(1e-3)
    training, validation = (training - mean) / scale, (validation - mean) / scale
    results = {}
    for seed in args.readout_seeds:
        base.seed_all(seed)
        stage = base.start_stage(device)
        model = SuffixReadout().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.readout_lr, weight_decay=1e-4)
        best = float("inf")
        for epoch in range(args.readout_epochs):
            model.train()
            order = torch.randperm(len(training), device=device)
            for begin in range(0, len(training), args.readout_batch_size):
                rows = order[begin:begin + args.readout_batch_size]
                fine, coarse = model(training[rows])
                loss = F.cross_entropy(fine, labels["train"]["fine"][rows]) / math.log(FINE_CLASSES)
                loss += (F.cross_entropy(coarse, labels["train"]["coarse"][rows])
                         / math.log(COARSE_CLASSES))
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            model.eval()
            correct, ce, predictions = np.zeros(2), 0.0, []
            with torch.no_grad():
                for begin in range(0, len(validation), args.readout_batch_size):
                    rows = slice(begin, begin + args.readout_batch_size)
                    fine, coarse = model(validation[rows])
                    y_fine = labels["validation"]["fine"][rows]
                    y_coarse = labels["validation"]["coarse"][rows]
                    ce += (F.cross_entropy(fine, y_fine, reduction="sum") / math.log(FINE_CLASSES)
                           + F.cross_entropy(coarse, y_coarse, reduction="sum")
                           / math.log(COARSE_CLASSES)).item()
                    correct += [(fine.argmax(1) == y_fine).sum().item(),
                                (coarse.argmax(1) == y_coarse).sum().item()]
                    predictions.append(
                        torch.stack((fine.argmax(1), coarse.argmax(1)), 1).cpu().numpy())
            if ce < best:
                best = ce
                torch.save({"model": model.state_dict(), "mean": mean.cpu(), "scale": scale.cpu(),
                            "seed": seed, "epoch": epoch + 1, "seals": seals,
                            "feature": SUFFIX_FEATURE}, args.output / f"readout_s{seed}.pt")
                np.save(args.output / f"readout_s{seed}_validation.npy",
                        np.concatenate(predictions))
                results[str(seed)] = {"selected_epoch": epoch + 1,
                                      "fine_accuracy": float(correct[0] / len(validation)),
                                      "coarse_accuracy": float(correct[1] / len(validation)),
                                      "normalized_ce_sum": ce / len(validation)}
        results[str(seed)].update(**base.finish_stage(stage, device),
                                  image_exposures=len(training) * args.readout_epochs)
        del model, optimizer
    report = {"readouts": results, "feature": SUFFIX_FEATURE, "test_used": False,
              "readout_epochs": args.readout_epochs, "readout_lr": args.readout_lr,
              "readout_batch_size": args.readout_batch_size,
              "seconds": time.perf_counter() - started}
    write_json(args.output / "readouts.json", report)
    print(json.dumps(report), flush=True)
    return report


def read_validation(directory: Path) -> dict:
    """Validation metrics of one candidate under its better decoding route per task."""
    seal = json.loads((directory / "validation/sealed.json").read_text())
    if seal["test_used"] or seal["dataset"]["split"] != "validation":
        raise ValueError("Selection requires development validation results")
    fitted = json.loads((directory / "readouts.json").read_text())
    if fitted["test_used"] or set(fitted["readouts"]) != {str(seed) for seed in READOUT_SEEDS}:
        raise ValueError("Selection requires both declared adapted-readout seeds")
    adapted = float(np.mean([row["fine_accuracy"] for row in fitted["readouts"].values()]))
    adapted_coarse = float(np.mean([row["coarse_accuracy"] for row in fitted["readouts"].values()]))
    inherited = seal["fine_accuracy"]
    return {"seal": seal, "readouts": fitted,
            "fine_accuracy": max(adapted, inherited), "adapted_fine_accuracy": adapted,
            "fine_route": "inherited" if inherited >= adapted else "adapted_seed_mean",
            "coarse_accuracy": max(adapted_coarse, seal["coarse_accuracy"]),
            "adapted_coarse_accuracy": adapted_coarse,
            "coarse_route": ("inherited" if seal["coarse_accuracy"] >= adapted_coarse
                             else "adapted_seed_mean"),
            "actual_bpp": seal["actual_bpp"]}


def pooled_requirement_selection(groups: list[dict], thresholds: dict) -> dict:
    """Compare nested requirement sets over exactly the same fitted candidates."""
    selected = {}
    for requirement, metrics in (("original_coarse", ("coarse_accuracy",)),
                                 ("expanded_joint", ("coarse_accuracy", "fine_accuracy"))):
        feasible = [group for group in groups
                    if all(group[metric] >= thresholds[metric] for metric in metrics)]
        selected[requirement] = min(feasible, key=lambda group: group["actual_bpp"]) if feasible else None
    return selected


def freeze_selection(args: argparse.Namespace) -> None:
    """Choose each objective's lowest demonstrated feasible development rate."""
    reference = read_validation(args.reference)
    if not reference["seal"]["raw_reference"]:
        raise ValueError("Reference must use uncompressed features")
    if reference["seal"]["mapping_sha256"] != sha256(args.mapping):
        raise ValueError("Reference mapping differs from the selected mapping file")
    thresholds = {"coarse_accuracy": reference["coarse_accuracy"] - args.coarse_drop,
                  "fine_accuracy": reference["fine_accuracy"] - args.fine_drop}
    candidates, grouped = [], {}
    for directory in args.candidates:
        result = read_validation(directory)
        metadata = json.loads((directory / "assessment_run.json").read_text())
        checkpoint = Path(metadata["checkpoint"])
        fit = json.loads((checkpoint.parent / "fit.json").read_text())
        if result["seal"]["checkpoint_sha256"] != sha256(checkpoint):
            raise ValueError("Validation predictions were produced by a different checkpoint")
        if (result["seal"]["teacher"] != reference["seal"]["teacher"]
                or result["seal"]["provenance"] != reference["seal"]["provenance"]):
            raise ValueError("Candidate and reference teacher or coding implementation differ")
        saved_fit = torch.load(checkpoint, map_location="cpu", weights_only=True)
        fit_keys = ("args", "image_exposures", "initial_state_sha256", "mapping_sha256",
                    "provenance", "teacher")
        if any(saved_fit[key] != fit[key] for key in fit_keys):
            raise ValueError("Fit metadata differs from the actual candidate checkpoint")
        del saved_fit
        if (result["seal"]["dataset"]["membership_sha256"]
                != reference["seal"]["dataset"]["membership_sha256"]
                or result["seal"]["mapping_sha256"] != reference["seal"]["mapping_sha256"]):
            raise ValueError("Candidate and raw-reference development memberships differ")
        train = json.loads((directory / "train/sealed.json").read_text())
        raw_train = json.loads((args.reference / "train/sealed.json").read_text())
        if train["dataset"]["membership_sha256"] != raw_train["dataset"]["membership_sha256"]:
            raise ValueError("Adapted training memberships differ")
        for key in ("checkpoint_sha256", "teacher", "provenance", "mapping_sha256"):
            if train[key] != result["seal"][key]:
                raise ValueError("Candidate training and validation message identities differ")
        if any(result["readouts"][key] != reference["readouts"][key]
               for key in ("readout_epochs", "readout_lr", "readout_batch_size", "feature")):
            raise ValueError("Candidate and raw-reference readout fitting budgets differ")
        row = {"directory": str(directory), "checkpoint": str(checkpoint),
               "checkpoint_sha256": sha256(checkpoint), "objective": fit["args"]["objective"],
               "beta": fit["args"]["beta"], "seed": fit["args"]["seed"],
               "image_exposures": fit["image_exposures"],
               "initial_state_sha256": fit["initial_state_sha256"],
               "coarse_accuracy": result["coarse_accuracy"], "fine_accuracy": result["fine_accuracy"],
               "fine_route": result["fine_route"],
               "adapted_fine_accuracy": result["adapted_fine_accuracy"],
               "coarse_route": result["coarse_route"],
               "adapted_coarse_accuracy": result["adapted_coarse_accuracy"],
               "actual_bpp": result["actual_bpp"],
               "validation_seal_sha256": sha256(directory / "validation/sealed.json"),
               "readout_sha256": {str(seed): sha256(directory / f"readout_s{seed}.pt")
                                  for seed in READOUT_SEEDS}}
        row["feasible"] = all(row[name] >= value for name, value in thresholds.items())
        candidates.append(row)
        grouped.setdefault((row["objective"], row["beta"]), []).append(row)
    paired = {}
    for row in candidates:
        paired.setdefault((row["seed"], row["beta"]), []).append(row)
    for rows in paired.values():
        if len(rows) != 2 or {row["objective"] for row in rows} != set(base.TASK_OBJECTIVES):
            raise ValueError("Final comparison requires both objectives for every declared beta "
                             "and codec seed")
        if (len({row["image_exposures"] for row in rows}) != 1
                or len({row["initial_state_sha256"] for row in rows}) != 1):
            raise ValueError("Matched objective fits differ in initialization or training exposure")
    groups = []
    for (objective, beta), rows in grouped.items():
        if len({row["seed"] for row in rows}) != len(rows):
            raise ValueError("Duplicate codec seed within one objective and rate")
        group = {"objective": objective, "beta": beta,
                 "seeds": sorted(row["seed"] for row in rows),
                 "directories": [row["directory"] for row in rows]}
        for metric in ("coarse_accuracy", "fine_accuracy", "actual_bpp"):
            group[metric] = float(np.mean([row[metric] for row in rows]))
        group["feasible"] = all(group[name] >= value for name, value in thresholds.items())
        groups.append(group)
    selected = {}
    for objective in base.TASK_OBJECTIVES:
        feasible = [group for group in groups
                    if group["objective"] == objective and group["feasible"]]
        selected[objective] = min(feasible, key=lambda row: row["actual_bpp"]) if feasible else None
    report = {"test_used": False, "selection": SELECTION_RULE, "decoder_rule": DECODER_RULE,
              "coarse_drop": args.coarse_drop, "fine_drop": args.fine_drop,
              "thresholds": thresholds, "reference": str(args.reference),
              "reference_metrics": reference,
              "reference_readout_sha256": {str(seed): sha256(args.reference / f"readout_s{seed}.pt")
                                           for seed in READOUT_SEEDS},
              "candidates": candidates, "groups": groups, "selected": selected,
              "pooled_selected": pooled_requirement_selection(groups, thresholds),
              "mapping_sha256": sha256(args.mapping),
              "assessment_source_sha256": sha256(Path(__file__)),
              "confirmation_units": "codec seeds, with two adapted-readout seeds per codec"}
    write_json(args.output / "selection.json", report)
    print(json.dumps({"thresholds": thresholds, "groups": groups, "selected": selected,
                      "pooled_selected": report["pooled_selected"]}), flush=True)


@torch.no_grad()
def score_readouts(source: Path, target: Path, device: torch.device, expected: dict) -> dict:
    """Apply the two frozen readouts to one held-out split without refitting."""
    features = np.load(target / "penultimate.npy", mmap_mode="r")
    with np.load(target / "predictions.npz", allow_pickle=False) as saved:
        fine_labels, coarse_labels = saved["fine_target"], saved["coarse_target"]
    results = {}
    for seed in READOUT_SEEDS:
        path = source / f"readout_s{seed}.pt"
        if sha256(path) != expected[str(seed)]:
            raise ValueError("Frozen readout checkpoint changed")
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        model = SuffixReadout().to(device).eval()
        model.load_state_dict(checkpoint["model"])
        mean, scale = checkpoint["mean"].to(device), checkpoint["scale"].to(device)
        predictions = []
        for begin in range(0, len(features), 256):
            value = torch.tensor(np.asarray(features[begin:begin + 256]), device=device,
                                 dtype=torch.float32)
            fine, coarse = model((value - mean) / scale)
            predictions.append(torch.stack((fine.argmax(1), coarse.argmax(1)), 1).cpu().numpy())
        prediction = np.concatenate(predictions)
        np.save(target / f"readout_s{seed}_predictions.npy", prediction)
        results[str(seed)] = {"fine_accuracy": float((prediction[:, 0] == fine_labels).mean()),
                              "coarse_accuracy": float((prediction[:, 1] == coarse_labels).mean())}
    write_json(target / "adapted_results.json", {"readouts": results, "test_used": True})
    return results


def run_test(args: argparse.Namespace, teacher: FrozenTeacher, model: FeatureCodec | None,
             device: torch.device, mapping: dict) -> None:
    """Score one frozen selection on the held-out split through its fixed route."""
    selection = json.loads(args.selection.read_text())
    if selection["test_used"] or selection["mapping_sha256"] != sha256(args.mapping):
        raise ValueError("Invalid frozen selection")
    if selection["assessment_source_sha256"] != sha256(Path(__file__)):
        raise ValueError("Assessment implementation changed after selection")
    if model is None:
        source = Path(selection["reference"])
        expected = selection["reference_readout_sha256"]
        fine_route = selection["reference_metrics"]["fine_route"]
        coarse_route = selection["reference_metrics"]["coarse_route"]
    else:
        selected_groups = list(selection["selected"].values()) + list(
            selection["pooled_selected"].values())
        selected_dirs = {directory for group in selected_groups if group
                         for directory in group["directories"]}
        matches = [row for row in selection["candidates"]
                   if row["checkpoint_sha256"] == sha256(args.checkpoint)
                   and row["directory"] in selected_dirs]
        if len(matches) != 1:
            raise ValueError("Checkpoint was not uniquely selected before test scoring")
        source, expected = Path(matches[0]["directory"]), matches[0]["readout_sha256"]
        fine_route, coarse_route = matches[0]["fine_route"], matches[0]["coarse_route"]
    data = ImageNetImages(args.data_root, mapping, "test")
    inherited = code_suffix(model, teacher, data, args, device, args.output / "test")
    adapted = score_readouts(source, args.output / "test", device, expected)
    write_json(args.output / "test_summary.json",
               {"inherited": inherited, "adapted": adapted,
                "primary_coarse_accuracy": (inherited["coarse_accuracy"]
                                            if coarse_route == "inherited" else
                                            float(np.mean([row["coarse_accuracy"]
                                                           for row in adapted.values()]))),
                "fine_route": fine_route, "coarse_route": coarse_route,
                "primary_fine_accuracy": (inherited["fine_accuracy"] if fine_route == "inherited"
                                          else float(np.mean([row["fine_accuracy"]
                                                              for row in adapted.values()]))),
                "selection_sha256": sha256(args.selection), "test_used": True})


def dispatch(args: argparse.Namespace) -> None:
    """Run one assessment stage."""
    args.output.mkdir(parents=True, exist_ok=False)
    if args.mode == "freeze":
        freeze_selection(args)
        return
    device = torch.device(args.device)
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    base.seed_all(args.seed)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    mapping = json.loads(args.mapping.read_text())
    teacher = FrozenTeacher(mapping).to(device).eval()
    model = (base.load_codec(args.checkpoint, args.mapping, device, teacher.identity)
             if args.checkpoint else None)
    write_json(args.output / "assessment_run.json",
               {"mode": args.mode,
                "checkpoint": str(args.checkpoint) if args.checkpoint else None,
                "mapping_sha256": sha256(args.mapping),
                "assessment_source_sha256": sha256(Path(__file__)),
                "base_provenance": base.provenance(), "test_used": args.mode == "test"})
    if args.mode == "cache":
        for split, per_class in (("train", args.train_per_class), ("validation", None)):
            data = ImageNetImages(args.data_root, mapping, split, per_class)
            code_suffix(model, teacher, data, args, device, args.output / split)
        del model, teacher
        fit_suffix_readouts(args, device)
    else:
        run_test(args, teacher, model, device, mapping)

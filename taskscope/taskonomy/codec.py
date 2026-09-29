"""Training, calibration, and evaluation of the Taskonomy task-family codec.

One encoder produces a single coded latent and every task in the family reads it. The
families differ only in which decoders are trained, so a family comparison isolates what the
message has to preserve.

Modes:

  calibrate  semantic class weights, depth statistics, and the edge target mean
  reference  train an uncompressed encoder, with the entropy model frozen and unused
  codec      train encoder, entropy model, and decoders against distortion plus rate
  probe      train fresh decoders on a frozen encoder, so a readout is added after the fact
  evaluate   score a checkpoint, optionally against entropy-decoded reconstructions

Loss terms are weighted 10 for depth, 1 for semantics, 10 for edges, and 100 for RGB. Depth
is a Smooth-L1 loss on the log of depths clamped below at 1e-4, semantics is a cross entropy
over 17 labels with uncertain pixels excluded and inverse-square-root frequency weights, and
edges and RGB use squared error. Evaluation normalises each image by its own valid-pixel
count, or by its valid semantic weight sum, before averaging over images.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from shutil import copy2
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader

from . import models
from .data import (
    NUM_SEMANTIC_CLASSES,
    SEMANTIC_CLASS_NAMES,
    SEMANTIC_IGNORE_INDEX,
    TaskonomyRequirements,
    move_batch,
)
from .edges import fixed_gaussian_sobel

FAMILIES = {
    "D": ("depth",),
    "DS": ("depth", "semantic"),
    "DE": ("depth", "edge"),
    "DSE": ("depth", "semantic", "edge"),
    "RGB": ("rgb",),
}
LOSS_SCALES = {"depth": 10.0, "semantic": 1.0, "edge": 10.0, "rgb": 100.0}


def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def log_record(path: Path, value: Mapping[str, Any]) -> None:
    line = json.dumps(value, allow_nan=False)
    with path.open("a") as handle:
        handle.write(line + "\n")
    print(line, flush=True)


class MetricAccumulator:
    """Task metrics pooled over pixels, not over images."""

    def __init__(self) -> None:
        self.rgb_squared_error = 0.0
        self.rgb_values = 0
        self.confusion = torch.zeros(
            (NUM_SEMANTIC_CLASSES, NUM_SEMANTIC_CLASSES), dtype=torch.int64
        )
        self.depth_squared_error = 0.0
        self.depth_abs_relative = 0.0
        self.depth_delta = [0, 0, 0]
        self.depth_values = 0

    @torch.no_grad()
    def update_task(self, task: str, prediction: Tensor, batch: Mapping[str, Any]) -> None:
        if task == "rgb":
            self._update_rgb(prediction, batch["rgb"])
        elif task == "semantic":
            self._update_semantic(prediction, batch["semantic"])
        elif task == "depth":
            self._update_depth(prediction, batch["depth"], batch["depth_valid"])
        else:
            raise ValueError(f"unknown task {task!r}")

    def _update_rgb(self, prediction: Tensor, target: Tensor) -> None:
        rgb_difference = (prediction - target).double()
        self.rgb_squared_error += float((rgb_difference * rgb_difference).sum())
        self.rgb_values += rgb_difference.numel()

    def _update_semantic(self, prediction: Tensor, semantic_target: Tensor) -> None:
        semantic_prediction = prediction.argmax(dim=1)
        valid_semantic = semantic_target != SEMANTIC_IGNORE_INDEX
        pairs = (
            semantic_target[valid_semantic] * NUM_SEMANTIC_CLASSES
            + semantic_prediction[valid_semantic]
        ).cpu()
        self.confusion += torch.bincount(pairs, minlength=NUM_SEMANTIC_CLASSES**2).reshape(
            NUM_SEMANTIC_CLASSES, NUM_SEMANTIC_CLASSES
        )

    def _update_depth(self, prediction: Tensor, target: Tensor, valid_depth: Tensor) -> None:
        predicted_depth = prediction[valid_depth].double().clamp_min(1e-4)
        target_depth = target[valid_depth].double().clamp_min(1e-4)
        error = predicted_depth - target_depth
        self.depth_squared_error += float((error * error).sum())
        self.depth_abs_relative += float((error.abs() / target_depth).sum())
        ratio = torch.maximum(predicted_depth / target_depth, target_depth / predicted_depth)
        for index, threshold in enumerate((1.25, 1.25**2, 1.25**3)):
            self.depth_delta[index] += int((ratio < threshold).sum())
        self.depth_values += target_depth.numel()

    def compute(self) -> dict[str, Any]:
        mse = self.rgb_squared_error / max(1, self.rgb_values)
        intersection = self.confusion.diag().double()
        union = self.confusion.sum(0) + self.confusion.sum(1) - intersection
        present = self.confusion.sum(1) > 0
        per_class = torch.full((NUM_SEMANTIC_CLASSES,), float("nan"))
        per_class[present] = (intersection[present] / union[present]).float()
        total_semantic = int(self.confusion.sum())
        return {
            "rgb": {"mse": mse, "psnr_db": -10.0 * math.log10(max(mse, 1e-12))},
            "semantic": {
                "miou": float(per_class[present].mean()) if torch.any(present) else float("nan"),
                "miou_objects": (
                    float(per_class[1:][present[1:]].mean())
                    if torch.any(present[1:])
                    else float("nan")
                ),
                "pixel_accuracy": float(intersection.sum() / max(1, total_semantic)),
                "class_names": list(SEMANTIC_CLASS_NAMES),
                "per_class_iou": [
                    None if math.isnan(float(value)) else float(value) for value in per_class
                ],
                "confusion": self.confusion.tolist(),
            },
            "depth": {
                "rmse_m": math.sqrt(self.depth_squared_error / max(1, self.depth_values)),
                "abs_rel": self.depth_abs_relative / max(1, self.depth_values),
                "delta_1": self.depth_delta[0] / max(1, self.depth_values),
                "delta_2": self.depth_delta[1] / max(1, self.depth_values),
                "delta_3": self.depth_delta[2] / max(1, self.depth_values),
                "valid_pixels": self.depth_values,
            },
        }


def losses(
    predictions: Mapping[str, Tensor], batch: Mapping[str, Any], weights: Tensor
) -> dict[str, Tensor]:
    """Training losses, each already averaged over the valid elements of the batch."""
    result = {}
    for task, prediction in predictions.items():
        if task == "depth":
            valid = batch["depth_valid"]
            if not valid.any():
                raise ValueError("No valid depth pixels in batch")
            result[task] = F.smooth_l1_loss(
                prediction[valid].clamp_min(1e-4).log(), batch["depth"][valid].clamp_min(1e-4).log()
            )
        elif task == "semantic":
            result[task] = F.cross_entropy(
                prediction, batch["semantic"], weight=weights, ignore_index=SEMANTIC_IGNORE_INDEX
            )
        else:
            result[task] = F.mse_loss(prediction, batch[task])
    return result


def prepared(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    batch = move_batch(batch, device)
    batch["edge"] = fixed_gaussian_sobel(batch["rgb"])
    return batch


def observation_input(
    batch: Mapping[str, Any], calibration: Mapping[str, Any], observation: str
) -> Tensor:
    """Encoder input. A depth observation is log-scaled, masked, and padded to three channels."""
    if observation == "rgb":
        return batch["rgb"]
    if observation == "depth":
        scaled = torch.log1p(batch["depth"] / calibration["depth_geometric_mean_m"])
        return torch.cat((scaled, batch["depth_valid"].float(), torch.zeros_like(scaled)), dim=1)
    raise ValueError(f"Unknown observation {observation}")


def evaluation_loss_totals(
    predictions: Mapping[str, Tensor], batch: Mapping[str, Any], weights: Tensor
) -> dict[str, tuple[float, int]]:
    """Sum per-image risks with exact within-image valid-pixel normalization."""
    result = {}
    for task, prediction in predictions.items():
        if task == "depth":
            valid = batch["depth_valid"]
            error = F.smooth_l1_loss(
                prediction.clamp_min(1e-4).log(),
                batch["depth"].clamp_min(1e-4).log(),
                reduction="none",
            )
            numerator = (error * valid).flatten(1).sum(1)
            denominator = valid.flatten(1).sum(1)
        elif task == "semantic":
            target = batch["semantic"]
            valid = target != SEMANTIC_IGNORE_INDEX
            error = F.cross_entropy(
                prediction,
                target,
                weight=weights,
                ignore_index=SEMANTIC_IGNORE_INDEX,
                reduction="none",
            )
            numerator = error.flatten(1).sum(1)
            denominator = (weights[target.masked_fill(~valid, 0)] * valid).flatten(1).sum(1)
        else:
            error = (prediction - batch[task]).square()
            numerator = error.flatten(1).mean(1)
            denominator = torch.ones_like(numerator)
        observed = denominator > 0
        per_image = numerator[observed] / denominator[observed]
        result[task] = (per_image.double().sum().item(), int(observed.sum()))
    return result


def loader(args: argparse.Namespace, split: str, count: int, shuffle: bool = False) -> DataLoader:
    dataset = TaskonomyRequirements(Path(args.data_root), split, count, Path(args.manifest))
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
        drop_last=shuffle,
    )


@torch.no_grad()
def calibrate(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    """Training-split statistics: class weights, depth means, and the edge target moments."""
    data = loader(args, "train", args.calibration_samples)
    counts = torch.zeros(NUM_SEMANTIC_CLASSES, dtype=torch.float64, device=device)
    totals: defaultdict[str, float] = defaultdict(float)
    keys: list[str] = []
    for raw in data:
        batch = prepared(raw, device)
        target = batch["semantic"]
        counts += torch.bincount(
            target[target != SEMANTIC_IGNORE_INDEX].flatten(), minlength=NUM_SEMANTIC_CLASSES
        )
        valid = batch["depth_valid"]
        depth = batch["depth"][valid].double()
        totals["depth_sum"] += depth.sum().item()
        totals["depth_log_sum"] += depth.clamp_min(1e-4).log().sum().item()
        totals["depth_pixels"] += depth.numel()
        edge = batch["edge"].double()
        totals["edge_sum"] += edge.sum().item()
        totals["edge_squared_sum"] += edge.square().sum().item()
        totals["edge_pixels"] += edge.numel()
        keys.extend(batch["key"])
    frequencies = counts / counts.sum()
    weights = frequencies.clamp_min(1 / counts.sum()).rsqrt()
    weights /= (weights * frequencies).sum()
    edge_mean = totals["edge_sum"] / totals["edge_pixels"]
    value = {
        "split": "train",
        "images": len(keys),
        "keys": keys,
        "semantic_counts": counts.long().cpu().tolist(),
        "semantic_weights": weights.float().cpu().tolist(),
        "semantic_weight_rule": (
            "inverse square root of training pixel frequency, normalized to "
            "frequency-weighted mean one"
        ),
        "depth_mean_m": totals["depth_sum"] / totals["depth_pixels"],
        "depth_geometric_mean_m": math.exp(totals["depth_log_sum"] / totals["depth_pixels"]),
        "edge_mean": edge_mean,
        "edge_variance": totals["edge_squared_sum"] / totals["edge_pixels"] - edge_mean**2,
        "loss_scales": LOSS_SCALES,
    }
    write_json(Path(args.calibration), value)
    return value


@torch.no_grad()
def evaluate(
    model: models.FamilyCodec,
    data: DataLoader,
    device: torch.device,
    tasks: Sequence[str],
    compressed: bool,
    weights: Tensor,
    calibration: Mapping[str, Any],
    coded: bool = False,
    per_image_path: Path | None = None,
    observation: str = "rgb",
) -> dict[str, Any]:
    """Score one checkpoint. ``coded`` replaces estimated rates by measured packet bytes."""
    model.eval()
    if coded:
        model.stream.update(force=True)
    accumulator = MetricAccumulator()
    constant = MetricAccumulator()
    totals: defaultdict[str, float] = defaultdict(float)
    image_count = 0
    records: list[dict[str, Any]] = []
    for raw in data:
        batch = prepared(raw, device)
        inputs = observation_input(batch, calibration, observation)
        if coded:
            latents = []
            for i, rgb in enumerate(inputs):
                try:
                    latent, packet = model.code(rgb[None])
                except AssertionError as error:
                    raise AssertionError(
                        f"Actual coding failed for {batch['key'][i]}: {error}"
                    ) from error
                latents.append(latent)
                bpp = 8 * packet.total_bytes / (rgb.shape[-2] * rgb.shape[-1])
                records.append(
                    {
                        "key": batch["key"][i],
                        "bpp": bpp,
                        "bytes": packet.total_bytes,
                        "main_bytes": packet.main_bytes,
                        "hyper_bytes": packet.hyper_bytes,
                        "header_bytes": packet.header_bytes,
                    }
                )
                totals["actual_bpp"] += bpp
            latent = torch.cat(latents)
            rate = latent.new_zeros(())
        else:
            latent, rate = model.encode(inputs, compressed)
            if per_image_path is not None:
                records.extend({"key": key} for key in batch["key"])
        predictions = model.predict(latent, tasks)
        task_losses = evaluation_loss_totals(predictions, batch, weights)
        constant_predictions = {}
        for task, prediction in predictions.items():
            if task == "semantic":
                prior = (prediction.new_tensor(calibration["semantic_counts"]) + 1) * weights
                logits = (prior / prior.sum()).log()
                constant_predictions[task] = logits[None, :, None, None].expand_as(prediction)
            else:
                value = {
                    "depth": calibration["depth_geometric_mean_m"],
                    "edge": calibration["edge_mean"],
                    "rgb": 0.5,
                }[task]
                constant_predictions[task] = torch.full_like(prediction, value)
        constant_losses = evaluation_loss_totals(constant_predictions, batch, weights)
        count = len(batch["key"])
        if coded or per_image_path is not None:
            for i in range(count):
                single_predictions = {task: value[i : i + 1] for task, value in predictions.items()}
                single_batch = {key: value[i : i + 1] for key, value in batch.items()}
                record = records[len(records) - count + i]
                record["losses"] = {
                    task: total / observed if observed else None
                    for task, (total, observed) in evaluation_loss_totals(
                        single_predictions, single_batch, weights
                    ).items()
                }
                single_constants = {
                    task: value[i : i + 1] for task, value in constant_predictions.items()
                }
                record["constant_losses"] = {
                    task: total / observed if observed else None
                    for task, (total, observed) in evaluation_loss_totals(
                        single_constants, single_batch, weights
                    ).items()
                }
                if "semantic" in predictions:
                    target = batch["semantic"][i]
                    predicted = predictions["semantic"][i].argmax(0)
                    valid = target != SEMANTIC_IGNORE_INDEX
                    pairs = target[valid] * NUM_SEMANTIC_CLASSES + predicted[valid]
                    record["semantic_confusion"] = (
                        torch.bincount(pairs, minlength=NUM_SEMANTIC_CLASSES**2)
                        .reshape(NUM_SEMANTIC_CLASSES, NUM_SEMANTIC_CLASSES)
                        .cpu()
                        .tolist()
                    )
                if "depth" in predictions:
                    valid = batch["depth_valid"][i]
                    target = batch["depth"][i][valid].double().clamp_min(1e-4)
                    predicted = predictions["depth"][i][valid].double().clamp_min(1e-4)
                    error = predicted - target
                    record["depth"] = {
                        "pixels": target.numel(),
                        "squared_error_sum": error.square().sum().item(),
                        "abs_relative_sum": (error.abs() / target).sum().item(),
                        "delta_1_count": int(
                            (torch.maximum(predicted / target, target / predicted) < 1.25).sum()
                        ),
                    }
        image_count += count
        totals["estimated_bpp"] += rate.item() * count
        for task, (loss_sum, observed_images) in task_losses.items():
            totals[f"loss_{task}"] += loss_sum
            totals[f"loss_images_{task}"] += observed_images
            totals[f"constant_loss_{task}"] += constant_losses[task][0]
        for task, prediction in predictions.items():
            if task == "edge":
                target = batch["edge"]
                totals["edge_error"] += (prediction - target).square().sum().item()
                totals["edge_constant_error"] += (
                    (calibration["edge_mean"] - target).square().sum().item()
                )
                totals["edge_values"] += target.numel()
            else:
                accumulator.update_task(task, prediction, batch)
                if task == "depth":
                    constant.update_task(
                        task, torch.full_like(prediction, calibration["depth_mean_m"]), batch
                    )
                elif task == "semantic":
                    baseline = torch.zeros_like(prediction)
                    baseline[:, int(torch.tensor(calibration["semantic_counts"]).argmax())] = 1
                    constant.update_task(task, baseline, batch)
    all_metrics = accumulator.compute()
    constant_metrics = constant.compute()
    metrics = {task: all_metrics[task] for task in tasks if task != "edge"}
    if "edge" in tasks:
        metrics["edge"] = {
            "mse": totals["edge_error"] / totals["edge_values"],
            "constant_mse": totals["edge_constant_error"] / totals["edge_values"],
            "normalized_mse": totals["edge_error"] / totals["edge_constant_error"],
        }
    result = {
        "images": image_count,
        "tasks": list(tasks),
        "metrics": metrics,
        "encoder_observation": observation,
        "entropy_convolution_policy": (
            "deterministic cuDNN with benchmarking disabled during entropy coding"
            if coded
            else None
        ),
        "constant_metrics": {
            task: constant_metrics[task] for task in tasks if task in ("depth", "semantic")
        },
        "losses": {
            task: totals[f"loss_{task}"] / totals[f"loss_images_{task}"] for task in tasks
        },
        "loss_aggregation": (
            "mean over images with valid targets, with within-image pixel normalization"
        ),
        "loss_images": {task: int(totals[f"loss_images_{task}"]) for task in tasks},
        "constant_losses": {
            task: totals[f"constant_loss_{task}"] / totals[f"loss_images_{task}"] for task in tasks
        },
        "constant_loss_definitions": {
            "depth": "training geometric mean depth",
            "semantic": (
                "add-one-smoothed training class counts multiplied by class loss weights "
                "and normalized"
            ),
            "edge": "training mean edge target",
            "rgb": "constant 0.5",
        },
        "estimated_bpp": None if coded else totals["estimated_bpp"] / image_count,
        "actual_bpp": totals["actual_bpp"] / image_count if coded else None,
    }
    if per_image_path is not None:
        per_image_path.write_text("".join(json.dumps(record) + "\n" for record in records))
    return result


def load_model(path: Path | str, device: torch.device) -> tuple[models.FamilyCodec, dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model = models.FamilyCodec(
        config["channels"], config["base_channels"], config["head_channels"]
    ).to(device)
    state = checkpoint["model"]
    for name, module, buffers in (
        (
            "stream.entropy_bottleneck",
            model.stream.entropy_bottleneck,
            ["_quantized_cdf", "_offset", "_cdf_length"],
        ),
        (
            "stream.gaussian",
            model.stream.gaussian,
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
        ),
    ):
        models.load_entropy_buffers(module, name, buffers, state)
    model.load_state_dict(state)
    return model, checkpoint


def checkpoint_compressed(checkpoint: Mapping[str, Any]) -> bool:
    if "representation_compressed" in checkpoint:
        return checkpoint["representation_compressed"]
    mode = checkpoint["config"]["mode"]
    if mode == "probe":
        raise ValueError(
            "Probe checkpoint does not identify whether its representation was compressed"
        )
    return mode == "codec"


def train(args: argparse.Namespace, device: torch.device, calibration: Mapping[str, Any]) -> None:
    set_seed(args.seed)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    tasks = FAMILIES[args.family]
    compressed = args.mode != "reference"
    start_step = 0
    checkpoint = None
    if args.resume:
        model, checkpoint = load_model(args.resume, device)
        start_step = checkpoint["step"]
        prior = checkpoint["config"]
        for name in (
            "family",
            "mode",
            "channels",
            "base_channels",
            "head_channels",
            "lambda_rate",
            "observation",
        ):
            previous = prior.get(name, "rgb") if name == "observation" else prior[name]
            if previous != getattr(args, name):
                raise ValueError(f"Resume changes {name}")
    elif args.checkpoint:
        model, checkpoint = load_model(args.checkpoint, device)
        if args.mode == "probe":
            if args.observation != checkpoint["config"].get("observation", "rgb"):
                raise ValueError("A frozen probe must retain its fitted encoder observation")
            for name in ("channels", "base_channels"):
                if checkpoint["config"][name] != getattr(args, name):
                    raise ValueError(f"Frozen encoder architecture changes {name}")
            # Identical decoder initialization and exposure for every frozen representation.
            set_seed(args.seed)
            fresh = models.FamilyCodec(
                args.channels, args.base_channels, args.head_channels
            ).to(device)
            model.decoders = fresh.decoders
        else:
            raise ValueError("Use --resume for continued training or --mode probe for readouts")
    else:
        model = models.FamilyCodec(args.channels, args.base_channels, args.head_channels).to(device)
    if args.mode == "probe":
        for parameter in model.analysis.parameters():
            parameter.requires_grad_(False)
        for parameter in model.stream.parameters():
            parameter.requires_grad_(False)
        if checkpoint is None:
            raise ValueError("Probe mode requires a checkpoint")
        compressed = checkpoint_compressed(checkpoint)
    elif args.mode == "reference":
        for parameter in model.stream.parameters():
            parameter.requires_grad_(False)
    encoder_config = (
        checkpoint.get("encoder_config", checkpoint["config"])
        if args.mode == "probe"
        else vars(args)
    )
    for task, decoder in model.decoders.items():
        if task not in tasks:
            for parameter in decoder.parameters():
                parameter.requires_grad_(False)
    main = [
        parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and not name.endswith(".quantiles")
    ]
    auxiliary = [
        parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and name.endswith(".quantiles")
    ]
    optimizer = torch.optim.AdamW(main, lr=args.learning_rate, weight_decay=1e-4)
    aux_optimizer = torch.optim.Adam(auxiliary, lr=1e-3) if auxiliary else None
    if args.resume:
        optimizer.load_state_dict(checkpoint["optimizer"])
        if aux_optimizer:
            aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
    weights = torch.tensor(calibration["semantic_weights"], device=device)
    train_data = loader(args, "train", args.train_samples, True)
    val_data = loader(args, "val", args.val_samples)
    write_json(output / "config.json", vars(args))
    write_json(
        output / "data_keys.json",
        {
            "train": [sample.key for sample in train_data.dataset.samples],
            "val": [sample.key for sample in val_data.dataset.samples],
        },
    )
    iterator = iter(train_data)
    started = time.perf_counter()
    best_score = checkpoint.get("best_validation_score", math.inf) if args.resume else math.inf
    if math.isfinite(best_score) and not (output / "best.pt").exists():
        original_best = Path(args.resume).with_name("best.pt")
        if not original_best.is_file():
            raise ValueError("Resuming historical validation selection requires its best.pt")
        copy2(original_best, output / "best.pt")
        original_result = original_best.with_name("best_validation.json")
        if original_result.is_file():
            copy2(original_result, output / "best_validation.json")
    interval_started = started
    totals: defaultdict[str, float] = defaultdict(float)
    interval_steps = 0
    for step in range(start_step + 1, args.steps + 1):
        model.train()
        if args.mode == "probe":
            model.analysis.eval()
            model.stream.eval()
        try:
            raw = next(iterator)
        except StopIteration:
            iterator = iter(train_data)
            raw = next(iterator)
        batch = prepared(raw, device)
        inputs = observation_input(batch, calibration, args.observation)
        optimizer.zero_grad(set_to_none=True)
        if aux_optimizer:
            aux_optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=args.amp):
            latent, rate = model.encode(inputs, compressed)
            predictions = model.predict(latent, tasks)
        values = losses(predictions, batch, weights)
        objective = sum(LOSS_SCALES[task] * loss for task, loss in values.items())
        if args.mode == "codec":
            objective = objective + args.lambda_rate * rate
        if not torch.isfinite(objective):
            raise FloatingPointError(f"Nonfinite objective at update {step}")
        objective.backward()
        nn.utils.clip_grad_norm_(main, 1.0)
        optimizer.step()
        if aux_optimizer:
            model.stream.aux_loss().backward()
            aux_optimizer.step()
        totals["objective"] += objective.item()
        totals["estimated_bpp"] += rate.item()
        for task, value in values.items():
            totals[f"loss_{task}"] += value.item()
        interval_steps += 1
        if step % args.log_every == 0 or step == args.steps:
            elapsed = time.perf_counter() - interval_started
            log_record(
                output / "train.jsonl",
                {
                    "step": step,
                    "image_exposures": step * args.batch_size,
                    "elapsed_s": time.perf_counter() - started,
                    "interval_s": elapsed,
                    "seconds_per_update": elapsed / interval_steps,
                    **{key: value / interval_steps for key, value in totals.items()},
                },
            )
            totals.clear()
            interval_steps = 0
            interval_started = time.perf_counter()
        if step % args.eval_every == 0 or step == args.steps:
            result = evaluate(
                model,
                val_data,
                device,
                tasks,
                compressed,
                weights,
                calibration,
                observation=args.observation,
            )
            score = sum(LOSS_SCALES[task] * value for task, value in result["losses"].items())
            if args.mode == "codec":
                score += args.lambda_rate * result["estimated_bpp"]
            improved = score < best_score
            best_score = min(best_score, score)
            result.update(step=step, elapsed_s=time.perf_counter() - started)
            result["selection_score"] = score
            log_record(output / "validation.jsonl", result)
            state = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "aux_optimizer": aux_optimizer.state_dict() if aux_optimizer else None,
                "config": vars(args),
                "step": step,
                "calibration": calibration,
                "representation_compressed": compressed,
                "encoder_config": encoder_config,
                "best_validation_score": best_score,
            }
            torch.save(state, output / "last.pt")
            if improved:
                torch.save(state, output / "best.pt")
                write_json(output / "best_validation.json", result)
            interval_started = time.perf_counter()
    if args.mode == "codec":
        model, selected = load_model(output / "best.pt", device)
        coded_data = loader(args, "val", args.coded_samples)
        result = evaluate(
            model,
            coded_data,
            device,
            tasks,
            True,
            weights,
            calibration,
            True,
            output / "coded_rates.jsonl",
            args.observation,
        )
        result.update(checkpoint=str(output / "best.pt"), checkpoint_step=selected["step"])
        write_json(output / "coded_validation.json", result)
    write_json(
        output / "finished.json",
        {
            "step": args.steps,
            "elapsed_s": time.perf_counter() - started,
            "image_exposures": (args.steps - start_step) * args.batch_size,
            "pid": os.getpid(),
        },
    )


def run_evaluation(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    """Score a checkpoint on one split and record the encoder and readout provenance."""
    output = Path(args.output)
    model, checkpoint = load_model(args.checkpoint, device)
    if not set(FAMILIES[args.family]).issubset(FAMILIES[checkpoint["config"]["family"]]):
        raise ValueError(
            "Requested readout was not trained in this checkpoint. Fit a matched frozen probe first"
        )
    weights = torch.tensor(checkpoint["calibration"]["semantic_weights"], device=device)
    compressed = checkpoint_compressed(checkpoint)
    if args.actual and not compressed:
        raise ValueError("An uncompressed reference has no entropy-coded message")
    data = loader(args, args.split, args.val_samples)
    result = evaluate(
        model,
        data,
        device,
        FAMILIES[args.family],
        compressed,
        weights,
        checkpoint["calibration"],
        args.actual,
        output / ("coded_rates.jsonl" if args.actual else "image_metrics.jsonl"),
        checkpoint["config"].get("observation", "rgb"),
    )
    result["split"] = args.split
    result.update(checkpoint=args.checkpoint, checkpoint_step=checkpoint["step"])
    encoder_config = checkpoint.get("encoder_config", checkpoint["config"])
    result.update(
        preservation_family=encoder_config["family"],
        encoder_training_mode=encoder_config["mode"],
        encoder_rate_weight=encoder_config["lambda_rate"],
        encoder_seed=encoder_config["seed"],
        readout_family=checkpoint["config"]["family"],
        readout_head_channels=checkpoint["config"]["head_channels"],
        readout_seed=checkpoint["config"]["seed"],
    )
    write_json(output / "evaluation.json", result)
    return result


def dispatch(args: argparse.Namespace) -> None:
    """Resolve the run directory defaults, then execute the requested mode."""
    if args.mode == "codec" and args.observation != "rgb":
        raise ValueError("Codec family comparisons use the shared RGB observation")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if args.manifest is None:
        args.manifest = str(output / "taskonomy_manifest.json")
    if args.calibration is None:
        args.calibration = str(output / "calibration.json")
    torch.set_num_threads(4)
    set_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    device = resolve_device(args.device)
    calibration_path = Path(args.calibration)
    calibration = (
        json.loads(calibration_path.read_text())
        if calibration_path.exists()
        else calibrate(args, device)
    )
    if args.mode == "calibrate":
        reported = {key: value for key, value in calibration.items() if key != "keys"}
        print(json.dumps(reported), flush=True)
    elif args.mode == "evaluate":
        run_evaluation(args, device)
    else:
        train(args, device, calibration)

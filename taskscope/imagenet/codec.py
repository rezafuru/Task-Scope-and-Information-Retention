"""Shared feature initializer, matched task fits, and sealed message coding.

One initializer minimizes normalized feature squared error plus estimated rate. Every
reported task fit starts from that single checkpoint and then optimizes either coarse KL
alone or coarse KL plus fine cross entropy, with the architecture, the observation, the
rate term, the image order, and the fitting exposure held fixed. Neither task objective
includes feature error.

``code_split`` seals a split before any adapted fitting: it entropy codes every image to
an independent packet, writes those packets to one stream, decodes each packet back, and
records the byte components, the inherited predictions, and the digests of what it wrote.
Rates reported from a seal are complete message lengths, including the packet header and
the four framing bytes each stored packet carries.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import struct
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from . import models
from .data import ImageNetImages, sha256, write_json
from .models import FeatureCodec, FrozenTeacher, tensor_state_sha256

# Deterministic cuBLAS reductions. Read when a kernel first runs, not at import.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

OBJECTIVES = ("feature_initialization", "coarse", "added_fine")
TASK_OBJECTIVES = ("coarse", "added_fine")
POOLED_CACHE_GRID = 4
GRADIENT_CLIP = 5.0
AUXILIARY_LR = 1e-3


def provenance() -> dict:
    """Bind a checkpoint or a seal to the implementation that produced it."""
    here = Path(__file__).parent
    return {"codec_sha256": sha256(here / "codec.py"),
            "models_sha256": sha256(here / "models.py"),
            "data_sha256": sha256(here / "data.py"),
            "codec_configuration": models.CODEC_CONFIGURATION}


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def start_stage(device: torch.device) -> float:
    sync(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    return time.perf_counter()


def finish_stage(started: float, device: torch.device) -> dict:
    sync(device)
    return {"seconds": time.perf_counter() - started,
            "peak_allocated_bytes": (torch.cuda.max_memory_allocated(device)
                                     if device.type == "cuda" else None)}


def record(path: Path, row: dict) -> None:
    line = json.dumps(row, allow_nan=False)
    with path.open("a") as stream:
        stream.write(line + "\n")
    print(line, flush=True)


def grouped_log_probabilities(logits: torch.Tensor, groups: torch.Tensor) -> torch.Tensor:
    """Normalize over the selected fine classes and sum within each superclass."""
    log_probability = logits.float().log_softmax(1)
    return torch.stack([log_probability[:, groups == index].logsumexp(1)
                        for index in range(int(groups.max()) + 1)], dim=1)


def task_losses(decoded_logits: torch.Tensor, reference_logits: torch.Tensor,
                fine_labels: torch.Tensor,
                groups: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    reference = grouped_log_probabilities(reference_logits.detach(), groups)
    decoded = grouped_log_probabilities(decoded_logits, groups)
    coarse = F.kl_div(decoded, reference, log_target=True, reduction="batchmean")
    coarse = coarse / math.log(reference.shape[1])
    fine = F.cross_entropy(decoded_logits.float(), fine_labels) / math.log(decoded_logits.shape[1])
    return coarse, fine


def objective_loss(coarse: torch.Tensor, fine: torch.Tensor, rate: torch.Tensor,
                   objective: str, beta: float, fine_weight: float) -> torch.Tensor:
    if objective not in TASK_OBJECTIVES:
        raise ValueError("Unknown preservation objective")
    return coarse + beta * rate + (fine_weight * fine if objective == "added_fine" else 0)


def loader(data: ImageNetImages, args: argparse.Namespace, shuffle: bool = False) -> DataLoader:
    generator = torch.Generator().manual_seed(args.seed)
    return DataLoader(data, batch_size=args.batch_size, shuffle=shuffle,
                      num_workers=args.workers, pin_memory=args.device.startswith("cuda"),
                      generator=generator, persistent_workers=args.workers > 0)


@torch.no_grad()
def calibrate(teacher: FrozenTeacher, data: ImageNetImages, args: argparse.Namespace,
              device: torch.device) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Channel statistics of the observation, taken from training images only."""
    channels, grid = models.FEATURE_SHAPE[0], models.FEATURE_SHAPE[1] * models.FEATURE_SHAPE[2]
    total = torch.zeros(channels, dtype=torch.float64, device=device)
    square, count, images = total.clone(), 0, 0
    started = start_stage(device)
    for batch_index, (image, _, _, _) in enumerate(loader(data, args, shuffle=True)):
        value = teacher.observe(image.to(device)).double()
        total += value.sum((0, 2, 3))
        square += value.square().sum((0, 2, 3))
        count += len(value) * grid
        images += len(value)
        if batch_index + 1 >= args.calibration_batches:
            break
    mean = total / count
    variance = (square / count - mean.square()).clamp_min(0)
    scale = variance.sqrt().clamp_min(0.05 * variance.mean().sqrt())
    if not torch.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError("Invalid training-only feature normalization")
    return mean.float()[None, :, None, None], scale.float()[None, :, None, None], {
        "training_images": images, "channel_values": count,
        **finish_stage(started, device),
        "rule": "Training channel mean and population std, floor 0.05 times RMS channel std"}


def fit(args: argparse.Namespace, teacher: FrozenTeacher, training: ImageNetImages,
        device: torch.device) -> None:
    """Fit the shared initializer, or one task objective starting from it."""
    seed_all(args.seed)
    model = FeatureCodec().to(device)
    initialization = None
    if args.initialization:
        prior = torch.load(args.initialization, map_location="cpu", weights_only=True)
        if (prior["provenance"] != provenance() or prior["teacher"] != teacher.identity
                or prior["mapping_sha256"] != sha256(args.mapping)
                or prior["args"]["objective"] != "feature_initialization"):
            raise ValueError("Require the common matching feature-reconstruction initializer")
        model.load_state_dict(prior["model"])
        normalization = prior["normalization"]
        initialization = {"path": str(args.initialization), "sha256": sha256(args.initialization),
                          "image_exposures": prior["image_exposures"],
                          "objective": prior["args"]["objective"], "training": prior["training"]}
    else:
        mean, scale, normalization = calibrate(teacher, training, args, device)
        model.mean.copy_(mean)
        model.scale.copy_(scale)
    initial_hash = tensor_state_sha256(model.state_dict())
    parameters = [value for name, value in model.named_parameters()
                  if not name.endswith(".quantiles")]
    quantiles = [value for name, value in model.named_parameters() if name.endswith(".quantiles")]
    optimizer = torch.optim.Adam(parameters, lr=args.lr)
    auxiliary = torch.optim.Adam(quantiles, lr=AUXILIARY_LR)
    stream = loader(training, args, shuffle=True)
    started, steps, exposures = start_stage(device), 0, 0
    for epoch in range(args.epochs):
        model.train()
        sums, epoch_images = np.zeros(4), 0
        for image, fine, _, _ in stream:
            image, fine = image.to(device), fine.to(device)
            with torch.no_grad():
                feature = teacher.observe(image)
                reference = (teacher.predict(feature)
                             if args.objective != "feature_initialization" else None)
            decoded, rate = model(feature)
            feature_mse = ((decoded - feature) / model.scale).square().mean()
            if args.objective == "feature_initialization":
                coarse_loss, fine_loss = feature_mse.detach() * 0, feature_mse.detach() * 0
                loss = feature_mse + args.beta * rate
            else:
                coarse_loss, fine_loss = task_losses(teacher.predict(decoded), reference, fine,
                                                     teacher.groups)
                loss = objective_loss(coarse_loss, fine_loss, rate, args.objective, args.beta,
                                      args.fine_weight)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite initialization or matched task objective")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, GRADIENT_CLIP)
            optimizer.step()
            auxiliary.zero_grad(set_to_none=True)
            model.stream.aux_loss().backward()
            auxiliary.step()
            size = len(image)
            sums += np.asarray([feature_mse.item(), coarse_loss.item(), fine_loss.item(),
                                rate.item()]) * size
            epoch_images += size
            exposures += size
            steps += 1
            if steps == 100 or steps % 1000 == 0:
                sync(device)
                record(args.output / "progress.jsonl",
                       {"steps": steps, "image_exposures": exposures,
                        "epoch_normalized_feature_mse": sums[0] / epoch_images,
                        "seconds": time.perf_counter() - started})
            if steps >= args.max_steps:
                break
        sync(device)
        record(args.output / "training.jsonl",
               {"epoch": epoch + 1, "steps": steps, "image_exposures": exposures,
                "normalized_feature_mse": sums[0] / epoch_images,
                "coarse_kl_normalized": sums[1] / epoch_images,
                "fine_ce_normalized": sums[2] / epoch_images,
                "estimated_bpp": sums[3] / epoch_images,
                "seconds": time.perf_counter() - started})
        if steps >= args.max_steps:
            break
    metadata = {"args": {key: str(value) if isinstance(value, Path) else value
                         for key, value in vars(args).items()},
                "normalization": normalization, "initial_state_sha256": initial_hash,
                "initialization": initialization, "training": training.identity,
                "mapping_sha256": sha256(args.mapping), "provenance": provenance(),
                "teacher": teacher.identity, "steps": steps, "image_exposures": exposures,
                "selection": "fixed final optimizer step", "test_used": False}
    torch.save({**metadata, "model": model.state_dict()}, args.output / "last.pt")
    write_json(args.output / "fit.json", metadata)
    write_json(args.output / "fit_timing.json",
               {**finish_stage(started, device), "image_exposures": exposures,
                "steps": steps, "normalization": normalization})


def load_codec(path: Path, mapping: Path, device: torch.device,
               teacher_identity: dict) -> FeatureCodec:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["mapping_sha256"] != sha256(mapping):
        raise ValueError("Checkpoint mapping mismatch")
    if checkpoint.get("provenance") != provenance() or checkpoint.get("teacher") != teacher_identity:
        raise ValueError("Checkpoint implementation, codec configuration or teacher mismatch")
    model = FeatureCodec().to(device).eval()
    model.load_state_dict(checkpoint["model"])
    model.stream.update(force=True, update_quantiles=True)
    return model


def pooled(feature: torch.Tensor) -> torch.Tensor:
    return F.adaptive_avg_pool2d(feature.float(), POOLED_CACHE_GRID).flatten(1)


@torch.no_grad()
def code_split(model: FeatureCodec | None, teacher: FrozenTeacher, data: ImageNetImages,
               args: argparse.Namespace, device: torch.device, output: Path) -> dict:
    """Seal actual messages and pooled decoded tensors before any adapted fitting."""
    output.mkdir(parents=True, exist_ok=False)
    started = start_stage(device)
    if model is not None:
        model.eval()
        model.stream.update(force=True, update_quantiles=True)
    width = models.FEATURE_SHAPE[0] * POOLED_CACHE_GRID ** 2
    # The pooled cache is part of the sealed record and is hashed into sealed.json.
    cache = np.lib.format.open_memmap(output / "pooled.npy", mode="w+", dtype=np.float16,
                                      shape=(len(data), width))
    names, fine_targets, coarse_targets = [], [], []
    predictions, references, components = [], [], []
    offset, feature_mse, coarse_kl, fine_ce = 0, 0.0, 0.0, 0.0
    with (output / "messages.bin").open("xb") as messages:
        for images, fines, coarses, ids in loader(data, args):
            feature = teacher.observe(images.to(device))
            reference_logits = teacher.predict(feature)
            # Each entropy packet is independently decodable. Four outer bytes frame its length.
            decoded, counts = [], []
            for row in range(len(feature)):
                if model is None:
                    reconstruction, count = feature[row:row + 1], [0, 0, 0, 0]
                else:
                    reconstruction, packet = model.code(feature[row:row + 1])
                    messages.write(struct.pack("!I", len(packet.packet)))
                    messages.write(packet.packet)
                    count = [packet.main_bytes, packet.hyper_bytes, packet.header_bytes,
                             models.OUTER_FRAMING_BYTES]
                decoded.append(reconstruction)
                counts.append(count)
            decoded = torch.cat(decoded)
            logits = teacher.predict(decoded)
            coarse, fine = task_losses(logits, reference_logits, fines.to(device), teacher.groups)
            coarse_kl += coarse.item() * len(feature)
            fine_ce += fine.item() * len(feature)
            feature_mse += F.mse_loss(decoded, feature).item() * len(feature)
            cache[offset:offset + len(feature)] = pooled(decoded).cpu().numpy().astype(np.float16)
            offset += len(feature)
            predictions.extend(zip(
                grouped_log_probabilities(logits, teacher.groups).argmax(1).cpu().tolist(),
                logits.argmax(1).cpu().tolist()))
            references.extend(zip(
                grouped_log_probabilities(reference_logits, teacher.groups).argmax(1).cpu().tolist(),
                reference_logits.argmax(1).cpu().tolist()))
            names.extend(ids)
            fine_targets.extend(fines.tolist())
            coarse_targets.extend(coarses.tolist())
            components.extend(counts)
    cache.flush()
    del cache
    sync(device)
    predictions = np.asarray(predictions)
    references = np.asarray(references)
    components = np.asarray(components)
    np.savez_compressed(output / "predictions.npz", ids=np.asarray(names),
                        fine_target=fine_targets, coarse_target=coarse_targets,
                        prediction=predictions, reference=references, bytes=components)
    report = {"images": len(data), "dataset": data.identity, "raw_reference": model is None,
              "main_bytes": int(components[:, 0].sum()), "hyper_bytes": int(components[:, 1].sum()),
              "header_bytes": int(components[:, 2].sum()),
              "outer_framing_bytes": int(components[:, 3].sum()),
              "total_bytes": int(components.sum()),
              "actual_bpp": float(components.sum() * 8 / len(data) / 224**2),
              "coarse_accuracy": float((predictions[:, 0] == coarse_targets).mean()),
              "fine_accuracy": float((predictions[:, 1] == fine_targets).mean()),
              "coarse_teacher_agreement": float((predictions[:, 0] == references[:, 0]).mean()),
              "fine_teacher_agreement": float((predictions[:, 1] == references[:, 1]).mean()),
              "reference_coarse_accuracy": float((references[:, 0] == coarse_targets).mean()),
              "reference_fine_accuracy": float((references[:, 1] == fine_targets).mean()),
              "coarse_kl_normalized": coarse_kl / len(data), "fine_ce_normalized": fine_ce / len(data),
              "feature_mse": feature_mse / len(data), **finish_stage(started, device),
              "test_used": data.split == "test", "mapping_sha256": sha256(args.mapping),
              "provenance": provenance(), "teacher": teacher.identity,
              "checkpoint_sha256": None if model is None else sha256(args.checkpoint),
              "files_sha256": {name: sha256(output / name) for name in
                               ("messages.bin", "pooled.npy", "predictions.npz")}}
    if (output / "messages.bin").stat().st_size != report["total_bytes"]:
        raise RuntimeError("Complete serialized bytes disagree with rate components")
    write_json(output / "sealed.json", report)
    print(json.dumps({"stage": "code", "split": data.split, **report}), flush=True)
    return report


def run_fit(args: argparse.Namespace) -> None:
    """Set up the frozen teacher and training reader, then run one fitting stage."""
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    seed_all(args.seed)
    args.output.mkdir(parents=True, exist_ok=False)
    mapping = json.loads(args.mapping.read_text())
    teacher = FrozenTeacher(mapping).to(device).eval()
    training = ImageNetImages(args.data_root, mapping, "train")
    write_json(args.output / "runtime.json",
               {"provenance": provenance(), "teacher": teacher.identity,
                "torch": str(torch.__version__), "device": str(device),
                "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                "test_used": False})
    fit(args, teacher, training, device)


__all__ = ["calibrate", "code_split", "fit", "finish_stage",
           "grouped_log_probabilities", "load_codec", "loader", "objective_loss", "pooled",
           "provenance", "record", "run_fit", "seed_all", "start_stage", "sync", "task_losses"]

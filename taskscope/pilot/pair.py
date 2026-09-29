"""Species-pair assessment: 200-way CUB readers restricted to Indigo Bunting and Blue Grosbeak.

An observation is one depth of the frozen ImageNet ResNet-50 read in one of two ways: the
complete native feature map at that depth, or the RGB reconstruction the inverse of that depth
produces. Eight observations in all, the four exits and their four reconstructions.

Stages, in order:

  readouts  fit the two 200-way readers of one observation on the CUB development population
  heldout   restrict every fitted reader to the pair on the official CUB test photographs

Every reader has a fresh 200-class output and is fitted with class-weighted cross entropy, 50
AdamW epochs at learning rate and weight decay 1e-4 and batch 64, with the epoch chosen by
validation macro accuracy then validation macro cross entropy. The fitting population is 4,794
photographs across 200 species with six validation photographs per species, so one reader runs
3,750 updates. Native readers take the pretrained fourth stage at ``layer3`` and the pretrained
final residual block at ``block1`` to ``block3``, in both cases with global average pooling.
Reconstruction readers adapt the complete pretrained network to the reconstructions of one
inverse, with training-only horizontal reflections.

The reported statistic is the accuracy of each fitted reader on the 30 official test photographs
per species after its 200-class output is restricted to the two species, together with the
arithmetic mean of the two. The readers are never combined into a probability ensemble and the
accuracies carry no interval.
"""

from __future__ import annotations

import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Subset, TensorDataset

from taskscope.pilot import features as feat
from taskscope.pilot.common import device_name, file_hash, log, read_json, write_json
from taskscope.pilot.data import load_inventory

SPECIES = (13, 53)
NAMES = ("Indigo Bunting", "Blue Grosbeak")
CLASSES = 200
PAIR_CLASSES = len(SPECIES)
FITTING_IMAGES = 4794
VALIDATION_PER_SPECIES = 6
TEST_PER_SPECIES = 30
SEEDS = (17, 23)
EPOCHS, BATCH, LEARNING_RATE, WEIGHT_DECAY = 50, 64, 1e-4, 1e-4
INVERSE_PREFIX = "inverse_"
OBSERVATIONS = feat.EXITS + tuple(INVERSE_PREFIX + name for name in feat.EXITS)
BLOCK_READER = ("Pretrained final fourth-stage residual block, global average pooling, fresh "
                "200-way output, identical at block1 to block3")
READER_DESCRIPTIONS = {
    "rgb": "Complete pretrained ImageNet ResNet-50 with a fresh 200-way output, adapted to the "
           "reconstructions of one inverse with training-only horizontal flips p=0.5",
    "layer3": "Pretrained fourth stage (all three residual blocks), global average pooling, fresh "
              "200-way output",
} | {f"block{index}": BLOCK_READER for index in (1, 2, 3)}


def exit_of(observation: str) -> str:
    """The residual-block exit an observation is read at."""
    return observation[len(INVERSE_PREFIX):] if observation.startswith(INVERSE_PREFIX) else observation


def reader_of(observation: str) -> str:
    """The reader family fitted for an observation. Reconstructions are read as RGB."""
    return "rgb" if observation.startswith(INVERSE_PREFIX) else observation


def development_records(inventory: Path) -> tuple[list[dict], dict]:
    """The complete development population: 4,794 fitting photographs, six validation per species."""
    rows = load_inventory(inventory, "development")
    rows.sort(key=lambda row: row["observation_id"])
    if sum(row["split"] == "train" for row in rows) != FITTING_IMAGES:
        raise ValueError(f"Require the {FITTING_IMAGES} fitting photographs of the development split")
    validation = Counter(row["fine_label"] for row in rows if row["split"] == "val")
    if validation != {label: VALIDATION_PER_SPECIES for label in range(CLASSES)}:
        raise ValueError("Require six validation photographs per species")
    for row in rows:
        row["label"] = row["fine_label"]
    return rows, {"path": str(inventory), "sha256": file_hash(inventory)}


def cub_test_records(inventory: Path) -> tuple[list[dict], dict]:
    """The official CUB test photographs of the pair, 30 per species, labelled by pair position."""
    rows = [row for row in load_inventory(inventory, "test") if row["fine_label"] in SPECIES]
    rows.sort(key=lambda row: row["observation_id"])
    if Counter(row["fine_label"] for row in rows) != {label: TEST_PER_SPECIES for label in SPECIES}:
        raise ValueError("Pair assessment requires 30 official test photographs per species")
    for row in rows:
        row["label"] = SPECIES.index(row["fine_label"])
    return rows, {"path": str(inventory), "sha256": file_hash(inventory)}


def load_rgb(paths_and_hashes) -> torch.Tensor:
    values = []
    for path, digest in paths_and_hashes:
        if file_hash(path) != digest:
            raise ValueError(f"RGB source changed: {path}")
        with Image.open(path) as image:
            if image.mode != "RGB" or image.size != (224, 224):
                raise ValueError("Unexpected RGB geometry")
            values.append(np.asarray(image).copy().transpose(2, 0, 1))
    return torch.from_numpy(np.stack(values))


def reconstruction_rows(inversions: Path, exit_name: str,
                        records: list[dict]) -> tuple[list[dict], dict]:
    """Reconstruction records of one exit in the order of ``records``, with their sources checked."""
    path = Path(inversions) / exit_name / "inventory.json"
    inventory = read_json(path)
    if inventory["inverse"]["view"] != exit_name:
        raise ValueError(f"Reconstruction inventory targets a different exit: {path}")
    indexed = {row["observation_id"]: row for row in inventory["records"]}
    if len(indexed) != len(inventory["records"]):
        raise ValueError("Duplicate reconstruction observations")
    rows = []
    for original in records:
        row = indexed[original["observation_id"]]
        if (any(row[key] != original[key] for key in ("group_id", "split", "fine_label"))
                or row["original_input_sha256"] != original["input_sha256"]):
            raise ValueError("Reconstruction lost its original source identity")
        rows.append(row)
    return rows, {"inventory": str(path), "inventory_sha256": file_hash(path),
                  "inverse": inventory["inverse"]}


def load_observation(observation: str, records: list[dict], cache: Path,
                     inversions: Path) -> tuple[torch.Tensor, dict]:
    """Complete observations at one depth: the native feature map or its RGB reconstruction.

    ``cache`` holds the extracted feature arrays of the population, ``inversions`` one
    subdirectory per exit with the inventory and the exported reconstructions.
    """
    exit_name = exit_of(observation)
    if observation.startswith(INVERSE_PREFIX):
        rows, provenance = reconstruction_rows(inversions, exit_name, records)
        root = Path(inversions) / exit_name
        images = load_rgb([(root / row["input_path"], row["input_sha256"]) for row in rows])
        return images, provenance
    values, provenance = feat.checked_feature_subset(cache, records, exit_name)
    return torch.from_numpy(values), provenance


def prepare(reader: str, batch: torch.Tensor, device: torch.device,
            flip: torch.Generator | None = None) -> torch.Tensor:
    if reader == "rgb":
        if flip is not None:
            batch = feat.training_images(batch, flip)
        return feat.normalized(batch, device)
    return batch.to(device, non_blocking=True)


def macro_scores(prob: np.ndarray, target: np.ndarray, k: int) -> tuple[float, float]:
    """Macro accuracy and macro cross entropy, in the order the checkpoint selection reads them."""
    pred = prob.argmax(1)
    entropy = -np.log(np.maximum(prob[np.arange(len(target)), target], 1e-12))
    return (float(np.mean([(pred[target == c] == c).mean() for c in range(k)])),
            float(np.mean([entropy[target == c].mean() for c in range(k)])))


def metrics(prob: np.ndarray, target: np.ndarray, k: int) -> dict:
    pred = prob.argmax(1)
    accuracy, entropy = macro_scores(prob, target, k)
    return {"macro_accuracy": accuracy, "macro_cross_entropy": entropy,
            "per_class_accuracy": [float((pred[target == c] == c).mean()) for c in range(k)],
            "counts": np.bincount(target, minlength=k).tolist(),
            "confusion": np.bincount(target * k + pred, minlength=k * k).reshape(k, k).tolist()}


@torch.inference_mode()
def apply(model: nn.Module, reader: str, values: torch.Tensor, device: torch.device,
          index: np.ndarray | None = None) -> np.ndarray:
    """Apply a fitted reader to ``values``, or to the rows ``index`` selects."""
    model.eval()
    rows = np.arange(len(values)) if index is None else index
    out = []
    for start in range(0, len(rows), BATCH):
        batch = values[rows[start:start + BATCH]]
        out.append(model(prepare(reader, batch, device)).softmax(1).float().cpu().numpy())
    prob = np.concatenate(out)
    if not np.isfinite(prob).all():
        raise ValueError("Nonfinite reader probabilities")
    return prob


def restrict(prob: np.ndarray) -> np.ndarray:
    """The fitted 200-class output restricted to the two species and renormalised."""
    values = prob[:, list(SPECIES)]
    return values / values.sum(1, keepdims=True)


def fit(reader: str, values: torch.Tensor, target: np.ndarray, train_index: np.ndarray,
        val_index: np.ndarray, seed: int, device: torch.device) -> tuple[nn.Module, dict]:
    """One reader. The checkpoint is the earliest epoch with the best validation key."""
    model = feat.new_reader(reader, CLASSES, seed, pretrained=True).to(device)
    generator = torch.Generator().manual_seed(seed)
    flip = torch.Generator().manual_seed(seed) if reader == "rgb" else None
    dataset = TensorDataset(values, torch.from_numpy(target).long())
    loader = DataLoader(Subset(dataset, train_index.tolist()), batch_size=BATCH, shuffle=True,
                        generator=generator, pin_memory=device.type == "cuda")
    counts = np.bincount(target[train_index], minlength=CLASSES)
    if (counts == 0).any():
        raise ValueError("Every species must occur in the fitting split")
    weights = torch.as_tensor(len(train_index) / (CLASSES * counts), dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    best, best_epoch, state, history = None, None, None, []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total, seen = 0.0, 0
        for batch, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(prepare(reader, batch, device, flip)),
                                               labels.to(device), weight=weights)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite fitting loss")
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(labels)
            seen += len(labels)
        accuracy, entropy = macro_scores(apply(model, reader, values, device, val_index),
                                         target[val_index], CLASSES)
        history.append({"epoch": epoch, "training_loss": total / seen,
                        "val_macro_accuracy": accuracy, "val_macro_cross_entropy": entropy})
        if best is None or (accuracy, -entropy) > best:
            best, best_epoch = (accuracy, -entropy), epoch
            state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    model.load_state_dict(state)
    return model, {"seed": seed, "selected_epoch": best_epoch, "val_macro_accuracy": best[0],
                   "val_macro_cross_entropy": -best[1], "history": history}


def readouts(output: Path, *, observation: str, inventory: Path, cache: Path, inversions: Path,
             device: torch.device, seeds=SEEDS) -> dict:
    """Fit the readers of one observation and save a checkpoint per fit."""
    if observation not in OBSERVATIONS:
        raise ValueError(f"Unknown observation: {observation}")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    reader = reader_of(observation)
    records, manifest = development_records(inventory)
    target = np.array([row["label"] for row in records])
    train_index = np.flatnonzero([row["split"] == "train" for row in records])
    val_index = np.flatnonzero([row["split"] == "val" for row in records])
    values, provenance = load_observation(observation, records, cache, inversions)
    result = {
        "status": "running", "observation": observation, "reader": reader, "classes": CLASSES,
        "manifest": manifest, "provenance": provenance, "observation_shape": list(values.shape),
        "training": {"epochs": EPOCHS, "batch_size": BATCH, "optimizer": "AdamW",
                     "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY,
                     "updates_per_reader": EPOCHS * ((len(train_index) + BATCH - 1) // BATCH),
                     "loss": "Cross entropy with class weights n_train/(classes*count)",
                     "seeds": list(seeds), "weights": feat.WEIGHTS_NAME,
                     "reader": READER_DESCRIPTIONS[reader],
                     "augmentation": "Training-only horizontal flips p=0.5" if reader == "rgb"
                                     else "None",
                     "validation": "The 1200 development validation photographs, six per species",
                     "selection": "Epoch selected by validation macro accuracy then macro cross "
                                  "entropy, earliest epoch"},
        "runtime": feat.runtime_provenance(), "pretrained_checkpoint": feat.checkpoint_provenance(),
        "device": device_name(device), "candidates": []}
    write_json(output / "models.json", result, indent=1)
    for position, seed in enumerate(seeds):
        start = time.monotonic()
        model, info = fit(reader, values, target, train_index, val_index, seed, device)
        path = output / f"reader_seed{seed}.pt"
        torch.save({"state_dict": model.state_dict(), "observation": observation, "reader": reader,
                    "seed": seed, "classes": CLASSES}, path)
        result["candidates"].append({
            "seed_index": position, "seed": seed, "selected_epoch": info["selected_epoch"],
            "val_macro_accuracy": info["val_macro_accuracy"],
            "val_macro_cross_entropy": info["val_macro_cross_entropy"],
            "checkpoint": {"path": str(path.relative_to(output)), "sha256": file_hash(path)},
            "history": info["history"], "seconds": time.monotonic() - start})
        log(stage="readouts", observation=observation, seed=seed,
            selected_epoch=info["selected_epoch"],
            val_macro_accuracy=round(info["val_macro_accuracy"], 4),
            seconds=round(time.monotonic() - start, 1))
        del model
        write_json(output / "models.json", result, indent=1)
        if device.type == "cuda":
            torch.cuda.empty_cache()
    result["status"] = "complete"
    result["seconds"] = time.monotonic() - started
    write_json(output / "models.json", result, indent=1)
    log(stage="complete", observation=observation, seconds=round(result["seconds"]))
    return result


def load_reader(models: dict, candidate: dict, directory: Path, device: torch.device) -> nn.Module:
    """Rebuild one fitted reader from the checkpoint its readouts record names."""
    path = Path(directory) / candidate["checkpoint"]["path"]
    if file_hash(path) != candidate["checkpoint"]["sha256"]:
        raise ValueError(f"Fitted reader changed: {path}")
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if (saved["observation"] != models["observation"] or saved["reader"] != models["reader"]
            or saved["seed"] != candidate["seed"] or saved["classes"] != CLASSES):
        raise ValueError("Fitted reader identity differs from its record")
    model = feat.new_reader(models["reader"], CLASSES, candidate["seed"], pretrained=False)
    model.load_state_dict(saved["state_dict"])
    return model.to(device).eval()


def heldout(output: Path, *, readouts_root: Path, test_inventory: Path, test_cache: Path,
            inversions: Path, device: torch.device) -> dict:
    """Restrict every fitted reader to the pair on the official CUB test photographs."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    records, manifest = cub_test_records(test_inventory)
    target = np.array([row["label"] for row in records])
    fine = np.array([row["fine_label"] for row in records])
    membership_keys = ("observation_id", "group_id", "cub_image_id", "fine_label", "input_sha256")
    result = {
        "status": "running", "species": dict(zip(NAMES, SPECIES)),
        "protocol": "Two readers per observation, each fitted 200-way on the development "
                    "population and restricted to the two species. The reported accuracies are "
                    "the individual fitted readers and their arithmetic mean. No probability "
                    "ensemble and no interval.",
        "runtime": feat.runtime_provenance(), "device": device_name(device),
        "cub_test": {"n": len(records), "photographs_per_species": TEST_PER_SPECIES,
                     "manifest": manifest,
                     "membership": [{key: row[key] for key in membership_keys} for row in records],
                     "restricted_200way": {}}}
    probabilities = {}
    for observation in OBSERVATIONS:
        directory = Path(readouts_root) / observation
        models = read_json(directory / "models.json")
        if models["status"] != "complete" or models["observation"] != observation:
            raise ValueError(f"Incomplete or mismatched readouts for {observation}")
        if len(models["candidates"]) != len(SEEDS):
            raise ValueError("Require the two fitted readers of every observation")
        values, provenance = load_observation(observation, records, test_cache, inversions)
        candidates = []
        for candidate in models["candidates"]:
            model = load_reader(models, candidate, directory, device)
            prob = apply(model, models["reader"], values, device)
            pair = restrict(prob)
            probabilities[f"{observation}_seed{candidate['seed']}"] = pair
            candidates.append({"seed_index": candidate["seed_index"], "seed": candidate["seed"],
                               "restricted": metrics(pair, target, PAIR_CLASSES),
                               "top1_200way_correct_fraction": float((prob.argmax(1) == fine).mean())})
            del model
        entry = {"readouts": {"path": str(directory / "models.json"),
                              "sha256": file_hash(directory / "models.json")},
                 "provenance": provenance, "candidates": candidates,
                 "mean_macro_accuracy": float(np.mean([row["restricted"]["macro_accuracy"]
                                                       for row in candidates]))}
        result["cub_test"]["restricted_200way"][observation] = entry
        log(stage="heldout", observation=observation,
            per_reader=[round(100 * row["restricted"]["macro_accuracy"], 1) for row in candidates],
            mean=round(100 * entry["mean_macro_accuracy"], 1))
        write_json(output / "evaluation.json", result, indent=1)
        del values
        if device.type == "cuda":
            torch.cuda.empty_cache()
    np.savez_compressed(output / "predictions.npz", target=target,
                        observation_id=np.array([row["observation_id"] for row in records]),
                        **probabilities)
    result["status"] = "complete"
    result["seconds"] = time.monotonic() - started
    write_json(output / "evaluation.json", result, indent=1)
    log(stage="complete", seconds=round(result["seconds"]))
    return result

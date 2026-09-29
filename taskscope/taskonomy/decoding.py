"""Validation-selected Taskonomy decoders and decoded example exports."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import default_collate

from . import codec
from .data import TaskonomyRequirements

TASKS = codec.FAMILIES["DSE"]
EXAMPLES = (("d30", "d_r30", "D lambda30"), ("ds3", "ds_r3", "DS lambda3"),
            ("de3", "de_r3", "DE lambda3"))


def read_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required input: {path}")
    return json.loads(path.read_text())


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def population_hash(keys: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(keys).encode()).hexdigest()


def relative_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as error:
        raise ValueError(f"Input {path.name} must be inside --checkpoint-root") from error


def checkpoint_path(name: str, root: Path) -> Path:
    path = Path(name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("Checkpoint identifiers must be relative to --checkpoint-root")
    return root / path


def _same_checkpoint(name: str, expected: str, root: Path) -> bool:
    expected_path = checkpoint_path(expected, root).resolve()
    supplied = Path(name)
    candidates = (supplied,) if supplied.is_absolute() else (supplied, root / supplied)
    return any(path.resolve() == expected_path for path in candidates)


def _validation_keys(directory: Path) -> list[str]:
    keys = read_json(directory / "data_keys.json")["val"]
    if len(keys) != 500 or len(set(keys)) != 500:
        raise ValueError("Decoder selection requires 500 distinct validation keys")
    return keys


def coded_records(directory: Path, keys: Sequence[str] | None = None) -> tuple[dict, list[dict]]:
    result = read_json(directory / "evaluation.json")
    path = directory / "coded_rates.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"Missing per-image coded records: {path}")
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    actual_keys = [row["key"] for row in records]
    if (len(records) != result["images"] or len(set(actual_keys)) != len(records)
            or not records):
        raise ValueError("Coded records must contain each evaluated image exactly once")
    if keys is not None and actual_keys != list(keys):
        raise ValueError("Coded evaluation and selection use different image populations")
    for row in records:
        if (not math.isfinite(row["bpp"]) or row["bpp"] < 0
                or row["bytes"] != sum(row[name] for name in ("main_bytes", "hyper_bytes", "header_bytes"))
                or row["bpp"] != 8 * row["bytes"] / (256 * 256)):
            raise ValueError("Coded records have inconsistent packet lengths or rates")
    mean = sum(row["bpp"] for row in records) / len(records)
    if not math.isclose(mean, result["actual_bpp"], rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("Mean coded rate differs from its per-image records")
    return result, records


def _candidate(directory: Path, filename: str, checkpoint: str, root: Path,
               config: Mapping[str, Any]) -> dict[str, Any]:
    path = directory / filename
    record = (json.loads(path.read_text().splitlines()[-1])
              if filename.endswith("jsonl") else read_json(path))
    if record.get("split", config.get("split", "val")) != "val" or record["images"] != 500:
        raise ValueError("Decoder selection requires the fixed 500 validation images")
    losses = record["losses"]
    if set(losses) != set(codec.FAMILIES[config["family"]]):
        raise ValueError("Validation losses differ from the trained task family")
    if any(not math.isfinite(value) or value < 0 for value in losses.values()):
        raise ValueError("Validation risks must be finite and nonnegative")
    if any(record["loss_images"][task] != 500 for task in losses):
        raise ValueError("Every validation risk must use the same 500 images")
    checkpoint_file = directory / checkpoint
    candidate = {
        "checkpoint": relative_path(checkpoint_file, root),
        "validation_source": relative_path(path, root),
        "validation_sha256": file_hash(path),
        "validation_step": record["step"],
        "losses": losses,
    }
    if checkpoint_file.is_file():
        candidate["checkpoint_sha256"] = file_hash(checkpoint_file)
    return candidate


def select_decoders(native: Path, probe: Path, root: Path,
                    rgb_evaluation: Path | None = None,
                    codec_evaluation: Path | None = None) -> dict[str, Any]:
    """Use native, best-probe, last-probe, then RGB-reference risks in that tie order."""
    for directory in (native, probe):
        read_json(directory / "finished.json")
    native_config = read_json(native / "config.json")
    probe_config = read_json(probe / "config.json")
    if native_config["mode"] != "codec":
        raise ValueError("The native run must train a compressed codec")
    if (probe_config["mode"] != "probe" or probe_config["head_channels"] != 48
            or probe_config["family"] != "DSE"):
        raise ValueError("Selection requires fresh width-48 DSE probes")
    for field in ("channels", "base_channels", "observation"):
        if native_config.get(field, "rgb") != probe_config.get(field, "rgb"):
            raise ValueError(f"The probe changes the encoder {field}")
    keys = _validation_keys(native)
    if _validation_keys(probe) != keys:
        raise ValueError("Native and probe validation populations differ")
    candidates = [
        _candidate(native, "best_validation.json", "best.pt", root, native_config),
        _candidate(probe, "best_validation.json", "best.pt", root, probe_config),
        _candidate(probe, "validation.jsonl", "last.pt", root, probe_config),
    ]
    if rgb_evaluation is not None:
        if native_config["family"] != "RGB" or native_config.get("observation", "rgb") != "rgb":
            raise ValueError("The reference route requires an RGB-input RGB codec")
        result, _ = coded_records(rgb_evaluation, keys)
        if result["split"] != "val":
            raise ValueError("RGB-reference selection must use validation data")
        if result.get("readout_profile") != "reconstructed_rgb_then_frozen_reference":
            raise ValueError("Expected evaluation of the frozen reference on reconstructed RGB")
        if not _same_checkpoint(result["codec_checkpoint"], candidates[0]["checkpoint"], root):
            raise ValueError("The RGB-reference evaluation used another codec checkpoint")
        if codec_evaluation is not None:
            native_result, _ = coded_records(codec_evaluation, keys)
            identity = native_result.get("codec_checkpoint", native_result.get("checkpoint"))
            if (native_result["split"] != "val"
                    or not _same_checkpoint(identity, candidates[0]["checkpoint"], root)):
                raise ValueError("The native validation evaluation used another codec or split")
        if (set(result["losses"]) != set(TASKS)
                or any(result["loss_images"][task] != 500 for task in TASKS)
                or any(not math.isfinite(value) or value < 0 for value in result["losses"].values())):
            raise ValueError("Reference risks must use all 500 validation images")
        checkpoint_path(result["reference_checkpoint"], root)
        candidate = {
            "kind": "rgb_reference", "checkpoint": result["reference_checkpoint"],
            "validation_source": relative_path(rgb_evaluation / "evaluation.json", root),
            "validation_sha256": file_hash(rgb_evaluation / "evaluation.json"),
            "validation_step": result["reference_checkpoint_step"], "losses": result["losses"],
        }
        reference_path = checkpoint_path(candidate["checkpoint"], root)
        if reference_path.is_file():
            candidate["checkpoint_sha256"] = file_hash(reference_path)
        candidates.append(candidate)
    heads = {}
    for task in TASKS:
        selected = min((row for row in candidates if task in row["losses"]),
                       key=lambda row: row["losses"][task])
        heads[task] = {key: value for key, value in selected.items() if key != "losses"}
        heads[task].update(kind=selected.get("kind", "latent"), validation_risk=selected["losses"][task])
    result = {
        "selection_split": "val", "validation_images": 500,
        "validation_keys_sha256": population_hash(keys),
        "codec_checkpoint": candidates[0]["checkpoint"], "heads": heads,
        "selection_policy": "native plus retained fresh width-48 best and last heads",
        "selection_basis": "lowest quantized-forward validation risk per task",
        "eligible_sources": candidates,
    }
    if "checkpoint_sha256" in candidates[0]:
        result["codec_checkpoint_sha256"] = candidates[0]["checkpoint_sha256"]
    if rgb_evaluation is not None:
        result["selection_policy"] += " plus reconstructed-RGB frozen-reference route"
        result["selection_basis"] += ", including entropy-decoded RGB-reference validation risks"
    return result


class SelectedDecoders(nn.Module):
    def __init__(self, message: nn.Module, sources: Mapping[str, nn.Module],
                 routes: Mapping[str, tuple[str, str]]) -> None:
        super().__init__()
        self.codec = message
        self.sources = nn.ModuleDict(sources)
        self.routes = dict(routes)

    @property
    def stream(self) -> nn.Module:
        return self.codec.stream

    def code(self, inputs: Tensor) -> tuple[Tensor, Any]:
        return self.codec.code(inputs)

    def predict(self, latent: Tensor, tasks: Sequence[str]) -> dict[str, Tensor]:
        groups = defaultdict(list)
        for task in tasks:
            groups[self.routes[task]].append(task)
        result, reconstruction = {}, None
        for (source_key, kind), selected_tasks in groups.items():
            source = self.sources[source_key]
            if kind == "latent":
                result.update(source.predict(latent, selected_tasks))
            else:
                if reconstruction is None:
                    reconstruction = self.codec.predict(latent, ("rgb",))["rgb"]
                reference_latent, _ = source.encode(reconstruction, False)
                result.update(source.predict(reference_latent, selected_tasks))
        return result


def _load_checkpoint(name: str, root: Path, device: torch.device,
                     expected_hash: str | None = None) -> tuple[nn.Module, dict]:
    path = checkpoint_path(name, root)
    if not path.is_file():
        raise FileNotFoundError(f"Missing external checkpoint {name}. Supply it under --checkpoint-root.")
    if expected_hash is not None and file_hash(path) != expected_hash:
        raise ValueError(f"Checkpoint {name} differs from the validation-selected file")
    return codec.load_model(path, device)


def _same_representation(message: nn.Module, decoder: nn.Module) -> bool:
    for group in ("analysis", "stream"):
        left = dict(getattr(message, group).named_parameters())
        right = dict(getattr(decoder, group).named_parameters())
        if left.keys() != right.keys() or any(not torch.equal(value, right[name])
                                            for name, value in left.items()):
            return False
    # CDF tables are derived caches and may have been saved before or after evaluation.
    message.stream.update(force=True)
    decoder.stream.update(force=True)
    for group in ("analysis", "stream"):
        left = getattr(message, group).state_dict()
        right = getattr(decoder, group).state_dict()
        if left.keys() != right.keys() or any(not torch.equal(value, right[name])
                                            for name, value in left.items()):
            return False
    return True


def load_composition(selection: Mapping[str, Any], root: Path, device: torch.device
                     ) -> tuple[SelectedDecoders, dict, dict[str, dict]]:
    if selection.get("selection_split") != "val" or selection.get("validation_images") != 500:
        raise ValueError("Decoder choices must be fixed on the 500-image validation set")
    if set(selection["heads"]) != set(TASKS):
        raise ValueError("Choose one decoder for each DSE requirement")
    message, state = _load_checkpoint(selection["codec_checkpoint"], root, device,
                                      selection.get("codec_checkpoint_sha256"))
    if not codec.checkpoint_compressed(state):
        raise ValueError("The selected message must be compressed")
    sources, states, routes, indices = {}, {}, {}, {}
    for task, choice in selection["heads"].items():
        name = choice["checkpoint"]
        if name not in indices:
            key = str(len(indices))
            indices[name] = key
            sources[key], states[key] = _load_checkpoint(name, root, device,
                                                         choice.get("checkpoint_sha256"))
        key = indices[name]
        source, source_state = sources[key], states[key]
        config = source_state["config"]
        if task not in codec.FAMILIES[config["family"]]:
            raise ValueError(f"The selected {task} decoder was not trained for that task")
        if source_state["step"] != choice["validation_step"]:
            raise ValueError(f"The selected {task} checkpoint step differs from its validation record")
        statistics = ("semantic_weights", "semantic_counts", "depth_mean_m",
                      "depth_geometric_mean_m", "edge_mean")
        if any(source_state["calibration"][field] != state["calibration"][field]
               for field in statistics):
            raise ValueError("Selected decoders must use the same training calibration")
        kind = choice["kind"]
        if kind == "latent":
            if (not codec.checkpoint_compressed(source_state)
                    or config.get("observation", "rgb") != state["config"].get("observation", "rgb")
                    or not _same_representation(message, source)):
                raise ValueError(f"The {task} decoder comes from a different coded representation")
        elif kind == "rgb_reference":
            if (state["config"]["family"] != "RGB"
                    or state["config"].get("observation", "rgb") != "rgb"
                    or config["family"] != "DSE" or codec.checkpoint_compressed(source_state)
                    or config.get("observation", "rgb") != "rgb"):
                raise ValueError("The cascade requires an RGB codec and an uncompressed RGB-input DSE reference")
        else:
            raise ValueError(f"Unknown decoder route {kind}")
        routes[task] = (key, kind)
    return SelectedDecoders(message, sources, routes), state, states


def rgb_selection(message: str, reference: str, root: Path, device: torch.device) -> dict:
    _, state = _load_checkpoint(reference, root, device)
    return {
        "selection_split": "val", "validation_images": 500, "codec_checkpoint": message,
        "heads": {task: {"kind": "rgb_reference", "checkpoint": reference,
                          "validation_step": state["step"]} for task in TASKS},
    }


def evaluate_selection(selection: Mapping[str, Any], root: Path, data_args: argparse.Namespace,
                       split: str, samples: int, output: Path, device: torch.device) -> dict:
    model, state, source_states = load_composition(selection, root, device)
    data = codec.loader(data_args, split, samples)
    keys = [sample.key for sample in data.dataset.samples]
    if (split == "val" and "validation_keys_sha256" in selection
            and population_hash(keys) != selection["validation_keys_sha256"]):
        raise ValueError("Evaluation does not use the selected validation population")
    calibration = state["calibration"]
    weights = torch.tensor(calibration["semantic_weights"], device=device)
    output.mkdir(parents=True, exist_ok=True)
    result = codec.evaluate(model, data, device, TASKS, True, weights, calibration,
                            coded=True, per_image_path=output / "coded_rates.jsonl",
                            observation=state["config"].get("observation", "rgb"))
    result.update(
        split=split, readout_profile="available_validation_selected_heads", selection=selection,
        codec_checkpoint=selection["codec_checkpoint"], codec_checkpoint_step=state["step"],
        checkpoint_sha256={name: file_hash(checkpoint_path(name, root)) for name in
                           {selection["codec_checkpoint"], *(choice["checkpoint"] for choice in selection["heads"].values())}},
        head_checkpoint_steps={task: source_states[model.routes[task][0]]["step"] for task in TASKS},
        extra_decoder_computation=any(choice["kind"] == "rgb_reference" for choice in selection["heads"].values()),
    )
    codec.write_json(output / "evaluation.json", result)
    return result


@torch.inference_mode()
def export_examples(heads: Path, fixed_metadata: Path, root: Path,
                    data_args: argparse.Namespace, output: Path, device: torch.device) -> None:
    historical = read_json(fixed_metadata)
    data = TaskonomyRequirements(Path(data_args.data_root), "val", 500, Path(data_args.manifest))
    batch = codec.prepared(default_collate([data[index] for index in range(5)]), device)
    keys = list(batch["key"])
    if keys != historical["keys"]:
        raise ValueError("Decoded examples must retain the five preselected validation image keys")
    arrays = {name: batch[name].cpu().numpy() for name in
              ("rgb", "depth", "depth_valid", "semantic", "edge")}
    arrays["keys"] = np.array(keys)
    sources = {}
    all_keys = [sample.key for sample in data.samples]
    for prefix, directory, role in EXAMPLES:
        path = heads / directory
        selection = read_json(path / "selection.json")
        evaluation, _ = coded_records(path, all_keys)
        if evaluation["split"] != "val" or evaluation["images"] != 500:
            raise ValueError("Example rate labels require the 500-image validation evaluation")
        if evaluation.get("selection") != selection:
            raise ValueError("Example decoder selection differs from its rate evaluation")
        if "checkpoint_sha256" in evaluation:
            names = {selection["codec_checkpoint"],
                     *(choice["checkpoint"] for choice in selection["heads"].values())}
            current_hashes = {name: file_hash(checkpoint_path(name, root)) for name in names}
            if current_hashes != evaluation["checkpoint_sha256"]:
                raise ValueError("Example checkpoints differ from the retained validation evaluation")
        model, state, _ = load_composition(selection, root, device)
        if state["config"]["family"] != role.split()[0]:
            raise ValueError("Example codec family differs from its figure label")
        model.eval()
        model.stream.update(force=True)
        inputs = codec.observation_input(batch, state["calibration"], state["config"].get("observation", "rgb"))
        decoded, packet_bytes = [], []
        for value in inputs:
            latent, packet = model.code(value[None])
            decoded.append(latent)
            packet_bytes.append(packet.total_bytes)
        for task, value in model.predict(torch.cat(decoded), TASKS).items():
            arrays[f"{prefix}_{task}"] = (value.argmax(1).byte() if task == "semantic" else value).cpu().numpy()
        sources[prefix] = {
            "selection": selection, "evaluation_source": f"{directory}/evaluation.json",
            "evaluation_sha256": file_hash(path / "evaluation.json"),
            "validation_images": 500, "actual_mean_bpp": evaluation["actual_bpp"],
            "example_packet_bytes": packet_bytes,
        }
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "focal_readout_example_predictions.npz", **arrays)
    codec.write_json(output / "focal_readout_example_predictions.json", {
        "selection": historical["selection"], "keys": keys,
        "roles": {prefix: role for prefix, _, role in EXAMPLES},
        "decoding": "actual entropy decoding with the fixed primary validation-selected task heads",
        "sources": sources,
    })

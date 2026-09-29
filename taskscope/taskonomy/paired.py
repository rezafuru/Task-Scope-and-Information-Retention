"""Paired image bootstrap and fixed-setting aggregation for retained Taskonomy fits.

Replicates resample images within their observed building, keeping each building's image
count, with the codec and readout weights held fixed. The resulting percentile intervals are
componentwise and condition on the fitted models and the observed buildings, so they carry no
training-seed, joint-family, or new-building coverage.

Both entry points read per-image record files (one JSON object per line, keyed by
``building/point``). Those records are not part of the released evidence.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np

TASKS = ("depth", "semantic", "edge")
FAMILIES = {"D": ("depth",), "DS": ("depth", "semantic"),
            "DE": ("depth", "edge"), "DSE": TASKS}
SEEDS = (20260905, 20260906, 20260907)
REPETITIONS = 2000
RESAMPLING_SEED = 20260906


def read_records(path: Path | str) -> dict[str, dict]:
    records = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    keys = [row["key"] for row in records]
    if len(keys) != len(set(keys)) or not keys:
        raise ValueError(f"Empty or duplicate image keys in {path}")
    return {row["key"]: row for row in records}


def vector(records: dict, keys, field: str, task: str | None = None) -> np.ndarray:
    values = [records[key].get(field, {}).get(task) if task else records[key].get(field)
              for key in keys]
    return np.asarray([np.nan if value is None else value for value in values], dtype=float)


def mean(values: np.ndarray) -> float | None:
    valid = np.isfinite(values)
    return float(values[valid].mean()) if valid.any() else None


def metrics(records: dict, keys) -> dict:
    result = {
        "images": len(keys),
        "losses": {task: mean(vector(records, keys, "losses", task)) for task in TASKS},
        "constant_losses": {task: mean(vector(records, keys, "constant_losses", task))
                            for task in TASKS},
        "actual_bpp": mean(vector(records, keys, "bpp")),
    }
    confusion = np.sum([records[key]["semantic_confusion"] for key in keys], axis=0, dtype=np.int64)
    union = confusion.sum(0) + confusion.sum(1) - np.diag(confusion)
    present = confusion.sum(1)[1:] > 0
    result["semantic_object_miou"] = (
        float((np.diag(confusion)[1:][present] / union[1:][present]).mean())
        if present.any() else None)
    pixels = sum(records[key]["depth"]["pixels"] for key in keys)
    result["depth_rmse_m"] = float(np.sqrt(
        sum(records[key]["depth"]["squared_error_sum"] for key in keys) / pixels)) if pixels else None
    return result


def stratified_weights(keys, repetitions: int, seed: int) -> np.ndarray:
    """Resample images within each observed building, retaining its image count."""
    rng = np.random.default_rng(seed)
    buildings = np.asarray([key.split("/", 1)[0] for key in keys])
    weights = np.zeros((repetitions, len(keys)), dtype=np.int32)
    for building in sorted(set(buildings)):
        indices = np.flatnonzero(buildings == building)
        weights[:, indices] = rng.multinomial(
            len(indices), np.full(len(indices), 1 / len(indices)), size=repetitions)
    return weights


def bootstrap_mean(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    valid = np.isfinite(values)
    numerator = np.einsum("ij,j->i", weights[:, valid], values[valid], optimize=False)
    denominator = weights[:, valid].sum(1)
    return np.divide(numerator, denominator, out=np.full(len(weights), np.nan),
                     where=denominator > 0)


def interval(values: np.ndarray) -> dict:
    finite = np.isfinite(values)
    return {"lower": float(np.quantile(values[finite], .025)) if finite.any() else None,
            "upper": float(np.quantile(values[finite], .975)) if finite.any() else None,
            "defined_replicates": int(finite.sum())}


def analyze(paths: dict[str, Path | str], reference_path: Path | str,
            tolerances_path: Path | str, repetitions: int = REPETITIONS,
            seed: int = RESAMPLING_SEED,
            validation_reference: Path | str | None = None) -> dict:
    """Summarize every named candidate against a shared reference on one image set.

    ``validation_reference`` overrides the location the tolerances record names relative to
    itself, which the released evidence layout no longer reproduces.
    """
    if repetitions < 100:
        raise ValueError("Use at least 100 resampling replicates")
    all_records = {name: read_records(path) for name, path in paths.items()}
    reference = read_records(reference_path)
    keys = sorted(reference)
    for name, records in all_records.items():
        if set(records) != set(keys):
            raise ValueError(f"Image-key mismatch for {name}")
        for task in TASKS:
            if not np.array_equal(np.isfinite(vector(records, keys, "losses", task)),
                                  np.isfinite(vector(reference, keys, "losses", task))):
                raise ValueError(f"Valid-image mismatch for {name}, {task}")
    requirement_record = json.loads(Path(tolerances_path).read_text())
    tolerances = requirement_record["tolerances"]
    validation_reference_path = (Path(validation_reference) if validation_reference is not None
                                 else Path(tolerances_path).parent / requirement_record["reference"])
    validation_reference_record = json.loads(validation_reference_path.read_text())
    weights = stratified_weights(keys, repetitions, seed)
    sources = {"reference": reference, **all_records}
    point = {name: metrics(records, keys) for name, records in sources.items()}
    draws = {name: {task: bootstrap_mean(vector(records, keys, "losses", task), weights)
                    for task in TASKS} for name, records in sources.items()}
    constant_draws = {task: bootstrap_mean(vector(reference, keys, "constant_losses", task), weights)
                      for task in TASKS}
    rows = []
    for name, records in sources.items():
        row = {"name": name, **point[name],
               "loss_intervals_95": {task: interval(draws[name][task]) for task in TASKS}}
        row["rate_interval_95"] = interval(bootstrap_mean(vector(records, keys, "bpp"), weights))
        row["normalized_excess"] = {}
        row["fixed_validation_excess"] = {}
        for task in TASKS:
            denominator = point["reference"]["constant_losses"][task] - point["reference"]["losses"][task]
            boot_denominator = constant_draws[task] - draws["reference"][task]
            normalized = np.divide(draws[name][task] - draws["reference"][task], boot_denominator,
                                   out=np.full(repetitions, np.nan), where=boot_denominator > 0)
            estimate = ((point[name]["losses"][task] - point["reference"]["losses"][task])
                        / denominator) if denominator > 0 else None
            row["normalized_excess"][task] = {"estimate": estimate,
                                              "interval_95": interval(normalized)}
            validation_loss = validation_reference_record["losses"][task]
            validation_gap = validation_reference_record["constant_losses"][task] - validation_loss
            fixed_draws = ((draws[name][task] - validation_loss) / validation_gap
                           if validation_gap > 0 else np.full(repetitions, np.nan))
            fixed_estimate = ((point[name]["losses"][task] - validation_loss) / validation_gap
                              if validation_gap > 0 else None)
            row["fixed_validation_excess"][task] = {"estimate": fixed_estimate,
                                                    "interval_95": interval(fixed_draws)}
        row["point_feasibility"] = {
            alpha: {family: all(row["losses"][task] <= threshold[task] for task in tasks)
                    for family, tasks in FAMILIES.items()}
            for alpha, threshold in tolerances.items()}
        row["buildings"] = {
            building: metrics(records, [key for key in keys if key.split("/", 1)[0] == building])
            for building in sorted({key.split("/", 1)[0] for key in keys})}
        rows.append(row)
    contrasts = []
    names = list(all_records)
    for index, left in enumerate(names):
        for right in names[index + 1:]:
            rate_difference = bootstrap_mean(vector(all_records[left], keys, "bpp")
                                             - vector(all_records[right], keys, "bpp"), weights)
            contrasts.append({
                "left": left, "right": right, "direction": "left minus right",
                "loss_differences": {
                    task: {"estimate": point[left]["losses"][task] - point[right]["losses"][task],
                           "interval_95": interval(draws[left][task] - draws[right][task])}
                    for task in TASKS},
                "rate_difference": {
                    "estimate": point[left]["actual_bpp"] - point[right]["actual_bpp"],
                    "interval_95": interval(rate_difference)}})
    return {
        "scope": "fixed codec/readout seeds, paired image resampling conditional on the "
                 "observed buildings",
        "intervals": "componentwise percentile 95%, no joint family-feasibility coverage claim",
        "training_seed_variation": "not estimated by image resampling, report separately",
        "sources": {"reference": str(reference_path),
                    **{name: str(path) for name, path in paths.items()}},
        "normalization_definitions": {
            "normalized_excess": "(current-split codec risk - current-split reference risk) / "
                                 "(current-split constant risk - current-split reference risk), "
                                 "descriptive",
            "fixed_validation_excess": "(current-split codec risk - fixed validation reference "
                                       "risk) / (fixed validation constant risk - fixed "
                                       "validation reference risk), alpha is the fixed "
                                       "requirement"},
        "validation_reference_source": str(validation_reference_path),
        "tolerances_source": str(tolerances_path), "bootstrap_repetitions": repetitions,
        "resampling_seed": seed, "images": len(keys), "rows": rows,
        "paired_contrasts": contrasts}


def summarize(manifest: dict, repetitions: int = REPETITIONS,
              resolve: Callable[[str], Path] = Path,
              validation_reference: Path | str | None = None) -> dict:
    """Aggregate the three fixed training seeds of every setting named by the manifest.

    ``resolve`` maps the recorded relative source strings onto real files. Seed-to-seed spread
    is reported as ranges and per-seed outcomes, never folded into the image intervals.
    """
    groups = manifest["settings"]
    if len({row["setting"] for row in groups}) != len(groups):
        raise ValueError("Setting names must be distinct")
    paths: dict[str, Path] = {}
    names: dict[str, list[str]] = {}
    for group in groups:
        if sorted(row["seed"] for row in group["replicates"]) != list(SEEDS):
            raise ValueError("Every setting requires the three fixed training seeds")
        names[group["setting"]] = []
        for row in group["replicates"]:
            name = f"{group['setting']}_s{row['seed']}"
            names[group["setting"]].append(name)
            paths[name] = resolve(row["records"])
    tolerances_path = resolve(manifest["tolerances"])
    detailed = analyze(paths, resolve(manifest["reference"]), tolerances_path, repetitions,
                       validation_reference=validation_reference)
    points = {row["name"]: row for row in detailed["rows"]}
    records = {name: read_records(path) for name, path in paths.items()}
    reference = read_records(resolve(manifest["reference"]))
    keys = sorted(reference)
    weights = stratified_weights(keys, repetitions, RESAMPLING_SEED)
    reference_draws = {task: bootstrap_mean(vector(reference, keys, "losses", task), weights)
                       for task in TASKS}
    constant_draws = {task: bootstrap_mean(vector(reference, keys, "constant_losses", task), weights)
                      for task in TASKS}
    validation = json.loads(Path(detailed["validation_reference_source"]).read_text())
    thresholds = json.loads(Path(tolerances_path).read_text())["tolerances"]
    rows, draws, rate_draws = [], {}, {}
    for setting, members in names.items():
        member_points = [points[name] for name in members]
        vectors = {task: np.mean([vector(records[name], keys, "losses", task) for name in members],
                                 axis=0) for task in TASKS}
        rates = np.mean([vector(records[name], keys, "bpp") for name in members], axis=0)
        draws[setting] = {task: bootstrap_mean(values, weights) for task, values in vectors.items()}
        rate_draws[setting] = bootstrap_mean(rates, weights)
        losses = {task: mean(values) for task, values in vectors.items()}
        row = {
            "setting": setting, "seeds": list(SEEDS), "members": members,
            "actual_bpp": mean(rates), "losses": losses,
            "loss_intervals_95": {task: interval(value) for task, value in draws[setting].items()},
            "rate_interval_95": interval(rate_draws[setting]),
            "seed_ranges": {
                "actual_bpp": [min(p["actual_bpp"] for p in member_points),
                               max(p["actual_bpp"] for p in member_points)],
                "losses": {task: [min(p["losses"][task] for p in member_points),
                                  max(p["losses"][task] for p in member_points)] for task in TASKS}},
            "mean_seed_metrics": {metric: float(np.mean([p[metric] for p in member_points]))
                                  for metric in ("semantic_object_miou", "depth_rmse_m")},
            "normalized_excess": {}, "fixed_validation_excess": {}}
        for task in TASKS:
            gap = points["reference"]["constant_losses"][task] - points["reference"]["losses"][task]
            boot_gap = constant_draws[task] - reference_draws[task]
            excess = np.divide(draws[setting][task] - reference_draws[task], boot_gap,
                               out=np.full(repetitions, np.nan), where=boot_gap > 0)
            estimate = ((losses[task] - points["reference"]["losses"][task]) / gap
                        if gap > 0 else None)
            row["normalized_excess"][task] = {"estimate": estimate,
                                              "interval_95": interval(excess)}
            validation_gap = validation["constant_losses"][task] - validation["losses"][task]
            row["fixed_validation_excess"][task] = {
                "estimate": (losses[task] - validation["losses"][task]) / validation_gap,
                "interval_95": interval((draws[setting][task] - validation["losses"][task])
                                        / validation_gap)}
        row["point_feasibility"] = {
            alpha: {family: all(losses[task] <= limit[task] for task in tasks)
                    for family, tasks in FAMILIES.items()}
            for alpha, limit in thresholds.items()}
        row["feasible_seed_counts"] = {
            alpha: {family: sum(p["point_feasibility"][alpha][family] for p in member_points)
                    for family in FAMILIES} for alpha in thresholds}
        rows.append(row)
    contrasts = []
    for index, left in enumerate(rows):
        for right in rows[index + 1:]:
            first, second = left["setting"], right["setting"]
            contrasts.append({
                "left": first, "right": second, "direction": "left minus right",
                "loss_differences": {
                    task: {"estimate": left["losses"][task] - right["losses"][task],
                           "interval_95": interval(draws[first][task] - draws[second][task])}
                    for task in TASKS},
                "rate_difference": {
                    "estimate": left["actual_bpp"] - right["actual_bpp"],
                    "interval_95": interval(rate_draws[first] - rate_draws[second])}})
    decisions = []
    for alpha in thresholds:
        for family in FAMILIES:
            feasible = [row for row in rows if row["point_feasibility"][alpha][family]]
            selected = min(feasible, key=lambda row: row["actual_bpp"]) if feasible else None
            decisions.append({"alpha": float(alpha), "family": family,
                              "selected_setting": selected["setting"] if selected else None,
                              "lowest_observed_mean_bpp": selected["actual_bpp"] if selected else None})
    result = {
        "split": manifest["split"], "readout_profile": manifest["readout_profile"],
        "manifest": manifest, "images": len(keys), "rows": rows,
        "interval_scope": "paired image resampling within the observed buildings, conditional on "
                          "the same three fixed fits per setting",
        "seed_variation": "reported separately as seed ranges and every seed outcome, not "
                          "estimated by the image intervals",
        "mean_metric_definition": "arithmetic mean of seed-level mIoU and RMSE, no ensemble "
                                  "prediction",
        "decision_scope": "point-risk decisions within this fixed three-setting subset on the "
                          "reported split",
        "decisions": decisions, "paired_setting_contrasts": contrasts,
        "per_seed_analysis": detailed}
    if manifest["split"] == "test":
        selected = json.loads(resolve(manifest["validation_decisions_source"]).read_text())
        if selected["split"] != "val" or selected["readout_profile"] != manifest["readout_profile"]:
            raise ValueError("Held-out decisions require validation choices for the same readout policy")
        if {row["setting"] for row in selected["rows"]} != {row["setting"] for row in rows}:
            raise ValueError("Validation and held-out setting sets differ")
        lookup = {row["setting"]: row for row in rows}
        heldout = []
        for decision in selected["decisions"]:
            setting = decision["selected_setting"]
            row = lookup[setting] if setting is not None else None
            alpha, family = str(decision["alpha"]), decision["family"]
            heldout.append({
                **decision,
                "validation_lowest_observed_mean_bpp": decision["lowest_observed_mean_bpp"],
                "test_actual_bpp": row["actual_bpp"] if row else None,
                "test_losses": row["losses"] if row else None,
                "test_point_feasible": row["point_feasibility"][alpha][family] if row else None,
                "test_feasible_seed_count": row["feasible_seed_counts"][alpha][family] if row else None})
        result["validation_selected_heldout_decisions"] = heldout
        result["decision_scope"] = ("decisions are descriptive test-pool choices; "
                                    "validation_selected_heldout_decisions evaluate the "
                                    "previously selected settings")
    return result

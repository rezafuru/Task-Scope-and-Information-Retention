"""Pool the CIFAR-100 codec runs into candidate groups and apply the frozen selection.

One group is a pair of codec fits (seeds 17 and 23) under one teacher, one loss, and
one rate weight, scored through two readout fits (seeds 17 and 23). Accuracy drops are
measured against the same teacher's uncoded reference, so every comparison stays within
a teacher.

On the validation split the rule picks, per teacher and per loss, the group with the
lowest mean complete rate that meets both accuracy allowances, and writes that choice
out. On the test split the stored validation file is required, its hash is recorded in
the report, and the selections it fixed are carried over unchanged.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

TEACHER_SEEDS = (17, 23, 29)
CODEC_SEEDS = (17, 23)
READOUT_DIRECTORIES = ("readouts_s17", "readouts_s23")
COARSE_ALLOWANCE = 0.03
FINE_ALLOWANCES = (0.03, 0.05, 0.10)
INTERVAL_SEED = 20260907

INTERVAL_SCOPE = (
    "Paired images resampled within fine classes, conditional on each teacher and the retained "
    "codec/readout fits. Teacher variation is descriptive."
)
SELECTION_RULE = (
    "Lowest mean actual rate among paired codec groups meeting both mean validation accuracy "
    "allowances."
)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def mean_predictions(path: Path, split: str):
    """Per-image correctness of the four readouts stored in one scoring directory."""
    archive = np.load(path / f"{split}_readout_predictions.npz")
    labels = archive["fine_target"]
    correct = {
        name: archive[f"{name}_logits"].argmax(1) == labels
        for name in ("early_linear", "early_mlp", "late_linear", "late_mlp")
    }
    return labels, correct


def interval(values, labels, samples: int = 2000) -> list[float]:
    """Percentile interval of a mean over images resampled within fine classes."""
    values = np.asarray(values, dtype=float)
    generator = np.random.default_rng(INTERVAL_SEED)
    draws = np.zeros(samples)
    for label in np.unique(labels):
        positions = np.flatnonzero(labels == label)
        counts = generator.multinomial(len(positions), np.full(len(positions), 1 / len(positions)),
                                       size=samples)
        draws += np.einsum("ij,j->i", counts, values[positions]) / len(labels)
    return np.quantile(draws, [0.025, 0.975]).tolist()


def analyze(root: Path, split: str, selection_path: Path | None) -> Path:
    """Aggregate one split and write its summary, per-fit rows, and selections."""
    rows, groups, missing, references, unreplicated = [], [], [], [], []
    arrays = {}
    for teacher_seed in TEACHER_SEEDS:
        teacher = root / f"teacher_{teacher_seed}"
        reference_paths = [teacher, teacher / "readouts_s23"]
        if any(not (path / name).exists() for path in reference_paths
               for name in (f"{split}_readout_predictions.npz", f"{split}_readouts.json")):
            missing.append(f"teacher_{teacher_seed} raw {split} readout scores")
            continue
        reference_scores = [mean_predictions(path, split) for path in reference_paths]
        labels = reference_scores[0][0]
        for other_labels, _ in reference_scores:
            if not np.array_equal(other_labels, labels):
                raise ValueError(f"Raw readout targets differ for teacher {teacher_seed}")
        reference_fine = np.mean([correct["early_mlp"] for _, correct in reference_scores], axis=0)
        raw = np.load(teacher / f"{split}_readout_predictions.npz")
        reference_coarse = raw["coarse_prediction"] == raw["coarse_target"]
        references.append({
            "teacher_seed": teacher_seed,
            "coarse_accuracy": float(reference_coarse.mean()),
            "early_fine_accuracy": float(reference_fine.mean()),
            "late_fine_accuracy": float(np.mean([correct["late_mlp"].mean()
                                                 for _, correct in reference_scores])),
            "early_readout_seed_values": [float(correct["early_mlp"].mean())
                                          for _, correct in reference_scores],
            "late_readout_seed_values": [float(correct["late_mlp"].mean())
                                         for _, correct in reference_scores],
        })
        candidates = {}
        declared_fits = {}
        for run in sorted(teacher.glob("codec_early_*")):
            if not run.name.endswith(("_controlled", "_extension", "_warm")):
                continue
            config = read_json(run / "config.json")
            if config["seed"] not in CODEC_SEEDS or config["objective"] not in ("early", "late"):
                continue
            key = (config["objective"], config["beta"])
            declared_fits.setdefault(key, set()).add(config["seed"])
            readout_paths = [run / name for name in READOUT_DIRECTORIES]
            coded_path = run / f"{split}_coded_results.npz"
            summary_path = run / ("coded_test_summary.json" if split == "test"
                                  else "coded_summary.json")
            if (not coded_path.exists() or not summary_path.exists()
                    or any(not (path / name).exists() for path in readout_paths
                           for name in (f"{split}_readout_predictions.npz",
                                        f"{split}_readouts.json"))):
                missing.append(f"teacher_{teacher_seed}/{run.name} {split} scores")
                continue
            coded = np.load(coded_path)
            if (not np.array_equal(coded["fine_target"], labels)
                    or not np.array_equal(coded["coarse_target"], raw["coarse_target"])):
                raise ValueError(f"Coded targets differ in {run}")
            readouts = [mean_predictions(path, split) for path in readout_paths]
            for other_labels, _ in readouts:
                if not np.array_equal(other_labels, labels):
                    raise ValueError(f"Readout targets differ in {run}")
            fine = np.mean([correct["early_mlp"] for _, correct in readouts], axis=0)
            coarse = coded["coarse_prediction"] == coded["coarse_target"]
            metrics = read_json(summary_path)[split]
            row = {
                "teacher_seed": teacher_seed,
                "codec_seed": config["seed"],
                "objective": config["objective"],
                "beta": config["beta"],
                "run": run.name,
                "actual_bpp": float(coded["bytes"][:, 0].mean() / 128),
                "coarse_accuracy": float(coarse.mean()),
                "fine_accuracy": float(fine.mean()),
                "fine_readout_seed_values": [float(correct["early_mlp"].mean())
                                             for _, correct in readouts],
                "logit_fine_accuracy": float(np.mean([correct["late_mlp"].mean()
                                                      for _, correct in readouts])),
                "fixed_fine_accuracy": float((coded["fine_prediction_fixed"] == labels).mean()),
                "reference_coarse_accuracy": float(reference_coarse.mean()),
                "reference_fine_accuracy": float(reference_fine.mean()),
                "coarse_drop": float((reference_coarse.astype(float) - coarse).mean()),
                "fine_drop": float((reference_fine - fine).mean()),
                "early_mse": metrics.get("early_mse"),
                "late_mse": metrics.get("late_mse"),
                "mean_header_bytes": float(coded["bytes"][:, 1].mean()),
                "mean_hyper_bytes": float(coded["bytes"][:, 2].mean()),
                "mean_main_bytes": float(coded["bytes"][:, 3].mean()),
            }
            row["point_feasible"] = (row["coarse_drop"] <= COARSE_ALLOWANCE
                                     and row["fine_drop"] <= 0.05)
            rows.append(row)
            key = (config["objective"], config["beta"])
            if config["seed"] in candidates.setdefault(key, {}):
                raise ValueError(f"Duplicate fitted seed for teacher {teacher_seed}, {key}")
            candidates[key][config["seed"]] = (row, fine, coarse)
        for (objective, beta), fits in candidates.items():
            if set(fits) != set(CODEC_SEEDS):
                key = f"teacher_{teacher_seed}/{objective}/beta_{beta}"
                if declared_fits[objective, beta] == set(CODEC_SEEDS):
                    missing.append(f"{key} paired codec fit")
                else:
                    unreplicated.append({"key": key, "codec_seeds": sorted(fits)})
                continue
            fit_rows = [fits[seed][0] for seed in CODEC_SEEDS]
            fine = np.mean([fits[seed][1] for seed in CODEC_SEEDS], axis=0)
            coarse = np.mean([fits[seed][2] for seed in CODEC_SEEDS], axis=0)
            key = f"teacher_{teacher_seed}/{objective}/beta_{beta:g}"
            group = {
                "key": key, "teacher_seed": teacher_seed, "objective": objective, "beta": beta,
                "runs": [row["run"] for row in fit_rows],
                "actual_bpp": float(np.mean([row["actual_bpp"] for row in fit_rows])),
                "coarse_accuracy": float(coarse.mean()), "fine_accuracy": float(fine.mean()),
                "fixed_fine_accuracy": float(np.mean([row["fixed_fine_accuracy"]
                                                      for row in fit_rows])),
                "logit_fine_accuracy": float(np.mean([row["logit_fine_accuracy"]
                                                      for row in fit_rows])),
                "reference_fine_accuracy": float(reference_fine.mean()),
                "reference_coarse_accuracy": float(reference_coarse.mean()),
                "codec_mean_fine_seed_values": [row["fine_accuracy"] for row in fit_rows],
                "readout_fine_seed_values": [row["fine_readout_seed_values"] for row in fit_rows],
                "coarse_drop": float((reference_coarse - coarse).mean()),
                "fine_drop": float((reference_fine - fine).mean()),
                "individual_fit_feasible": [row["point_feasible"] for row in fit_rows],
            }
            group["point_feasible"] = (group["coarse_drop"] <= COARSE_ALLOWANCE
                                       and group["fine_drop"] <= 0.05)
            group["fine_allowance_feasibility"] = {
                str(allowance): (group["coarse_drop"] <= COARSE_ALLOWANCE
                                 and group["fine_drop"] <= allowance)
                for allowance in FINE_ALLOWANCES
            }
            groups.append(group)
            arrays[key] = (labels, reference_coarse - coarse, reference_fine - fine, fine)
    selections_by_allowance = {}
    for allowance in FINE_ALLOWANCES:
        selections = []
        for teacher_seed in TEACHER_SEEDS:
            for objective in ("early", "late"):
                eligible = [group for group in groups
                            if group["teacher_seed"] == teacher_seed
                            and group["objective"] == objective
                            and group["fine_allowance_feasibility"][str(allowance)]]
                if eligible:
                    selections.append(min(eligible, key=lambda group: group["actual_bpp"])["key"])
        selections_by_allowance[str(allowance)] = selections
    if selection_path:
        selection = read_json(selection_path)
        if selection.get("split") != "validation" or selection.get("test_used") is not False:
            raise ValueError("The frozen candidate selection must come from validation-only evidence")
        frozen = selection["selected_groups"]
        selections_by_allowance = selection["selected_groups_by_fine_allowance"]
        required = set().union(*(set(keys) for keys in selections_by_allowance.values()),
                               set(frozen))
        absent = required - {group["key"] for group in groups}
        if split == "test" and absent:
            raise ValueError(f"Selected groups lack complete test assessment: {sorted(absent)}")
    else:
        frozen = selections_by_allowance["0.05"]
    for group in groups:
        group["selected_on_validation"] = group["key"] in frozen
        group["selected_fine_allowances"] = [float(allowance) for allowance, keys
                                             in selections_by_allowance.items()
                                             if group["key"] in keys]
        if split == "test":
            labels, coarse_drop, fine_drop, _ = arrays[group["key"]]
            group["coarse_drop_image_ci95"] = interval(coarse_drop, labels)
            group["fine_drop_image_ci95"] = interval(fine_drop, labels)
    comparisons = []
    for teacher_seed in TEACHER_SEEDS:
        selected = {group["objective"]: group for group in groups
                    if group["teacher_seed"] == teacher_seed and group["selected_on_validation"]}
        if set(selected) != {"early", "late"}:
            continue
        early, late = selected["early"], selected["late"]
        labels, _, _, early_fine = arrays[early["key"]]
        other_labels, _, _, late_fine = arrays[late["key"]]
        if not np.array_equal(labels, other_labels):
            raise ValueError("Selected objective comparisons have different targets "
                             f"for teacher {teacher_seed}")
        comparison = {
            "teacher_seed": teacher_seed,
            "early_group": early["key"], "late_group": late["key"],
            "late_minus_early_fine_accuracy": float((late_fine - early_fine).mean()),
            "relative_rate_saving": 1 - late["actual_bpp"] / early["actual_bpp"],
            "both_assessed_mean_requirements_met": early["point_feasible"] and late["point_feasible"],
            "scope": "Two fixed validation-selected candidate groups, without an interpolated "
                     "rate frontier.",
        }
        if split == "test":
            bounds = interval(late_fine - early_fine, labels)
            comparison["fine_difference_image_ci95"] = bounds
            comparison["within_one_percentage_point"] = bounds[0] >= -0.01 and bounds[1] <= 0.01
        comparisons.append(comparison)
    report = {
        "split": split, "test_used": split == "test",
        "codec_seeds": list(CODEC_SEEDS), "readout_seeds": list(CODEC_SEEDS),
        "coarse_allowance": COARSE_ALLOWANCE, "fine_allowance": 0.05,
        "fine_allowances": list(FINE_ALLOWANCES),
        "selected_groups": frozen, "selected_groups_by_fine_allowance": selections_by_allowance,
        "interval_scope": INTERVAL_SCOPE,
        "selection_rule": SELECTION_RULE,
        "selection_source": str(selection_path) if selection_path else None,
        "selection_sha256": (hashlib.sha256(selection_path.read_bytes()).hexdigest()
                             if selection_path else None),
        "analysis_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "references": references, "rows": rows, "groups": groups, "comparisons": comparisons,
        "missing": missing,
        "unreplicated_development_groups": unreplicated,
    }
    root.mkdir(parents=True, exist_ok=True)
    output = root / f"{split}_extension_summary.json"
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if rows:
        with (root / f"{split}_extension_rows.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps({"output": str(output), "rows": len(rows), "groups": groups,
                      "missing_count": len(missing)}, indent=2))
    return output

"""Reanalysis of the retained Taskonomy fits into the compact released tables.

Nothing here fits a model or moves a requirement. The candidate pool, the input-independent
mixture schedule, the RGB decoder contrast, the retained paired contrasts, and the run summary
come from the compact pool evidence alone. The focal per-seed tables and the semantic support
tables need the per-image record tree, which is too large to release, so they are written only
when ``records_dir`` points at a copy of it.
"""

from __future__ import annotations

import csv
import hashlib
import json
import platform
from collections import defaultdict
from pathlib import Path, PurePosixPath

import numpy as np
import scipy
from scipy.optimize import linprog

from taskscope.paths import REPO_ROOT, RESULTS

TASKS = ("depth", "semantic", "edge")
FAMILIES = {"D": ("depth",), "DS": ("depth", "semantic"),
            "DE": ("depth", "edge"), "DSE": TASKS}

POOL = RESULTS / "taskonomy/pool"
REFERENCE_VALIDATION = RESULTS / "taskonomy/reference/validation_evaluation.json"
REFERENCE_TEST = RESULTS / "taskonomy/reference/test_evaluation.json"

# The retained evaluations record their per-image sources under the directory the fitting run
# wrote them to. That prefix is stripped before joining onto a supplied record tree.
RECORDED_ROOT = PurePosixPath("results/round2_task_family")

RECORD_DERIVED = ("focal_seed_metrics.json", "focal_seed_metrics.csv",
                  "focal_seed_building_metrics.json", "focal_seed_building_metrics.csv",
                  "focal_building_means.csv", "focal_means.json",
                  "semantic_support.json", "semantic_support.csv")


def resolve_recorded(records_dir: Path, recorded: str) -> Path:
    """Map a recorded source string onto a copy of the per-image record tree."""
    parts = PurePosixPath(recorded).parts
    if parts[:len(RECORDED_ROOT.parts)] == RECORDED_ROOT.parts:
        parts = parts[len(RECORDED_ROOT.parts):]
    return Path(records_dir).joinpath(*parts)


def label_for(path: Path) -> str:
    resolved = Path(path).resolve()
    return str(resolved.relative_to(REPO_ROOT)) if resolved.is_relative_to(REPO_ROOT) else str(resolved)


def failures(losses: dict, limits: dict, tasks: tuple[str, ...]) -> list[str]:
    return [task for task in tasks if losses[task] > limits[task]]


def semantic_metrics(confusion: np.ndarray, class_mask: np.ndarray) -> dict:
    support = confusion.sum(axis=1)
    union = support + confusion.sum(axis=0) - np.diag(confusion)
    iou = np.divide(np.diag(confusion), union, out=np.zeros(len(union)), where=union > 0)
    present = support > 0
    present[0] = False
    return {
        "original_present_class_miou": float(iou[present].mean()),
        "fixed_validation_class_miou": float(iou[class_mask].mean()),
        "present_object_classes": np.flatnonzero(present).tolist(),
        "ground_truth_pixel_counts": support.tolist(),
        "per_class_iou": iou.tolist(),
        "background_fraction": float(support[0] / support.sum()),
    }


def summarize_records(records: list[dict], class_mask: np.ndarray) -> dict:
    losses = {task: float(np.mean([row["losses"][task] for row in records
                                   if row["losses"].get(task) is not None])) for task in TASKS}
    for row in records:
        if row["bytes"] != sum(row[key] for key in ("main_bytes", "hyper_bytes", "header_bytes")):
            raise ValueError(f"Byte components disagree for {row['key']}")
        if not np.isclose(row["bpp"], row["bytes"] * 8 / 256**2, rtol=0, atol=1e-12):
            raise ValueError(f"Byte rate disagrees for {row['key']}")
    result = {
        "images": len(records),
        "actual_bpp": float(np.mean([row["bpp"] for row in records])),
        "losses": losses,
        "mean_bytes": {key: float(np.mean([row[key] for row in records]))
                       for key in ("bytes", "main_bytes", "hyper_bytes", "header_bytes")},
        **semantic_metrics(np.sum([row["semantic_confusion"] for row in records], axis=0), class_mask),
    }
    result["header_fraction_of_total_bytes"] = (result["mean_bytes"]["header_bytes"]
                                                / result["mean_bytes"]["bytes"])
    return result


def mixture_analysis(validation: list[dict], test: list[dict], limits: dict,
                     choices: list[dict]) -> list[dict]:
    """Cheapest input-independent schedule over the twelve base-seed candidates per requirement."""
    settings = [row["setting"] for row in validation]
    test_by_setting = {row["setting"]: row for row in test}
    if len(set(settings)) != 12 or set(settings) != set(test_by_setting):
        raise ValueError("Expected the same twelve base-seed candidates on both splits")
    ordered_test = [test_by_setting[setting] for setting in settings]
    for validation_row, test_row in zip(validation, ordered_test):
        if validation_row["selected_heads"] != test_row["selected_heads"]:
            raise ValueError(f"Changed decoder selection for {validation_row['setting']}")
    rates = np.array([row["actual_bpp"] for row in validation])
    test_rates = np.array([row["actual_bpp"] for row in ordered_test])
    stored_choices = {(str(row["alpha"]), row["requirement"]): row for row in choices}
    rows = []
    for alpha, threshold in limits.items():
        for family, tasks in FAMILIES.items():
            risks = np.array([[row["losses"][task] for row in validation] for task in tasks])
            test_risks = np.array([[row["losses"][task] for row in ordered_test] for task in tasks])
            bounds = np.array([threshold[task] for task in tasks])
            fit = linprog(rates, A_ub=risks, b_ub=bounds, A_eq=np.ones((1, len(rates))),
                          b_eq=[1], bounds=(0, None), method="highs")
            if not fit.success:
                raise RuntimeError(f"LP failed for {alpha}, {family}: {fit.message}")
            if np.max(risks @ fit.x - bounds) > 1e-7 or abs(fit.x.sum() - 1) > 1e-7:
                raise ValueError(f"Invalid LP solution for {alpha}, {family}")
            feasible = np.flatnonzero(np.all(risks <= bounds[:, None], axis=0))
            single_index = int(feasible[np.argmin(rates[feasible])])
            if settings[single_index] != stored_choices[(alpha, family)]["selected_setting"]:
                raise ValueError(f"Original selection differs for {alpha}, {family}")
            test_feasible = np.flatnonzero(np.all(test_risks <= bounds[:, None], axis=0))
            test_index = int(test_feasible[np.argmin(test_rates[test_feasible])])
            mixture_test_losses = dict(zip(tasks, (test_risks @ fit.x).tolist()))
            single_test_losses = {task: ordered_test[single_index]["losses"][task] for task in tasks}
            rows.append({
                "alpha": float(alpha), "family": family,
                "limits": {task: threshold[task] for task in tasks},
                "single_validation_setting": settings[single_index],
                "single_validation_bpp": float(rates[single_index]),
                "single_test_bpp": float(test_rates[single_index]),
                "single_test_losses": single_test_losses,
                "single_test_failed_tasks": failures(single_test_losses, threshold, tasks),
                "descriptive_test_minimum_setting": settings[test_index],
                "descriptive_test_minimum_bpp": float(test_rates[test_index]),
                "mixture_weights": {setting: float(weight)
                                    for setting, weight in zip(settings, fit.x) if weight > 1e-8},
                "mixture_validation_bpp": float(fit.fun),
                "mixture_validation_losses": dict(zip(tasks, (risks @ fit.x).tolist())),
                "mixture_test_bpp": float(test_rates @ fit.x),
                "mixture_test_losses": mixture_test_losses,
                "mixture_test_failed_tasks": failures(mixture_test_losses, threshold, tasks),
                "validation_rate_reduction_percent": float(100 * (1 - fit.fun / rates[single_index])),
            })
    return rows


class Reanalysis:
    """Collects the compact tables and the hashes of every file they were read from."""

    def __init__(self, output: Path, pool: Path, reference_validation: Path,
                 reference_test: Path, records_dir: Path | None) -> None:
        self.output = Path(output)
        self.pool = Path(pool)
        self.reference_validation = Path(reference_validation)
        self.reference_test = Path(reference_test)
        self.records_dir = Path(records_dir) if records_dir is not None else None
        self.inputs: dict[str, Path] = {}
        self.written: list[str] = []

    def read_json(self, path: Path, label: str | None = None) -> dict:
        self.inputs[label or label_for(path)] = Path(path)
        return json.loads(Path(path).read_text())

    def read_records(self, path: Path, label: str) -> list[dict]:
        self.inputs[label] = Path(path)
        return [json.loads(line) for line in Path(path).read_text().splitlines() if line]

    def record_file(self, recorded: str) -> Path:
        if self.records_dir is None:
            raise ValueError(f"Per-image record tree required for {recorded}")
        path = resolve_recorded(self.records_dir, recorded)
        if not path.exists():
            raise FileNotFoundError(f"Missing record-tree input {recorded} under {self.records_dir}")
        return path

    def write_json(self, name: str, value: object) -> None:
        (self.output / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
        self.written.append(name)

    def write_csv(self, name: str, rows: list[dict]) -> None:
        with (self.output / name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        self.written.append(name)


def run(output: Path, *, pool: Path = POOL, reference_validation: Path = REFERENCE_VALIDATION,
        reference_test: Path = REFERENCE_TEST, records_dir: Path | None = None,
        command: str = "python scripts/taskonomy_analysis.py") -> dict:
    """Write the compact tables into ``output`` and report what was written and skipped."""
    state = Reanalysis(output, pool, reference_validation, reference_test, records_dir)
    state.output.mkdir(parents=True, exist_ok=True)

    validation = state.read_json(state.pool / "full_pool_available_validation_summary.json")["rows"]
    test = state.read_json(state.pool / "full_pool_available_test_summary.json")["rows"]
    limits = state.read_json(state.pool / "development_tolerances.json")["tolerances"]
    choices = state.read_json(state.pool / "full_pool_available_validation_choices.json")
    state.write_csv("candidate_pool.csv", [{
        "split": split, "setting": row["setting"], "actual_bpp": row["actual_bpp"],
        **row["losses"], "semantic_object_miou": row["metrics"]["semantic_object_miou"],
        "source": row["source"],
    } for split, candidates in (("validation", validation), ("test", test)) for row in candidates])
    mixtures = mixture_analysis(validation, test, limits, choices["rows"])
    state.write_json("mixture_selections.json", mixtures)
    state.write_csv("mixture_selections.csv", [{
        key: ",".join(value) if isinstance(value, list) else value
        for key, value in row.items() if not isinstance(value, dict)
    } for row in mixtures])

    reference_validation_record = state.read_json(state.reference_validation)
    validation_confusion = np.array(reference_validation_record["metrics"]["semantic"]["confusion"])
    class_mask = validation_confusion.sum(axis=1) > 0
    class_mask[0] = False
    semantic_rows: list[dict] = []

    def add_semantic(name: str, split: str, path: Path, source: str) -> None:
        evaluation = state.read_json(path, source)
        metrics = semantic_metrics(np.array(evaluation["metrics"]["semantic"]["confusion"]), class_mask)
        retained = evaluation["metrics"]["semantic"]["miou_objects"]
        if not np.isclose(metrics["original_present_class_miou"], retained, atol=1e-7, rtol=0):
            raise ValueError(f"Confusion-matrix mIoU disagrees for {path}")
        semantic_rows.append({"name": name, "split": split, "source": source,
                              "retained_original_miou": retained, **metrics})

    retained_focal = state.read_json(state.pool / "focal_available_validation_selected_heads_test.json")
    retained_seeds = {row["name"]: row for row in retained_focal["per_seed_analysis"]["rows"]}
    focal_seeds: list[dict] = []
    building_seeds: list[dict] = []
    shared_keys: list[str] | None = None

    if records_dir is not None:
        add_semantic("Reference", "validation", state.reference_validation,
                     label_for(state.reference_validation))
        add_semantic("Reference", "test", state.reference_test, label_for(state.reference_test))
        for split, candidates in (("validation", validation), ("test", test)):
            for row in candidates:
                add_semantic(row["setting"], split, state.record_file(row["source"]), row["source"])
        for setting in retained_focal["manifest"]["settings"]:
            for replicate in setting["replicates"]:
                recorded = replicate["records"]
                path = state.record_file(recorded)
                records = state.read_records(path, recorded)
                keys = [row["key"] for row in records]
                if len(set(keys)) != len(keys) or (shared_keys is not None and keys != shared_keys):
                    raise ValueError(f"Focal image keys disagree for {path}")
                shared_keys = keys
                summary = summarize_records(records, class_mask)
                seed_name = f"{setting['setting']}_s{replicate['seed']}"
                expected = retained_seeds[seed_name]
                np.testing.assert_allclose(
                    [summary["actual_bpp"], *summary["losses"].values()],
                    [expected["actual_bpp"], *(expected["losses"][task] for task in TASKS)],
                    rtol=0, atol=1e-9)
                focal_seeds.append({"setting": setting["setting"], "seed": replicate["seed"],
                                    **summary})
                evaluation_recorded = str(PurePosixPath(recorded).parent / "evaluation.json")
                add_semantic(seed_name, "test", state.record_file(evaluation_recorded),
                             evaluation_recorded)
                buildings: dict[str, list[dict]] = defaultdict(list)
                for row in records:
                    buildings[row["key"].split("/")[0]].append(row)
                for building, subset in sorted(buildings.items()):
                    building_seeds.append({"setting": setting["setting"], "seed": replicate["seed"],
                                           "building": building,
                                           **summarize_records(subset, class_mask)})

        for row in focal_seeds + building_seeds:
            row["failed_tasks"] = {
                alpha: {family: failures(row["losses"], threshold, tasks)
                        for family, tasks in FAMILIES.items()}
                for alpha, threshold in limits.items()}
        state.write_json("focal_seed_metrics.json", focal_seeds)
        state.write_json("focal_seed_building_metrics.json", building_seeds)
        state.write_csv("focal_seed_building_metrics.csv", [{
            "setting": row["setting"], "seed": row["seed"], "building": row["building"],
            "images": row["images"], "actual_bpp": row["actual_bpp"], **row["losses"],
            "original_miou": row["original_present_class_miou"],
            "fixed_validation_class_miou": row["fixed_validation_class_miou"],
        } for row in building_seeds])
        building_means = []
        for setting, building in sorted({(row["setting"], row["building"]) for row in building_seeds}):
            members = [row for row in building_seeds
                       if row["setting"] == setting and row["building"] == building]
            building_means.append({
                "setting": setting, "building": building, "fits": len(members),
                "actual_bpp": float(np.mean([row["actual_bpp"] for row in members])),
                **{task: float(np.mean([row["losses"][task] for row in members])) for task in TASKS},
                "mean_original_miou": float(np.mean([row["original_present_class_miou"]
                                                     for row in members])),
                "mean_fixed_validation_class_miou": float(np.mean([row["fixed_validation_class_miou"]
                                                                   for row in members])),
            })
        state.write_csv("focal_building_means.csv", building_means)
        state.write_csv("focal_seed_metrics.csv", [{
            "setting": row["setting"], "seed": row["seed"], "actual_bpp": row["actual_bpp"],
            **row["losses"], **row["mean_bytes"],
            "original_miou": row["original_present_class_miou"],
            "fixed_validation_class_miou": row["fixed_validation_class_miou"],
        } for row in focal_seeds])

        focal_means = []
        for expected in retained_focal["rows"]:
            members = [row for row in focal_seeds if row["setting"] == expected["setting"]]
            mean_losses = {task: float(np.mean([row["losses"][task] for row in members]))
                           for task in TASKS}
            np.testing.assert_allclose(list(mean_losses.values()),
                                       [expected["losses"][task] for task in TASKS],
                                       rtol=0, atol=1e-9)
            mean_bytes = {key: float(np.mean([row["mean_bytes"][key] for row in members]))
                          for key in members[0]["mean_bytes"]}
            pass_counts = {alpha: {family: sum(not row["failed_tasks"][alpha][family]
                                               for row in members)
                                   for family in FAMILIES} for alpha in limits}
            if pass_counts != expected["feasible_seed_counts"]:
                raise ValueError(f"Focal feasibility counts disagree for {expected['setting']}")
            floor = choices["semantic_miou_floors"]["0.25"]
            focal_means.append({
                "setting": expected["setting"], "seeds": [row["seed"] for row in members],
                "actual_bpp": float(np.mean([row["actual_bpp"] for row in members])),
                "losses": mean_losses, "mean_bytes": mean_bytes,
                "header_fraction_of_total_bytes": mean_bytes["header_bytes"] / mean_bytes["bytes"],
                "mean_original_miou": float(np.mean([row["original_present_class_miou"]
                                                     for row in members])),
                "mean_fixed_validation_class_miou": float(np.mean([row["fixed_validation_class_miou"]
                                                                    for row in members])),
                "fixed_class_moderate_pass_count": sum(row["fixed_validation_class_miou"] >= floor
                                                       for row in members),
                "feasible_seed_counts": pass_counts,
                "retained_image_intervals_95": expected["loss_intervals_95"],
            })
        state.write_json("focal_means.json", focal_means)
        state.write_json("semantic_support.json", {
            "class_names": reference_validation_record["metrics"]["semantic"]["class_names"],
            "validation_object_indices": np.flatnonzero(class_mask).tolist(),
            "original_validation_floors": choices["semantic_miou_floors"], "rows": semantic_rows,
        })
        state.write_csv("semantic_support.csv", [{
            "name": row["name"], "split": row["split"],
            "original_miou": row["original_present_class_miou"],
            "fixed_validation_class_miou": row["fixed_validation_class_miou"],
            "present_object_class_count": len(row["present_object_classes"]),
        } for row in semantic_rows])

    matched = state.read_json(state.pool / "full_pool_matched_test_summary.json")["rows"]
    rgb_rows = [
        {"policy": policy, "actual_bpp": row["actual_bpp"], "losses": row["losses"],
         "failed_tasks": {alpha: failures(row["losses"], threshold, TASKS)
                          for alpha, threshold in limits.items()},
         "source": row["source"]}
        for policy, candidates in (("matched_frozen_latent_heads", matched),
                                   ("available_validation_selected_heads", test))
        for row in candidates if row["setting"] == "RGB_lambda0.3"]
    if rgb_rows[0]["actual_bpp"] != rgb_rows[1]["actual_bpp"]:
        raise ValueError("RGB decoder comparison uses different message rates")
    state.write_json("decoder_comparison.json", rgb_rows)
    state.write_json("retained_paired_contrasts.json",
                     {"interval_scope": retained_focal["interval_scope"],
                      "rows": retained_focal["paired_setting_contrasts"]})

    focal_fits = sum(len(setting["replicates"]) for setting in retained_focal["manifest"]["settings"])
    focal_buildings = sorted(retained_focal["per_seed_analysis"]["rows"][0]["buildings"])
    focal_images = retained_focal["images"]
    if records_dir is not None:
        measured = sorted({row["building"] for row in building_seeds})
        if (len(focal_seeds), len(shared_keys), measured) != (focal_fits, focal_images, focal_buildings):
            raise ValueError("Focal records disagree with the retained focal report")
    state.write_json("summary.json", {
        "scope": "Retrospective reanalysis of retained fits. No new training or threshold adjustment.",
        "mixture_cell_count": len(mixtures),
        "mixture_test_pass_count": sum(not row["mixture_test_failed_tasks"] for row in mixtures),
        "single_validation_selected_test_pass_count": sum(not row["single_test_failed_tasks"]
                                                          for row in mixtures),
        "cell_count_interpretation": "Repeated family and tolerance choices, not independent "
                                     "statistical trials.",
        "focal_fit_count": focal_fits, "focal_image_count_per_fit": focal_images,
        "focal_buildings": focal_buildings,
        "mixture_interpretation": "Input-independent schedule over full messages and fixed selected "
                                  "heads. Shared model availability and schedule assumed. No "
                                  "per-image requirement guarantee. Model storage and switching "
                                  "costs excluded.",
        "fixed_class_interpretation": "Diagnostic restriction to validation-present classes. "
                                      "Original metric and failed original requirements remain "
                                      "unchanged. The new test class is excluded only in this "
                                      "diagnostic.",
    })

    skipped = [name for name in RECORD_DERIVED if name not in state.written]
    state.inputs[label_for(Path(__file__))] = Path(__file__).resolve()
    state.write_json("provenance.json", {
        "command": command,
        "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
        "per_image_record_tree": str(records_dir) if records_dir is not None else None,
        "skipped_without_records": skipped,
        "input_sha256": {label: hashlib.sha256(path.read_bytes()).hexdigest()
                         for label, path in sorted(state.inputs.items())},
    })
    return {"output": state.output, "written": state.written, "skipped": skipped,
            "mixture_cells": len(mixtures),
            "mixture_test_passes": sum(not row["mixture_test_failed_tasks"] for row in mixtures)}

"""Recompute the reported ImageNet comparison from the sealed predictions and byte counts.

Every quantity is recomputed from the per-image prediction arrays and re-checked against
what the sealed records state: the complete byte rate against its four components, each
inherited and adapted accuracy against the recorded value, the frozen-route accuracy, and
each validation group mean against the frozen selection. A disagreement raises rather than
being drawn.

File identity is not checked by digest. The recorded hashes are provenance, and the
integrity record for the released evidence is ``results/MANIFEST.json``. What the checks
above establish is stronger in any case: the plotted numbers are recomputed from the
per-image arrays rather than read from a summary.

Intervals come from 10,000 paired resamples of 25 images within each of the 240 fine
classes. One draw is reused for the uncompressed reference, every codec group, and both
tasks, so the intervals are conditional on the fitted models and on these classes. Each
task's one-sided 97.5% upper drop limit gives approximate 95% Bonferroni coverage for that
group's two requirements.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from taskscope.paths import REPO_ROOT, RESULTS

EVIDENCE = RESULTS / "imagenet"
SELECTION = EVIDENCE / "selection_frozen/selection.json"
TEST_DIRECTORIES = ("assessment_raw", "assessment_coarse_b0.3_s17", "assessment_coarse_b0.3_s23",
                    "assessment_added_fine_b0.3_s17", "assessment_added_fine_b0.3_s23")
TASKS = ("coarse", "fine")
READOUT_SEEDS = ("17", "23")
FINE_CLASSES = 240
IMAGES_PER_CLASS = 25
BOOTSTRAP_REPETITIONS = 10000
BOOTSTRAP_SEED = 84173
VALIDATION_FINE_ALLOWANCES = (3.0, 5.0, 10.0)
PIXELS = 224**2


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def chosen_correctness(prediction: np.ndarray, adapted: dict[str, np.ndarray],
                       targets: np.ndarray, routes: dict) -> np.ndarray:
    """Return coarse/fine correctness, averaging fitted readers without ensembling."""
    if prediction.shape != targets.shape or targets.ndim != 2 or targets.shape[1] != 2:
        raise ValueError("Inherited predictions and targets must have coarse/fine columns")
    result = np.empty(targets.shape, dtype=float)
    for column, task in enumerate(TASKS):
        if routes[f"{task}_route"] == "inherited":
            result[:, column] = prediction[:, column] == targets[:, column]
        elif routes[f"{task}_route"] == "adapted_seed_mean":
            if (set(adapted) != set(READOUT_SEEDS)
                    or any(value.shape != targets.shape for value in adapted.values())):
                raise ValueError("Both fitted readers must provide fine/coarse columns")
            result[:, column] = np.mean([value[:, 1 - column] == targets[:, column]
                                         for value in adapted.values()], axis=0)
        else:
            raise ValueError(f"Unknown frozen {task} readout route")
    return result


def stratified_resamples(values: np.ndarray, classes: np.ndarray,
                         repetitions: int, seed: int) -> np.ndarray:
    """Use identical within-class image draws for every value column."""
    if repetitions < 2 or values.shape[0] != len(classes):
        raise ValueError("Resampling requires aligned images and at least two draws")
    strata = [np.flatnonzero(classes == label) for label in np.unique(classes)]
    if len({len(rows) for rows in strata}) != 1:
        raise ValueError("This balanced ImageNet analysis requires equal class sizes")
    members = np.stack(strata)
    generator = np.random.default_rng(seed)
    means = np.empty((repetitions,) + values.shape[1:])
    for begin in range(0, repetitions, 64):
        count = min(64, repetitions - begin)
        offsets = generator.integers(members.shape[1], size=(count,) + members.shape)
        indices = members[np.arange(len(members))[None, :, None], offsets].reshape(count, -1)
        means[begin:begin + count] = values[indices].mean(axis=1)
    return means


class Inputs:
    """Read the evidence for one analysis and record what was read."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.hashes: dict[str, str] = {}

    def name(self, path: Path) -> str:
        """Name a file relative to the checkout so the record travels with it."""
        try:
            return str(path.resolve().relative_to(REPO_ROOT))
        except ValueError:
            return str(path)

    def record(self, path: Path) -> None:
        self.hashes[self.name(path)] = sha256(path)

    def json(self, path: Path) -> dict:
        self.record(path)
        return json.loads(path.read_text())

    def candidate_path(self, saved: str) -> Path:
        # The frozen selection retains the path the fitting run wrote. Only its basename
        # identifies the candidate, and the retained evidence keeps that name.
        return self.root / Path(saved).name

    def read_predictions(self, directory: Path, split: str, routes: dict,
                         stored: dict, adapted_metrics: dict,
                         reference: dict | None = None) -> dict:
        folder = directory / split
        path = folder / "predictions.npz"
        self.record(path)
        seal = self.json(folder / "sealed.json")
        if seal["test_used"] != (split == "test") or seal["dataset"]["split"] != split:
            raise ValueError("Prediction split differs from its recorded membership")
        with np.load(path, allow_pickle=False) as saved:
            ids = saved["ids"]
            targets = np.column_stack((saved["coarse_target"], saved["fine_target"]))
            prediction = saved["prediction"]
            original = saved["reference"]
            components = saved["bytes"]
        if len(set(ids.tolist())) != len(ids) or len(ids) != seal["images"]:
            raise ValueError("Image identifiers must be unique and match the recorded count")
        if (components.shape != (len(ids), 4) or not np.issubdtype(components.dtype, np.integer)
                or np.any(components < 0)):
            raise ValueError("Complete rates require four nonnegative integer byte components")
        for column, name in enumerate(("main_bytes", "hyper_bytes", "header_bytes",
                                       "outer_framing_bytes")):
            if int(components[:, column].sum()) != seal[name]:
                raise ValueError(f"Recomputed {name} differs from the recorded count")
        total = int(components.sum())
        bpp = components.sum(axis=1) * (8 / PIXELS)
        if total != seal["total_bytes"] or not np.isclose(bpp.mean(), seal["actual_bpp"],
                                                          atol=1e-12, rtol=0):
            raise ValueError("Recomputed complete rate differs from the recorded rate")
        # The serialized packet stream is external. Check it whenever it is present.
        packet_path = folder / "messages.bin"
        if packet_path.exists():
            self.record(packet_path)
            if packet_path.stat().st_size != total:
                raise ValueError("Serialized file differs from the complete packet byte count")
        adapted = {}
        for seed in READOUT_SEEDS:
            path = (directory / f"readout_s{seed}_validation.npy" if split == "validation"
                    else folder / f"readout_s{seed}_predictions.npy")
            self.record(path)
            adapted[seed] = np.load(path, allow_pickle=False)
            if adapted[seed].shape != targets.shape:
                raise ValueError("Adapted predictions do not match the image count")
            for column, task in enumerate(TASKS):
                actual = float((adapted[seed][:, 1 - column] == targets[:, column]).mean())
                if not np.isclose(actual, adapted_metrics[seed][f"{task}_accuracy"],
                                  atol=1e-12, rtol=0):
                    raise ValueError("Recomputed adapted accuracy differs from the recorded value")
        correct = chosen_correctness(prediction, adapted, targets, routes)
        for column, task in enumerate(TASKS):
            if not np.isclose((prediction[:, column] == targets[:, column]).mean(),
                              seal[f"{task}_accuracy"], atol=1e-12, rtol=0):
                raise ValueError("Recomputed inherited accuracy differs from the recorded value")
            if not np.isclose(correct[:, column].mean(), stored[f"{task}_accuracy"],
                              atol=1e-12, rtol=0):
                raise ValueError("Frozen-route accuracy differs from the recorded value")
        if reference is not None:
            lookup = {name: index for index, name in enumerate(ids.tolist())}
            if set(lookup) != set(reference["ids"].tolist()):
                raise ValueError("Raw and coded image memberships differ")
            order = np.array([lookup[name] for name in reference["ids"].tolist()])
            ids, targets = ids[order], targets[order]
            correct, bpp, original = correct[order], bpp[order], original[order]
            if (not np.array_equal(targets, reference["targets"])
                    or not np.array_equal(original, reference["original"])):
                raise ValueError("Raw and coded targets or original teacher predictions differ")
        return {"ids": ids, "targets": targets, "correct": correct, "bpp": bpp,
                "original": original, "seal": seal,
                "packet_file_checked": packet_path.exists()}


def group_key(group: dict) -> str:
    return f"{group['objective']}_b{group['beta']:g}"


def analyze(selection_path: Path, root: Path, test_directories: list[Path],
            repetitions: int = BOOTSTRAP_REPETITIONS, seed: int = BOOTSTRAP_SEED) -> dict:
    """Recheck the frozen selection on validation and assess it on the held-out split."""
    inputs = Inputs(root)
    selection = inputs.json(selection_path)
    if selection["test_used"]:
        raise ValueError("Selection must precede held-out assessment")
    reference_metrics = selection["reference_metrics"]
    reference_directory = inputs.candidate_path(selection["reference"])
    reference_readouts = inputs.json(reference_directory / "readouts.json")["readouts"]
    validation_reference = inputs.read_predictions(reference_directory, "validation",
                                                   reference_metrics, reference_metrics,
                                                   reference_readouts)
    if validation_reference["seal"] != reference_metrics["seal"]:
        raise ValueError("Raw validation result changed after selection")
    candidates = {row["directory"]: row for row in selection["candidates"]}
    validation = {}
    for saved, row in candidates.items():
        directory = inputs.candidate_path(saved)
        fitted = inputs.json(directory / "readouts.json")["readouts"]
        data = inputs.read_predictions(directory, "validation", row, row, fitted,
                                       validation_reference)
        validation[saved] = data
    selected_for: dict[str, list[str]] = {}
    for category in ("selected", "pooled_selected"):
        for name, group in selection[category].items():
            if group is not None:
                selected_for.setdefault(group_key(group), []).append(f"{category}.{name}")
    groups = {group_key(group): group for group in selection["groups"]}
    selected_directories = {directory for key in selected_for
                            for directory in groups[key]["directories"]}
    summaries = [inputs.json(directory / "test_summary.json") for directory in test_directories]
    for summary in summaries:
        if not summary["test_used"]:
            raise ValueError("Assessment must follow the frozen selection")
    raw_indices = [index for index, summary in enumerate(summaries)
                   if summary["inherited"]["raw_reference"]]
    if len(raw_indices) != 1:
        raise ValueError("Exactly one held-out raw reference is required")
    raw_index = raw_indices[0]

    def read_test(index: int, frozen: dict, reference: dict | None) -> dict:
        summary = summaries[index]
        if any(summary[f"{task}_route"] != frozen[f"{task}_route"] for task in TASKS):
            raise ValueError("Assessment readout route differs from the frozen route")
        stored = {f"{task}_accuracy": summary[f"primary_{task}_accuracy"] for task in TASKS}
        data = inputs.read_predictions(test_directories[index], "test", frozen, stored,
                                       summary["adapted"], reference)
        if data["seal"] != summary["inherited"]:
            raise ValueError("Assessment summary and inherited result disagree")
        return data

    test_reference = read_test(raw_index, reference_metrics, None)
    labels, counts = np.unique(test_reference["targets"][:, 1], return_counts=True)
    if len(labels) != FINE_CLASSES or not np.all(counts == IMAGES_PER_CLASS):
        raise ValueError("Final assessment requires 240 fine classes with 25 images each")
    tested = {}
    for index, summary in enumerate(summaries):
        if index == raw_index:
            continue
        matches = [saved for saved in selected_directories
                   if candidates[saved]["checkpoint_sha256"]
                   == summary["inherited"]["checkpoint_sha256"]]
        if len(matches) != 1 or matches[0] in tested:
            raise ValueError("Assessment checkpoint is duplicated or was not uniquely selected")
        tested[matches[0]] = read_test(index, candidates[matches[0]], test_reference)
    if set(tested) != selected_directories:
        raise ValueError("Every frozen-selected codec fit must have a held-out assessment")
    records, sample_values = [], []
    for key, group in groups.items():
        paths = group["directories"]
        if sorted(candidates[path]["seed"] for path in paths) != [17, 23]:
            raise ValueError("Each group requires both task-fitting seeds")
        val_accuracy = np.mean([validation[path]["correct"].mean(0) for path in paths], axis=0)
        val_bpp = float(np.mean([validation[path]["bpp"].mean() for path in paths]))
        for actual, metric in zip([*val_accuracy, val_bpp],
                                  ("coarse_accuracy", "fine_accuracy", "actual_bpp")):
            if not np.isclose(actual, group[metric], atol=1e-12, rtol=0):
                raise ValueError("Recomputed validation group differs from the frozen group")
        drop = 100 * (validation_reference["correct"].mean(0) - val_accuracy)
        record = {"key": key, "objective": group["objective"], "beta": group["beta"],
                  "selected_for": selected_for.get(key, []),
                  "validation": {"accuracy": val_accuracy.tolist(), "actual_bpp": val_bpp,
                                 "drop_pp": drop.tolist(),
                                 "fits": [{"seed": candidates[path]["seed"],
                                           "accuracy": validation[path]["correct"].mean(0).tolist(),
                                           "actual_bpp": float(validation[path]["bpp"].mean())}
                                          for path in paths]}}
        if key in selected_for:
            fits = [{"seed": candidates[path]["seed"],
                     "checkpoint_sha256": candidates[path]["checkpoint_sha256"],
                     "accuracy": tested[path]["correct"].mean(0).tolist(),
                     "drop_pp": (100 * (test_reference["correct"]
                                        - tested[path]["correct"]).mean(0)).tolist(),
                     "actual_bpp": float(tested[path]["bpp"].mean()),
                     "packet_file_checked": tested[path]["packet_file_checked"]}
                    for path in paths]
            accuracy = np.mean([tested[path]["correct"] for path in paths], axis=0)
            test_drop = 100 * (test_reference["correct"] - accuracy)
            rate = np.mean([tested[path]["bpp"] for path in paths], axis=0)
            sample_values.append(np.column_stack((test_drop, rate)))
            record["test"] = {
                "accuracy": accuracy.mean(0).tolist(), "drop_pp": test_drop.mean(0).tolist(),
                "actual_bpp": float(rate.mean()), "fits": fits,
                "fit_drop_range_pp": np.ptp([fit["drop_pp"] for fit in fits], axis=0).tolist(),
                "fit_bpp_range": float(np.ptp([fit["actual_bpp"] for fit in fits]))}
        records.append(record)
    draws = (stratified_resamples(np.stack(sample_values, axis=1),
                                  test_reference["targets"][:, 1], repetitions, seed)
             if sample_values else np.empty((repetitions, 0, 3)))
    selected_records = [record for record in records if "test" in record]
    allowance = 100 * np.array([selection["coarse_drop"], selection["fine_drop"]])
    for index, record in enumerate(selected_records):
        lower, upper = np.quantile(draws[:, index], [0.025, 0.975], axis=0)
        record["test"].update(
            drop_pointwise_95_interval_pp=np.column_stack((lower[:2], upper[:2])).tolist(),
            drop_one_sided_97_5_upper_pp=upper[:2].tolist(),
            joint_95_allowance_supported=bool(np.all(upper[:2] <= allowance)),
            actual_bpp_pointwise_95_interval=[float(lower[2]), float(upper[2])])
    comparisons = []
    for first in range(len(selected_records)):
        for second in range(first + 1, len(selected_records)):
            contrast = draws[:, second] - draws[:, first]
            comparisons.append({
                "second_minus_first": [selected_records[second]["key"],
                                       selected_records[first]["key"]],
                "quantities": ["coarse_drop_pp", "fine_drop_pp", "actual_bpp"],
                "difference": (sample_values[second].mean(0)
                               - sample_values[first].mean(0)).tolist(),
                "pointwise_95_interval": np.quantile(contrast, [0.025, 0.975], axis=0).T.tolist()})
    sensitivity = []
    for fine_allowance in VALIDATION_FINE_ALLOWANCES:
        feasible = [record for record in records
                    if record["validation"]["drop_pp"][0] <= allowance[0] + 1e-12
                    and record["validation"]["drop_pp"][1] <= fine_allowance + 1e-12]
        best = min(feasible, key=lambda record: record["validation"]["actual_bpp"]) if feasible else None
        sensitivity.append({"coarse_allowance_pp": float(allowance[0]),
                            "fine_allowance_pp": fine_allowance,
                            "pooled_validation_minimum": best["key"] if best else None,
                            "actual_bpp": best["validation"]["actual_bpp"] if best else None})
    inputs.record(Path(__file__))
    return {"task_order": list(TASKS), "allowance_pp": allowance.tolist(), "groups": records,
            "validation_reference_accuracy": validation_reference["correct"].mean(0).tolist(),
            "test_reference_accuracy": test_reference["correct"].mean(0).tolist(),
            "paired_group_comparisons": comparisons,
            "validation_allowance_sensitivity": sensitivity,
            "frozen_selections": {category: {name: group_key(group) if group else None
                                             for name, group in selection[category].items()}
                                  for category in ("selected", "pooled_selected")},
            "statistics": {
                "images": len(test_reference["ids"]), "fine_classes": FINE_CLASSES,
                "images_per_class": IMAGES_PER_CLASS, "bootstrap_repetitions": repetitions,
                "bootstrap_seed": seed,
                "sampling": "Resample 25 images within each fine class with replacement. Reuse "
                            "each draw for the raw reference, every codec group and both tasks.",
                "fit_averaging": "Average correctness across the two fitted readers when that "
                                 "route was selected, then across the two task-fitting seeds. No "
                                 "prediction ensemble and no resampling of fitted models.",
                "coverage": "Percentile intervals are nominal pointwise 95%, conditional on the "
                            "fitted models and these 240 classes. Each task's one-sided 97.5% "
                            "upper bound gives a Bonferroni nominal joint 95% assessment for that "
                            "group's two requirements. This is not simultaneous coverage over all "
                            "groups.",
                "rate": "All four transmitted byte components divided by 224 squared pixels. The "
                        "raw-reference zero is a sentinel and is excluded from the rate plot."},
            "selection_sha256": sha256(selection_path), "input_sha256": inputs.hashes}

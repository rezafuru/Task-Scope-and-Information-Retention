#!/usr/bin/env python3
"""Render the Taskonomy requirement comparison over the twelve matched test candidates.

Each panel places one task's normalized test risk against the complete rate. The vertical
coordinate is (test risk - validation reference risk) / (validation constant risk - validation
reference risk), so zero is the fixed validation reference, one is the fixed validation
constant prediction, and a point at or below a dashed line meets that component's requirement.
Open markers repeat the RGB candidate's message through reconstructed RGB and the frozen
reference predictor for depth and semantics, keeping the latent edge decoder.

Plotted values and intervals come from the compact retained summaries. ``--records-dir``
re-enables the per-image audit, which recomputes every byte sum, rate, and risk mean from the
per-image record tree that the release omits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path, PurePosixPath

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

from taskscope.figures import (
    INK,
    OCHRE,
    PLUM,
    SLATE,
    TEAL,
    configure_style,
    crop_to_ink,
    render_preview,
    save_figure,
    style_axis,
)
from taskscope.paths import FIGURES, REPO_ROOT, RESULTS
from taskscope.taskonomy.analysis import resolve_recorded

POOL = RESULTS / "taskonomy/pool"
MATCHED = POOL / "full_pool_matched_test_summary.json"
PAIRED = POOL / "full_pool_matched_test_paired.json"
TOLERANCES = POOL / "development_tolerances.json"
AVAILABLE = POOL / "full_pool_available_test_summary.json"
REFERENCE = RESULTS / "taskonomy/reference/validation_evaluation.json"

TASKS = ("depth", "semantic", "edge")
FAMILIES = {
    "D": (SLATE, "o"),
    "DS": (TEAL, "s"),
    "DE": (PLUM, "^"),
    "DSE": (OCHRE, "D"),
    "RGB": (INK, "P"),
}


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def recorded_key(value: str) -> str:
    """Normalize a recorded source string so identities compare independently of the layout."""
    return PurePosixPath(value).as_posix()


def finite(value: float, description: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"Non-finite {description}")
    return number


def check_close(actual: float, expected: float, description: str, tolerance: float = 1e-10) -> None:
    if not math.isfinite(actual) or not math.isclose(actual, expected, rel_tol=0, abs_tol=tolerance):
        raise ValueError(f"{description}: {actual!r} differs from {expected!r}")


def prepare_points(candidates: Path, tolerances: Path, paired: Path | None, reference_path: Path,
                   profile_filter: str | None, width_filter: int | None) -> tuple:
    """Place every measured candidate on the frozen validation risk scale."""
    rows = load(candidates)["rows"]
    rows = [row for row in rows
            if (profile_filter is None or row["readout_profile"] == profile_filter)
            and (width_filter is None or row["readout_head_channels"] == width_filter)]
    if not rows:
        raise ValueError("The candidate summary contains no measured rows")
    comparison = {(row["split"], row["images"], row["readout_profile"],
                   row["readout_head_channels"]) for row in rows}
    if len(comparison) != 1:
        raise ValueError("Use one split, image set size, readout profile and width per figure")
    split, images, profile, width = comparison.pop()
    if split not in {"val", "test"}:
        raise ValueError(f"Unsupported split {split}")
    identities = [(row["setting"], row["encoder_seed"], row["readout_seed"]) for row in rows]
    if len(set(identities)) != len(identities):
        raise ValueError("Duplicate candidate setting and seed pair")

    requirements = load(tolerances)
    reference = load(reference_path)
    alphas = sorted(float(value) for value in requirements["tolerances"])
    if alphas != [0.1, 0.25, 0.5]:
        raise ValueError("Expected the three frozen alpha requirements")
    gaps = {task: reference["constant_losses"][task] - reference["losses"][task] for task in TASKS}
    for alpha, thresholds in requirements["tolerances"].items():
        for task in TASKS:
            expected = reference["losses"][task] + float(alpha) * gaps[task]
            if gaps[task] <= 0 or not math.isclose(expected, thresholds[task], abs_tol=1e-10):
                raise ValueError(f"Invalid frozen reference gap or threshold for {task}")

    intervals = {}
    if paired is not None:
        report = load(paired)
        named = recorded_key(str(PurePosixPath(report["tolerances_source"]).parent
                                 / requirements["reference"]))
        if named != recorded_key(report["validation_reference_source"]):
            raise ValueError("The paired report uses a different validation reference")
        if report["images"] != images:
            raise ValueError("The paired report and candidate summary use different image counts")
        for row in report["rows"]:
            if row["name"] == "reference":
                continue
            key = recorded_key(report["sources"][row["name"]])
            if key in intervals:
                raise ValueError("The paired report repeats a candidate source")
            intervals[key] = row

    points = []
    for row in rows:
        family = row["preservation_family"]
        if family not in FAMILIES:
            raise ValueError(f"Unknown preservation family {family}")
        rate = finite(row["actual_bpp"], "actual rate")
        if rate <= 0:
            raise ValueError("Actual rates must be positive for the logarithmic display")
        risk = {task: finite((row["losses"][task] - reference["losses"][task]) / gaps[task],
                             f"{row['setting']} {task} risk") for task in TASKS}
        bounds = {}
        if paired is not None:
            measured = intervals[recorded_key(row["per_image_source"])]
            if not math.isclose(rate, measured["actual_bpp"], rel_tol=1e-6, abs_tol=1e-9):
                raise ValueError(f"Rate mismatch for {row['setting']}")
            for task in TASKS:
                record = measured["fixed_validation_excess"][task]
                estimate = finite(record["estimate"], f"{task} interval estimate")
                if not math.isclose(estimate, risk[task], rel_tol=1e-5, abs_tol=5e-6):
                    raise ValueError(f"Risk mismatch for {row['setting']} {task}")
                lower = finite(record["interval_95"]["lower"], f"{task} lower interval")
                upper = finite(record["interval_95"]["upper"], f"{task} upper interval")
                if lower > upper:
                    raise ValueError(f"Reversed interval for {row['setting']} {task}")
                risk[task] = estimate
                bounds[task] = (lower, upper)
        points.append({"family": family, "rate": rate, "risk": risk, "bounds": bounds,
                       "encoder_seed": row["encoder_seed"], "readout_seed": row["readout_seed"]})
    return points, alphas, (split, images, profile, width)


def audit_records(row: dict, common_keys: set[str] | None, records_dir: Path) -> tuple[set[str], Path]:
    """Check measured rates and risks against all retained per-image records."""
    path = resolve_recorded(records_dir, row["per_image_source"])
    if not path.exists():
        raise FileNotFoundError(f"Missing per-image records {row['per_image_source']} "
                                f"under {records_dir}")
    with path.open() as stream:
        records = [json.loads(line) for line in stream]
    keys = {record["key"] for record in records}
    buildings = Counter(record["key"].split("/")[0] for record in records)
    if len(records) != 2000 or len(keys) != 2000 or sorted(buildings.values()) != [400] * 5:
        raise ValueError(f"Unexpected image keys or building counts in {path}")
    if common_keys is not None and keys != common_keys:
        raise ValueError(f"Different test images in {path}")
    for record in records:
        total = sum(record[field] for field in ("main_bytes", "hyper_bytes", "header_bytes"))
        if total != record["bytes"] or record["header_bytes"] != 40:
            raise ValueError(f"Incomplete transmitted length in {path}")
        check_close(record["bpp"], 8 * total / 256**2, f"Per-image rate in {path}")
    check_close(math.fsum(record["bpp"] for record in records) / len(records),
                row["actual_bpp"], f"Mean complete rate in {path}")
    for task in TASKS:
        mean = math.fsum(record["losses"][task] for record in records) / len(records)
        check_close(mean, row["losses"][task], f"Mean {task} risk in {path}")
    return keys, path


def message_lengths(row: dict, records_dir: Path) -> dict:
    path = resolve_recorded(records_dir, row["per_image_source"])
    with path.open() as stream:
        records = [json.loads(line) for line in stream]
    return {record["key"]: tuple(record[field] for field in
                                 ("main_bytes", "hyper_bytes", "header_bytes")) for record in records}


def prepare_evidence(records_dir: Path | None) -> tuple[list[dict], dict, dict, dict]:
    points, alphas, comparison = prepare_points(MATCHED, TOLERANCES, PAIRED, REFERENCE,
                                                "matched_frozen", 48)
    rows = load(MATCHED)["rows"]
    if len(points) != 12 or len(rows) != 12 or comparison != ("test", 2000, "matched_frozen", 48):
        raise ValueError("Expected the original twelve matched test candidates")
    if any(row["encoder_seed"] != 20260905 or row["readout_seed"] != 20260905 for row in rows):
        raise ValueError("The matched comparison requires the original base seed")
    reference = load(REFERENCE)
    report = load(PAIRED)
    if report["bootstrap_repetitions"] != 2000 or report["resampling_seed"] != 20260906:
        raise ValueError("The retained conditional interval procedure differs")
    if alphas != [0.1, 0.25, 0.5]:
        raise ValueError("The original risk requirements differ")
    sources = {path: path for path in (MATCHED, PAIRED, TOLERANCES, AVAILABLE, REFERENCE,
                                       Path(__file__).resolve())}
    audited: list[str] = []
    common_keys = None
    for point, row in zip(points, rows):
        point["setting"] = row["setting"]
        if records_dir is not None:
            common_keys, path = audit_records(row, common_keys, records_dir)
            sources[path] = path
            audited.append(row["per_image_source"])

    alternate_rows = [row for row in load(AVAILABLE)["rows"] if row["setting"] == "RGB_lambda0.3"]
    if len(alternate_rows) != 1:
        raise ValueError("Expected one available-head RGB comparison")
    alternate = alternate_rows[0]
    matched_rgb = next(row for row in rows if row["setting"] == alternate["setting"])
    if alternate["encoder_checkpoint"] != matched_rgb["encoder_checkpoint"]:
        raise ValueError("The alternative decoder uses a different encoder")
    check_close(alternate["actual_bpp"], matched_rgb["actual_bpp"], "Same-message rate")
    if any(alternate["selected_heads"][task]["kind"] != "rgb_reference" for task in TASKS[:2]):
        raise ValueError("Depth and semantics must use the frozen RGB-reference route")
    edge_head = alternate["selected_heads"]["edge"]
    if edge_head["kind"] != "latent" or edge_head["checkpoint"] != matched_rgb["readout_checkpoint"]:
        raise ValueError("The alternative route must retain the matched latent edge head")
    check_close(alternate["losses"]["edge"], matched_rgb["losses"]["edge"], "Retained edge risk")
    if records_dir is not None:
        _, path = audit_records(alternate, common_keys, records_dir)
        sources[path] = path
        audited.append(alternate["per_image_source"])
        if message_lengths(matched_rgb, records_dir) != message_lengths(alternate, records_dir):
            raise ValueError("The RGB decoder comparisons have different per-image message lengths")

    alternate_point = {"family": "RGB", "rate": alternate["actual_bpp"], "risk": {}, "bounds": {}}
    for task in TASKS:
        baseline = reference["losses"][task]
        gap = reference["constant_losses"][task] - baseline
        alternate_point["risk"][task] = (alternate["losses"][task] - baseline) / gap
        interval = alternate["loss_intervals_95"][task]
        if interval["defined_replicates"] != 2000:
            raise ValueError("An alternative-decoder interval has missing replicates")
        alternate_point["bounds"][task] = tuple((interval[end] - baseline) / gap
                                                for end in ("lower", "upper"))
        check_close(alternate_point["risk"][task], alternate["fixed_validation_excess"][task],
                    f"Alternative {task} coordinate")
    provenance = {"record_audit_ran": records_dir is not None,
                  "record_tree": str(records_dir) if records_dir is not None else None,
                  "audited_per_image_sources": audited,
                  "sources": sorted(sources)}
    return points, alternate_point, reference, provenance


def draw(points: list[dict], alternate: dict) -> plt.Figure:
    configure_style()
    figure, axes = plt.subplots(1, 3, figsize=(6.5, 2.0), sharex=True, sharey=True)
    for axis, task, title in zip(axes, TASKS, ("(a) Depth", "(b) Semantics", "(c) Edges")):
        for threshold in (0.1, 0.25, 0.5):
            axis.axhline(threshold, color="#8A8A8A", linewidth=0.65,
                         linestyle=(0, (3, 3)), zorder=0)
        axis.axhline(0, color="#CBCBCB", linewidth=0.6, zorder=0)
        for point in points:
            color, marker = FAMILIES[point["family"]]
            axis.vlines(point["rate"], *point["bounds"][task], color=color, linewidth=0.9, zorder=2)
            axis.plot(point["rate"], point["risk"][task], marker=marker, color=color,
                      markersize=4.5, markeredgewidth=0.7, linestyle="none", zorder=3)
        if task != "edge":
            original = next(point for point in points if point["family"] == "RGB")
            axis.annotate("", xy=(alternate["rate"], alternate["risk"][task]),
                          xytext=(original["rate"], original["risk"][task]),
                          arrowprops={"arrowstyle": "->", "color": "#444444", "lw": 0.85,
                                      "shrinkA": 3.5, "shrinkB": 3.5}, zorder=2)
            axis.vlines(alternate["rate"], *alternate["bounds"][task], color="#444444",
                        linewidth=0.9, zorder=4)
            axis.plot(alternate["rate"], alternate["risk"][task], marker="P", color="#444444",
                      markerfacecolor="white", markersize=6, markeredgewidth=0.9,
                      linestyle="none", zorder=5)
            if task == "depth":
                axis.text(original["rate"], 0.39, "Same bits", ha="center", va="bottom",
                          fontsize=8, color="#444444")
        axis.set(xscale="log", xlim=(0.0059, 0.24), ylim=(-0.13, 0.96), title=title)
        axis.xaxis.set_major_locator(FixedLocator([0.01, 0.02, 0.05, 0.1, 0.2]))
        axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
        axis.xaxis.set_minor_locator(NullLocator())
        axis.set_yticks([0, 0.25, 0.5, 0.75])
        axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
        axis.set_title(title, fontsize=8.5, pad=4)
        style_axis(axis, grid=None)
    axes[0].set_ylabel("Normalized test risk", labelpad=4)
    for threshold in (0.1, 0.25, 0.5):
        axes[-1].text(1.02, threshold, rf"$\alpha={threshold:g}$",
                      transform=axes[-1].get_yaxis_transform(), fontsize=8,
                      color="#555555", va="center")
    handles = [Line2D([], [], color=color, marker=marker, markersize=4.5,
                      linestyle="none", label=family) for family, (color, marker) in FAMILIES.items()]
    handles.append(Line2D([], [], color="#444444", marker="P", markerfacecolor="white",
                          markersize=6, markeredgewidth=0.9, linestyle="none",
                          label="RGB + reference"))
    figure.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.0),
                  ncol=6, frameon=False, fontsize=8, handletextpad=0.4, columnspacing=1.4)
    figure.text(0.5, 0.04, "Complete rate (bpp)", ha="center", va="bottom", fontsize=8)
    figure.subplots_adjust(left=0.082, right=0.907, bottom=0.22, top=0.79, wspace=0.16)
    return figure


def relative_name(path: Path) -> str:
    """Name a path relative to the checkout so the record travels with it."""
    try:
        return str(Path(path).resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def write_provenance(points: list[dict], alternate: dict, reference: dict, provenance: dict,
                     figure_path: Path, path: Path) -> None:
    """Record the plotted numbers, the reference scale, and whether the record audit ran."""
    sources = [Path(item) for item in provenance.pop("sources")]
    record = {
        "figure": relative_name(figure_path),
        **provenance,
        "normalization": "(test risk - validation reference risk) / (validation constant risk - "
                         "validation reference risk)",
        "interval_scope": "conditional 95% percentile intervals from 2000 paired image resamples "
                          "within the five observed buildings, componentwise",
        "validation_reference": {task: {"reference": reference["losses"][task],
                                        "constant": reference["constant_losses"][task]}
                                 for task in TASKS},
        "candidates": [{"setting": point["setting"], "family": point["family"],
                        "rate_bpp": point["rate"],
                        "risk": {task: point["risk"][task] for task in TASKS}}
                       for point in sorted(points, key=lambda item: (item["family"], item["rate"]))],
        "rgb_via_reference": {"rate_bpp": alternate["rate"],
                              "risk": {task: alternate["risk"][task] for task in TASKS}},
        "source_sha256": {source.name: hashlib.sha256(source.read_bytes()).hexdigest()
                          for source in sources},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=FIGURES,
                        help="directory the figure is written to")
    parser.add_argument("--records-dir", type=Path,
                        help="copy of the per-image record tree, which re-enables the audit")
    arguments = parser.parse_args()
    points, alternate, reference, provenance = prepare_evidence(arguments.records_dir)
    arguments.output.mkdir(parents=True, exist_ok=True)
    base = arguments.output / "paper_taskonomy_requirements"
    save_figure(draw(points, alternate), base)
    crop_to_ink(base.with_suffix(".pdf"), dpi=288, guard=0.3, threshold=250)
    render_preview(base.with_suffix(".pdf"), base.with_suffix(".png"), dpi=200)
    write_provenance(points, alternate, reference, provenance, base.with_suffix(".pdf"),
                     base.with_suffix(".json"))
    audit = "with" if arguments.records_dir else "without"
    print(f"Validated 12 matched candidates and one RGB decoder comparison {audit} the per-image "
          f"record audit. Saved {base.with_suffix('.pdf')}.")


if __name__ == "__main__":
    main()

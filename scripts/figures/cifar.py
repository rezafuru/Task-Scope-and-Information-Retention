#!/usr/bin/env python3
"""Render the CIFAR-100 local-preservation figures from the frozen evidence.

The test figure places the nine assessed candidate groups against the coarse and
fine accuracy allowances and shows three readout routes decoded from the same
logit message. The validation figure shows every fitted codec, the paired-codec
group means, and the rate-minimal selection that was frozen before the test
assessment.

Both figures read only the two stored evidence files and recheck their
consistency (the selection hash binding, the complete byte rate identity, and
the paired-codec group means) before drawing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from taskscope.figures import FULL_COLOR, REDUCED_COLOR, crop_to_ink, render_preview, write_pdf
from taskscope.paths import FIGURES, REPO_ROOT, RESULTS

EVIDENCE = RESULTS / "cifar"
COLORS = {"early": FULL_COLOR, "late": REDUCED_COLOR}
READOUTS = (
    ("logit_fine_accuracy", "Refitted\nlogits", "s"),
    ("fixed_fine_accuracy", "Inherited\ntensor", "o"),
    ("fine_accuracy", "Refitted\ntensor", "D"),
)
TEACHERS = (17, 23, 29)
TEACHER_LABELS = {17: "A", 23: "B", 29: "C"}


def configure_style() -> None:
    plt.rcParams.update({
        "font.size": 8.5, "axes.labelsize": 8.5, "axes.titlesize": 8.5,
        "xtick.labelsize": 8.2, "ytick.labelsize": 8.2,
        "legend.fontsize": 8.2, "pdf.fonttype": 42,
        "axes.linewidth": 0.65,
    })


def read_evidence(source: Path, split: str) -> dict:
    """Load one aggregated split and recheck the rate identity it reports."""
    data = json.loads(source.read_text())
    if data["split"] != split or data["test_used"] is not (split == "test"):
        raise ValueError(f"Expected {split} evidence in {source}")
    if sorted({row["teacher_seed"] for row in data["rows"]}) != list(TEACHERS):
        raise ValueError("All three teachers must be present")
    for row in data["rows"]:
        values = [row[key] for key in ("actual_bpp", "coarse_drop", "fine_drop")]
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite measurements for {row['run']}")
        byte_rate = 8 * sum(row[key] for key in (
            "mean_header_bytes", "mean_hyper_bytes", "mean_main_bytes",
        )) / (32 * 32)
        if not np.isclose(byte_rate, row["actual_bpp"], rtol=0, atol=1e-12):
            raise ValueError(f"Complete byte rate disagrees for {row['run']}")
    return data


def group_rows(data: dict, group: dict) -> list[dict]:
    """Return the two codec fits behind one group after rechecking their means."""
    rows = [row for row in data["rows"] if (
        row["teacher_seed"] == group["teacher_seed"]
        and row["run"] in group["runs"]
    )]
    if len(rows) != 2 or sorted(row["codec_seed"] for row in rows) != [17, 23]:
        raise ValueError(f"Expected two codec fits for {group['key']}")
    for key in ("actual_bpp", "coarse_drop", "fine_drop", "fine_accuracy",
                "fixed_fine_accuracy", "logit_fine_accuracy"):
        if not np.isclose(np.mean([row[key] for row in rows]), group[key],
                          rtol=0, atol=1e-12):
            raise ValueError(f"Group mean disagrees for {group['key']}: {key}")
    return sorted(rows, key=lambda row: row["codec_seed"])


def save_figure(figure: Figure, base: Path, provenance: dict) -> None:
    """Write the cropped vector figure, its review rasters, and the provenance record."""
    path = write_pdf(figure, base, pad_inches=1 / 72)
    provenance["pdf_size_points"] = crop_to_ink(path, dpi=600, guard=0.3, threshold=255)
    provenance["crop"] = "Rendered ink at 600 dpi, with at most 0.3 point guard. PDF remains vector."
    render_preview(path, base.with_suffix(".png"), dpi=220)
    render_preview(path, base.with_name(base.name + "_manuscript_size.png"), dpi=100)
    base.with_suffix(".json").write_text(json.dumps(provenance, indent=2) + "\n")


def source_record(source: Path) -> dict:
    """Name the evidence relative to the checkout so the record travels with it."""
    try:
        name = str(source.resolve().relative_to(REPO_ROOT))
    except ValueError:
        name = str(source)
    return {"source": name, "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}


def plot_test(source: Path, validation_source: Path, output: Path) -> None:
    data = read_evidence(source, "test")
    validation = read_evidence(validation_source, "validation")
    validation_hash = hashlib.sha256(validation_source.read_bytes()).hexdigest()
    if data["selection_sha256"] != validation_hash:
        raise ValueError("Test assessment does not refer to this frozen validation file")
    if data["selected_groups"] != validation["selected_groups"]:
        raise ValueError("Test selections differ from frozen validation selections")
    if len(data["groups"]) != 9 or len(data["rows"]) != 18:
        raise ValueError("Expected all nine assessed groups and eighteen codec fits")
    ordered = []
    for teacher in TEACHERS:
        teacher_groups = [group for group in data["groups"]
                          if group["teacher_seed"] == teacher]
        for objective, selected in (("early", True), ("early", False), ("late", True)):
            matches = [group for group in teacher_groups
                       if group["objective"] == objective
                       and group["selected_on_validation"] is selected]
            if len(matches) != 1:
                raise ValueError(f"Ambiguous assessed row for teacher {teacher}")
            ordered.append(matches[0])
    for group in ordered:
        group_rows(data, group)
        if group["selected_on_validation"] != (group["key"] in validation["selected_groups"]):
            raise ValueError(f"Incorrect selection status for {group['key']}")
    configure_style()
    figure = plt.figure(figsize=(6.5, 2.60))
    # The text columns share the loss panels' row positions.
    band_ax = figure.add_axes((0.004, 0.19, 0.992, 0.66), zorder=0)
    label_ax = figure.add_axes((0.004, 0.19, 0.296, 0.66), facecolor="none")
    coarse_ax = figure.add_axes((0.325, 0.19, 0.175, 0.66), facecolor="none")
    fine_ax = figure.add_axes((0.527, 0.19, 0.175, 0.66), facecolor="none")
    readout_ax = figure.add_axes((0.729, 0.19, 0.267, 0.66), facecolor="none")
    positions = np.array([0, 0.83, 1.66, 2.85, 3.68, 4.51, 5.70, 6.53, 7.36])
    y_limits = (7.82, -0.47)
    band_ax.set_ylim(*y_limits)
    band_ax.axis("off")
    for position in positions[2::3]:
        band_ax.axhspan(position - 0.38, position + 0.38,
                        color=COLORS["late"], alpha=0.065, linewidth=0)
    for boundary in (2.25, 5.10):
        band_ax.axhline(boundary, color="0.87", lw=0.6)
    for axis in (label_ax, coarse_ax, fine_ax, readout_ax):
        axis.set_ylim(*y_limits)
    label_ax.set_xlim(0, 1)
    label_ax.axis("off")
    label_ax.text(0.01, 1.07, "Teacher / preservation", transform=label_ax.transAxes,
                  va="bottom", fontsize=8.5)
    label_ax.text(0.99, 1.07, "bpp", transform=label_ax.transAxes,
                  va="bottom", ha="right", fontsize=8.5)
    for index, (position, group) in enumerate(zip(positions, ordered)):
        if index % 3 == 1:
            label_ax.text(0.035, position, TEACHER_LABELS[group["teacher_seed"]],
                          va="center", ha="center", fontsize=8.5)
        label = "Logit (selected)" if group["objective"] == "late" else (
            "Feature (selected)" if group["selected_on_validation"] else "Feature (lower rate)"
        )
        label_ax.text(0.12, position, label, va="center", fontsize=8.2,
                      color="0.12")
        label_ax.text(0.99, position, f"{group['actual_bpp']:.3f}",
                      va="center", ha="right", fontsize=8.2)
        fits = group_rows(data, group)
        for axis, key in ((coarse_ax, "coarse_drop"), (fine_ax, "fine_drop")):
            color = COLORS[group["objective"]]
            mean = 100 * group[key]
            bounds = 100 * np.asarray(group[f"{key}_image_ci95"])
            if not (bounds[0] <= mean <= bounds[1]):
                raise ValueError(f"Stored interval excludes its mean for {group['key']}")
            axis.errorbar(mean, position, xerr=[[mean - bounds[0]], [bounds[1] - mean]],
                          fmt="D", ms=3.9, color=color, mec="white", mew=0.45,
                          elinewidth=1.0, capsize=2.1, capthick=0.8, zorder=3)
            axis.scatter([100 * row[key] for row in fits], position + np.array([-0.19, 0.19]),
                         s=7, edgecolors=color, facecolors="white", marker="o",
                         linewidths=0.65, zorder=4)
    for axis, limit, title, ticks in (
        (coarse_ax, 100 * data["coarse_allowance"], "(a) Coarse drop", [1, 2, 3, 4]),
        (fine_ax, 100 * data["fine_allowance"], "(b) Fine drop", [2, 3, 4, 5]),
    ):
        axis.axvline(limit, color="0.30", linestyle=(0, (3, 2)), lw=0.85, zorder=1)
        axis.text(0.5, -0.48, f"{limit:g}-point allowance", ha="center", va="bottom",
                  transform=axis.get_yaxis_transform(), fontsize=8.2, color="0.30")
        axis.set_title(title, pad=20, loc="center", fontsize=8.5)
        axis.set_xlabel("Accuracy drop (points)", labelpad=3, fontsize=8.2)
        axis.set_xticks(ticks)
        axis.set_yticks([])
        axis.spines[["top", "left", "right"]].set_visible(False)
        axis.tick_params(axis="x", length=2.5, pad=2)
    fine_ax.set_xlabel("")
    coarse_bounds = coarse_ax.get_position()
    shared_label_center = (coarse_bounds.x0 + fine_ax.get_position().x1) / 2
    coarse_ax.xaxis.label.set_x((shared_label_center - coarse_bounds.x0) / coarse_bounds.width)
    coarse_ax.set_xlim(0.95, 4.15)
    fine_ax.set_xlim(1.4, 5.15)
    readout_ax.set_title("(c) Same logit code", pad=20, loc="center", fontsize=8.5)
    readout_ax.set_xlim(41.0, 62.5)
    readout_ax.set_xticks([45, 50, 55, 60])
    readout_ax.set_xlabel("Fine accuracy (%)", labelpad=3, fontsize=8.2)
    readout_ax.set_yticks([])
    readout_ax.tick_params(axis="both", length=2.5, pad=2)
    readout_ax.spines[["top", "left", "right"]].set_visible(False)
    for position, teacher in zip(positions[2::3], TEACHERS):
        group = next(group for group in ordered if group["teacher_seed"] == teacher
                     and group["objective"] == "late")
        fits = group_rows(data, group)
        for key, _, marker in READOUTS:
            color = COLORS["late"]
            readout_ax.scatter(100 * group[key], position, s=22, marker=marker,
                               color=color, edgecolors="white", linewidths=0.45, zorder=3)
            readout_ax.scatter([100 * row[key] for row in fits],
                               position + np.array([-0.25, 0.25]), s=7,
                               marker="o", edgecolors=color, facecolors="white",
                               linewidths=0.65, zorder=4)
    for x_position, (_, label, _) in zip((43.0, 50.8, 59.7), READOUTS):
        readout_ax.text(x_position, -0.37, label, color="0.12", ha="center",
                        va="bottom", fontsize=8.2, linespacing=1.05)
    save_figure(figure, output, {
        "test_evidence": source_record(source),
        "frozen_validation_evidence": source_record(validation_source),
        "groups": ordered,
        "plotted_fit_count": 18,
        "rate_display": "Mean complete byte rates rounded to three decimals. Exact values are in groups.",
        "interval_scope": data["interval_scope"],
        "interpretation": "Test assessment of six fixed validation selections and three lower-rate feature-loss alternatives. No interpolated frontier. Shading identifies the three selected logit-loss messages and aligns them with panel (c). Hollow circles denote codec fits. Panel (c) shows three readout routes from the same message without image confidence intervals.",
    })


def plot_validation(source: Path, output: Path) -> None:
    data = read_evidence(source, "validation")
    rows = data["rows"]
    if len(rows) != 34 or len(data["groups"]) != 14:
        raise ValueError("Expected 34 validation fits and 14 paired groups")
    for group in data["groups"]:
        group_rows(data, group)
    configure_style()
    figure, axes = plt.subplots(2, 3, figsize=(6.5, 2.5), sharex=True, sharey=True)
    for column, teacher in enumerate(TEACHERS):
        for metric_index, (key, limit, label) in enumerate((
            ("coarse_drop", 100 * data["coarse_allowance"], "Coarse drop"),
            ("fine_drop", 100 * data["fine_allowance"], "Fine drop"),
        )):
            axis = axes[metric_index, column]
            for objective, color in COLORS.items():
                fits = [row for row in rows if row["teacher_seed"] == teacher
                        and row["objective"] == objective]
                axis.scatter([row["actual_bpp"] for row in fits],
                             [100 * row[key] for row in fits], s=15,
                             facecolors="white", edgecolors=color, linewidths=0.75, zorder=3)
                groups = [group for group in data["groups"]
                          if group["teacher_seed"] == teacher and group["objective"] == objective]
                for group in groups:
                    selected = group["selected_on_validation"]
                    axis.scatter(group["actual_bpp"], 100 * group[key],
                                 s=85 if selected else 28, marker="*" if selected else "D",
                                 facecolors="none", edgecolors=color,
                                 linewidths=0.75, zorder=4)
            axis.axhline(limit, color="0.35", linestyle=(0, (3, 2)), lw=0.85, zorder=1)
            axis.spines[["top", "right"]].set_visible(False)
            axis.set_xlim(0.90, 2.90)
            axis.set_ylim(1.0, 6.0)
            axis.set_xticks([1, 1.5, 2, 2.5])
            axis.set_yticks([1, 2, 3, 4, 5, 6])
            axis.tick_params(length=2.5, pad=2)
            if metric_index == 0:
                axis.set_title(f"({chr(97 + column)}) Teacher {TEACHER_LABELS[teacher]}",
                               loc="center", pad=4)
            if column == 0:
                axis.set_ylabel(f"{label}\n({limit:g}-point limit)", labelpad=3)
    figure.supxlabel("Complete rate (bpp)", x=0.54, y=0.105, fontsize=8.5)
    handles = [
        Line2D([], [], color=color, marker="s", lw=0, ms=4.5, label=label)
        for color, label in ((COLORS["early"], "Early-feature loss"), (COLORS["late"], "Logit loss"))
    ] + [
        Line2D([], [], color="0.3", marker="o", markerfacecolor="white", lw=0, ms=4, label="Codec fit"),
        Line2D([], [], color="0.3", marker="D", markerfacecolor="none", lw=0, ms=4,
               label="Two-fit mean"),
        Line2D([], [], color="0.3", marker="*", markerfacecolor="none", lw=0, ms=6,
               label="Selected mean"),
    ]
    figure.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.005),
                  ncol=5, frameon=False, handletextpad=0.4, columnspacing=1.0)
    figure.subplots_adjust(left=0.083, right=0.997, bottom=0.24, top=0.94,
                          wspace=0.10, hspace=0.16)
    save_figure(figure, output, {
        "frozen_validation_evidence": source_record(source),
        "rows": len(rows), "groups": len(data["groups"]),
        "selection_rule": data["selection_rule"],
        "interpretation": "Validation measurements. Every fitted codec appears in both its coarse and fine panel. Diamonds are paired-codec group means and stars are fixed selections. Six single-fit development controls have no paired-group mean. No connecting lines or interpolated frontier.",
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, default=EVIDENCE / "validation_selection_frozen.json",
                        help="frozen validation evidence that fixed the selections")
    parser.add_argument("--test-source", type=Path, default=EVIDENCE / "test_extension_summary.json",
                        help="test assessment of the frozen selections")
    parser.add_argument("--output", type=Path, default=FIGURES,
                        help="directory that receives both figures")
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=True)
    test_base = arguments.output / "paper_cifar_local_preservation"
    validation_base = arguments.output / "paper_cifar_validation"
    plot_test(arguments.test_source, arguments.source, test_base)
    plot_validation(arguments.source, validation_base)
    print(test_base.with_suffix(".pdf"))
    print(validation_base.with_suffix(".pdf"))


if __name__ == "__main__":
    main()

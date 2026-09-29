#!/usr/bin/env python3
"""Plot attained task risks and conditional errors from the saved finite ternary codes.

Panels (a) and (b) show both component risks with their simultaneous one-sided
99% upper bounds, for the known-cost and fitted-cost full-observation selectors
and the fitted reduced-observation selectors on the same three codebooks. Panel
(c) shows where the full-observation encoder places its original-task errors
across the five source states.
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
from matplotlib.legend_handler import HandlerBase, HandlerLine2D
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from taskscope.figures import FULL_COLOR, REDUCED_COLOR, crop_to_ink, render_preview, write_pdf
from taskscope.paths import FIGURES, REPO_ROOT, RESULTS

LIMIT = "#777777"
ALLOWED = "#F1F3F1"
GRID = "#EAE8EA"
TOLERANCES = np.array([0.35, 0.499])
SEEDS = [17, 23, 31]


class UpperBoundKey(HandlerBase):
    """Draw the plotted one-sided bound above its risk marker."""

    def create_artists(self, legend, orig_handle, xdescent, ydescent,
                       width, height, fontsize, trans):
        x = -xdescent + width / 2
        lower = -ydescent
        upper = lower + height
        return [
            Line2D([x, x], [lower, upper], color=FULL_COLOR, lw=0.7, transform=trans),
            Line2D([x], [upper], marker="_", color=FULL_COLOR, ms=4, mew=0.7,
                   linestyle="none", transform=trans),
            Line2D([x], [lower], marker="o", color=FULL_COLOR, markerfacecolor="white",
                   ms=3.4, mew=0.65, linestyle="none", transform=trans),
        ]


class AllowedRiskKey(HandlerBase):
    """Match the shaded allowed region and its dashed upper boundary."""

    def create_artists(self, legend, orig_handle, xdescent, ydescent,
                       width, height, fontsize, trans):
        left, bottom = -xdescent, -ydescent
        return [
            Rectangle((left, bottom), width, height, facecolor=ALLOWED,
                      edgecolor="none", transform=trans),
            Line2D([left, left + width], [bottom + height] * 2, color=LIMIT,
                   linestyle=(0, (3, 2)), lw=0.8, transform=trans),
        ]


def configure_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8,
        "axes.titlesize": 8.5, "axes.labelsize": 8,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.linewidth": 0.65, "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def save(figure: Figure, base: Path) -> list[float]:
    """Crop the rendered ink once, then export the preview at the same geometry."""
    path = write_pdf(figure, base, pad_inches=0.035)
    geometry = crop_to_ink(path)
    render_preview(path, base.with_suffix(".png"), dpi=300)
    return geometry


def shared_rate(records: list[dict]) -> float:
    rates = [record[key]["transmitted_rate"] if key == "known_full"
             else record["learned"][key]["transmitted_rate"]
             for record in records for key in ("known_full", "full", "reduced")]
    if len(set(rates)) != 1:
        raise ValueError("The shared-rate caption requires identical complete rates.")
    return rates[0]


def render(source: Path, reference: Path, base: Path) -> None:
    saved = json.loads(source.read_text())
    exact = json.loads(reference.read_text())
    records = saved["replicates"]
    if [record["seed"] for record in records] != SEEDS:
        raise ValueError(f"Expected the recorded codebooks in seed order {SEEDS}.")
    rate = shared_rate(records)
    configure_style()
    plot_height_pt = 2.10 * 72
    legend_extra_pt = 15
    figure_height_pt = plot_height_pt + legend_extra_pt
    figure = plt.figure(figsize=(6.5, figure_height_pt / 72))
    axes = [figure.add_axes((left, (bottom * plot_height_pt + legend_extra_pt) / figure_height_pt,
                             width, height * plot_height_pt / figure_height_pt))
            for left, bottom, width, height in (
        [0.075, 0.31, 0.215, 0.59],
        [0.365 + 7.2 / 468, 0.31, 0.215, 0.59],
        [0.690, 0.31, 0.305, 0.59],
    )]
    plotted = []
    source_offsets = np.array([-0.26, -0.16, -0.06])
    for task, axis in enumerate(axes[:2]):
        for index, record in enumerate(records):
            settings = [
                (record["known_full"], source_offsets[index], FULL_COLOR, False, "X_known"),
                (record["learned"]["full"], source_offsets[index] + 0.32, FULL_COLOR, True, "X_fitted"),
                (record["learned"]["reduced"], 1 + (index - 1) * 0.14, REDUCED_COLOR, True, "Y_fitted"),
            ]
            for values, x, color, hollow, condition in settings:
                mean = values["risks"][task]
                upper = values["risk_upper_study_99_hoeffding"][task]
                if not np.isfinite([mean, upper]).all() or upper < mean:
                    raise ValueError("Invalid saved one-sided risk bound.")
                axis.vlines(x, mean, upper, color=color, linewidth=0.7, zorder=2)
                axis.plot(x, upper, marker="_", color=color, ms=3.3, mew=0.7, zorder=2)
                axis.scatter(x, mean, s=12, facecolors="white" if hollow else color,
                             edgecolors=color, linewidths=0.65, zorder=3)
                plotted.append({"task": task, "seed": record["seed"], "condition": condition,
                                "risk": mean, "upper_99": upper, "categorical_x": float(x)})
        axis.axhline(TOLERANCES[task], color=LIMIT, linestyle=(0, (3, 2)), lw=0.8, zorder=1)
        axis.set(xlim=(-0.48, 1.42), xticks=[0, 1], xticklabels=["$X$", "$Y$"])
        if task == 0:
            axis.set(ylim=(0.19, 0.37), yticks=[0.20, 0.25, 0.30, 0.35], ylabel="Task risk")
        else:
            axis.set(ylim=(0.494, 0.522), yticks=[0.495, 0.499, 0.510, 0.520])
        axis.axhspan(axis.get_ylim()[0], TOLERANCES[task], color=ALLOWED, zorder=0)
        axis.set_title(("(a) Original task", "(b) Additional task")[task], pad=7)
    error_axis = axes[2]
    states = np.arange(5)
    conditional_offsets = np.array([-0.375, -0.225, -0.075])
    for index, record in enumerate(records):
        for values, offset, hollow in [
            (record["known_full"], conditional_offsets[index], False),
            (record["learned"]["full"], conditional_offsets[index] + 0.45, True),
        ]:
            error_axis.scatter(states + offset, values["state_error"], s=9,
                               facecolors="white" if hollow else FULL_COLOR,
                               edgecolors=FULL_COLOR, linewidths=0.6, zorder=3)
    error_axis.set(xticks=states,
                   xticklabels=[r"$0,-$", r"$0,+$", r"$1,-$", r"$1,+$", "$2$"],
                   xlim=(-0.48, 4.48), ylim=(0, 0.50), yticks=[0, 0.2, 0.4],
                   ylabel="Original-task error", xlabel="Class, confidence")
    error_axis.set_title("(c) Conditional error", pad=7)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(length=2.5, width=0.6, pad=3)
        axis.set_axisbelow(True)
        axis.grid(axis="y", color=GRID, lw=0.5)
        axis.xaxis.labelpad = 4
        axis.yaxis.labelpad = 4
    axes[0].set_xlabel("Encoder observation")
    shared_center = (axes[0].get_position().x0 + axes[1].get_position().x1) / 2
    shared_x = (shared_center - axes[0].get_position().x0) / axes[0].get_position().width
    for axis, x in ((axes[0], shared_x), (error_axis, 0.5)):
        axis.xaxis.set_label_coords(x, -0.22)
        axis.xaxis.label.set_verticalalignment("top")
    encoding_handles = [
        Line2D([], [], color=FULL_COLOR, linewidth=2.5, label="$X$  class + confidence"),
        Line2D([], [], color=REDUCED_COLOR, linewidth=2.5, label="$Y$  class only"),
        Line2D([], [], marker="o", linestyle="none", color=FULL_COLOR, markersize=4,
               label="Known costs"),
        Line2D([], [], marker="o", linestyle="none", color=FULL_COLOR, markerfacecolor="white",
               markersize=4, label="Fitted costs"),
    ]
    figure.legend(handles=encoding_handles, loc="lower center",
                  bbox_to_anchor=(0.5, 17.3 / figure_height_pt),
                  ncol=4, frameon=False, handlelength=1.2, handletextpad=0.4,
                  columnspacing=1.4, borderaxespad=0)
    upper_bound = Line2D([], [], label="Simultaneous 99% upper bound")
    allowed_risk = Rectangle((0, 0), 1, 1, label="Allowed risk / tolerance")
    codebooks = Line2D([], [], marker="o", linestyle="none", color=FULL_COLOR,
                       markersize=2.8, label="One circle per codebook")
    figure.legend(handles=[upper_bound, allowed_risk, codebooks], loc="lower center",
                  bbox_to_anchor=(0.5, 2.3 / figure_height_pt),
                  ncol=3, frameon=False, handlelength=1.4, handletextpad=0.4,
                  columnspacing=1.6, borderaxespad=0,
                  handler_map={upper_bound: UpperBoundKey(), allowed_risk: AllowedRiskKey(),
                               codebooks: HandlerLine2D(numpoints=3, marker_pad=0.1)})
    geometry = save(figure, base)
    base.with_suffix(".json").write_text(json.dumps({
        "source": str(source.resolve().relative_to(REPO_ROOT)),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "reference": str(reference.resolve().relative_to(REPO_ROOT)),
        "reference_sha256": hashlib.sha256(reference.read_bytes()).hexdigest(),
        "shared_rate": rate, "restricted_minimum": exact["reduced"]["rate"],
        "tolerances": TOLERANCES.tolist(), "page_points": geometry, "risk_marks": plotted,
        "conditional_error_marks": [
            {"seed": r["seed"], "known": r["known_full"]["state_error"],
             "fitted": r["learned"]["full"]["state_error"]} for r in records],
        "bounds": ("Saved simultaneous one-sided 99% Hoeffding upper bounds. Whiskers extend "
                   "from estimates to upper bounds."),
    }, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path,
                        default=RESULTS / "nonbinary/learned_selectors/summary.json")
    parser.add_argument("--reference", type=Path,
                        default=RESULTS / "nonbinary/matched_confirmation_reference.json")
    parser.add_argument("--output", type=Path, default=FIGURES)
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=True)
    render(arguments.source, arguments.reference, arguments.output / "figure3_risks")
    print(arguments.output / "figure3_risks.pdf")


if __name__ == "__main__":
    main()

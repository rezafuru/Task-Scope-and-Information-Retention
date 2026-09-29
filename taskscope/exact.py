"""Exact asymptotic rates for the two finite task families.

The quantities here are mutual-information rates of the stated finite models,
not measured packet lengths. The overlapping-threshold family has a reduced
observation that cannot attain the additional task's full-observation risk. The
confidence family shares one Bayes prediction across tasks while assigning
different costs to coding errors.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from scipy.optimize import LinearConstraint, brentq, minimize
from scipy.special import expit, xlogy

NOISE = 0.05
CONFIDENCE_WEIGHTS = np.array([0.9, 0.1])
CURVE_NAMES = ("overlapping_thresholds", "variable_confidence",
               "complementary_confidence_family")
GRID_POINTS = 401


def binary_entropy(probability):
    probability = np.asarray(probability, dtype=float)
    if np.any((probability < 0) | (probability > 1)):
        raise ValueError("A Bernoulli probability must lie in [0, 1].")
    return -(xlogy(probability, probability) + xlogy(1 - probability, 1 - probability)) / np.log(2)


def threshold_rates(risk):
    """Source and split-feature rates for the overlapping-threshold family."""
    risk = np.asarray(risk, dtype=float)
    error = (risk - NOISE) / (1 - 2 * NOISE)
    coarse_rate = float(binary_entropy(2 / 3))
    conditional_error = np.clip(1.5 * error, 0, 0.25)
    source = coarse_rate + (2 / 3) * (binary_entropy(0.25) - binary_entropy(conditional_error))
    source = np.where(error < 0, np.inf, source)
    split = np.where(error < (1 / 6) - 1e-14, np.inf, coarse_rate)
    return source, split


def confidence_rates(risk):
    """Source rate, split-feature rate, and per-state error allocation at one tolerance."""
    excess = float(risk) - 0.25
    if excess < 0:
        return float("inf"), float("inf"), np.full(2, np.nan)
    if excess >= 0.25:
        return 0.0, 0.0, np.full(2, 0.5)
    if excess == 0:
        return 1.0, 1.0, np.zeros(2)

    def residual(multiplier):
        errors = expit(-np.log(2) * multiplier * CONFIDENCE_WEIGHTS)
        return float(np.mean(CONFIDENCE_WEIGHTS * errors) - excess)

    upper = 1.0
    while residual(upper) > 0:
        upper *= 2
    multiplier = brentq(residual, 0, upper, xtol=1e-13)
    errors = expit(-np.log(2) * multiplier * CONFIDENCE_WEIGHTS)
    return float(1 - binary_entropy(errors).mean()), float(1 - binary_entropy(2 * excess)), errors


def independent_confidence_optimum(risk):
    """Optimize all four input-conditioned Bernoulli channels without imposing symmetry."""
    posterior = np.array([0.05, 0.45, 0.95, 0.55])
    coefficients = (1 - 2 * posterior) / 4

    def information(channel):
        return float(binary_entropy(channel.mean()) - binary_entropy(channel).mean())

    def gradient(channel):
        channel = np.clip(channel, 1e-12, 1 - 1e-12)
        mean = channel.mean()
        return (np.log2(channel / (1 - channel)) - np.log2(mean / (1 - mean))) / 4

    result = minimize(
        information, np.array([0.02, 0.02, 0.98, 0.98]), jac=gradient,
        constraints=[LinearConstraint(coefficients, -np.inf, risk - posterior.mean())],
        bounds=[(1e-12, 1 - 1e-12)] * 4, method="SLSQP",
        options={"ftol": 1e-12, "maxiter": 1000},
    )
    actual_risk = posterior.mean() + coefficients @ result.x
    if not result.success or actual_risk > risk + 1e-9:
        raise RuntimeError(f"Independent optimization failed: {result.message}, risk={actual_risk}")
    return float(result.fun), float(actual_risk)


def verify_calculations():
    """Check the closed forms against reference values and a free channel optimum."""
    source, split = threshold_rates(np.array([0.05, 0.20, 0.25]))
    assert np.allclose(source, [1.459147917027, 0.918295834054, 0.918295834054], atol=1e-11, rtol=0)
    assert np.isinf(split[0]) and np.allclose(split[1:], source[1:])
    comparisons = []
    for risk in [0.26, 0.275, 0.30, 0.35, 0.40, 0.45, 0.49]:
        analytic, split_rate, errors = confidence_rates(risk)
        numerical, achieved_risk = independent_confidence_optimum(risk)
        if abs(analytic - numerical) > 1e-7:
            raise AssertionError(f"Confidence optima disagree at risk {risk}")
        assert analytic < split_rate
        assert abs(0.25 + np.mean(CONFIDENCE_WEIGHTS * errors) - risk) < 1e-10
        comparisons.append({"risk": risk, "analytic_rate": analytic, "independent_rate": numerical,
                            "achieved_risk": achieved_risk,
                            "absolute_difference": abs(analytic - numerical)})
    return {"status": "passed", "independent_channel_comparisons": comparisons}


def rate_grids():
    threshold_risk = np.linspace(0.05, 0.25, GRID_POINTS)
    threshold_source, threshold_split = threshold_rates(threshold_risk)
    confidence_risk = np.linspace(0.25, 0.5, GRID_POINTS)
    confidence = [confidence_rates(risk) for risk in confidence_risk]
    return threshold_risk, threshold_source, threshold_split, confidence_risk, confidence


def write_rate_curves(path: Path) -> None:
    threshold_risk, threshold_source, threshold_split, confidence_risk, confidence = rate_grids()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["example", "risk", "source_direct_bits", "split_bits", "split_feasible"])
        for risk, source, split in zip(threshold_risk, threshold_source, threshold_split):
            writer.writerow(["overlapping_thresholds", risk, source,
                             split if np.isfinite(split) else "", bool(np.isfinite(split))])
        for risk, (source, split, _) in zip(confidence_risk, confidence):
            writer.writerow(["variable_confidence", risk, source, split, True])
            writer.writerow(["complementary_confidence_family", risk, split, split, True])


def build_summary() -> dict:
    single_rate, family_rate, single_errors = confidence_rates(0.30)
    return {
        "quantity": ("Ideal asymptotic mutual-information rates of finite models, "
                     "not measured packet lengths"),
        "threshold_label_noise": NOISE,
        "threshold_split_risk_floor": 0.20,
        "threshold_zero_allowance_baseline_gap_bits": float(binary_entropy(0.25) * 2 / 3),
        "confidence_label_noise": [0.05, 0.45],
        "confidence_bayes_risk": 0.25,
        "confidence_at_risk_0_30": dict(zip(["source_direct_bits", "split_bits"],
                                            confidence_rates(0.30)[:2])),
        "confidence_family_at_risk_0_30": {
            "source_direct_bits": family_rate,
            "split_bits": family_rate,
            "rate_increase_bits": family_rate - single_rate,
            "single_task_channel_second_task_risk":
                float(0.25 + np.mean(CONFIDENCE_WEIGHTS[::-1] * single_errors)),
        },
        "single_task_antecedent": ("Martinian, Wornell, and Zamir, IEEE TIT 2008, "
                                   "Section IV-E, equations (40)-(44)"),
        "verification": verify_calculations(),
    }


def load_curves(path: Path) -> dict[str, np.ndarray]:
    """Read the retained rate grid and check it against the closed forms."""
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    curves = {
        name: np.array([
            [float(row["risk"]), float(row["source_direct_bits"]),
             float(row["split_bits"]) if row["split_bits"] else np.inf]
            for row in rows if row["example"] == name
        ]) for name in CURVE_NAMES
    }
    for name, values in curves.items():
        if values.shape != (GRID_POINTS, 3) or not np.all(np.diff(values[:, 0]) > 0):
            raise ValueError(f"Unexpected retained rate grid for {name}")
    threshold = curves["overlapping_thresholds"]
    np.testing.assert_allclose(threshold[:, 1:], np.column_stack(threshold_rates(threshold[:, 0])),
                               rtol=0, atol=1e-12)
    single = curves["variable_confidence"]
    np.testing.assert_allclose(single[:, 1:],
                               np.array([confidence_rates(risk)[:2] for risk in single[:, 0]]),
                               rtol=0, atol=1e-12)
    pair = curves["complementary_confidence_family"]
    np.testing.assert_array_equal(pair[:, 0], single[:, 0])
    np.testing.assert_allclose(pair[:, 1:], np.repeat(single[:, 2:], 2, axis=1), rtol=0, atol=1e-12)
    return curves

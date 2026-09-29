"""Exact rates for the three-class family with a prescribed encoder observation.

The source draws a class Z in {0, 1, 2} and, for classes 0 and 1, an independent
fair confidence sign K. The full observation X = (Z, K) has five states, the
reduced observation Y = Z has three. Both observations share the Bayes
prediction, so the two tasks differ only in the conditional costs a coding error
incurs.

Rates here are mutual-information rates of the finite model, not measured packet
lengths. The reduced-observation minimum is available two ways: the numerical
dual over the marginal simplex in :func:`solve_risk`, and the closed ternary Fano
form in :func:`reduced_fano_bound`. They agree to about 2e-13 at the tolerance
the finite codes are confirmed against.
"""

from __future__ import annotations

import hashlib
import math
import platform
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
import scipy
from numpy.typing import NDArray
from scipy.optimize import brentq, minimize
from scipy.special import xlogy

CONFIRMATION_TOLERANCE = 0.499
PRIMARY_TOLERANCE = 0.35


class Observation(TypedDict):
    mass: NDArray[np.float64]
    cost: NDArray[np.float64]
    primary: NDArray[np.float64]
    classes: NDArray[np.int64]


class Solution(TypedDict):
    multiplier: float
    objective: float
    rate: float
    risk: float
    primary_risk: float
    marginal: list[float]
    channel: list[list[float]]
    convex_suboptimality_bound: float
    marginal_residual: float


def source_law(a: float = 0.55, gamma: float = 0.30,
               confidence_ratio: float = 0.85) -> tuple[Observation, Observation]:
    """Full and reduced observations of the three-class reference law.

    Rows of the full observation are (class 0, K=-), (class 0, K=+), (class 1,
    K=-), (class 1, K=+), class 2. The reduced observation averages the two
    confidence rows within each class under the source mass.
    """
    if not (1 / 3 < a < 1 and 0 < gamma < 1 and 0 <= confidence_ratio < 1):
        raise ValueError("Require 1/3 < a < 1, 0 < gamma < 1, and 0 <= confidence_ratio < 1")
    b = (1 - a) / 2
    h = a - b
    s = confidence_ratio * h / 3
    posterior = np.array([
        [a - s, b - s, b + 2 * s], [a + s, b + s, b - 2 * s],
        [b - s, a - s, b + 2 * s], [b + s, a + s, b - 2 * s],
        [b, b, a],
    ])
    classes = np.array([0, 0, 1, 1, 2])
    mass = np.array([(1 - gamma) / 4] * 4 + [gamma])
    if np.min(posterior) < 0 or not np.allclose(posterior.sum(1), 1):
        raise ValueError("Invalid source or conditional target probabilities")
    if not np.array_equal(posterior.argmax(1), classes):
        raise ValueError("The configuration changes the intended Bayes predictions")
    primary = (classes[:, None] != np.arange(3)).astype(float)
    full: Observation = {"mass": mass, "cost": 1 - posterior, "primary": primary, "classes": classes}
    reduced_mass = np.bincount(classes, weights=mass, minlength=3)
    reduced_cost = np.stack([(mass[classes == z, None] * full["cost"][classes == z]).sum(0)
                             / reduced_mass[z] for z in range(3)])
    reduced: Observation = {
        "mass": reduced_mass, "cost": reduced_cost,
        "primary": (np.arange(3)[:, None] != np.arange(3)).astype(float),
        "classes": np.arange(3),
    }
    return full, reduced


def variational(mass: NDArray[np.float64], cost: NDArray[np.float64], multiplier: float,
                marginal: NDArray[np.float64]) -> tuple[float, NDArray[np.float64], NDArray[np.float64]]:
    """Gibbs objective, its marginal gradient, and the induced prediction channel."""
    kernel = np.exp2(-multiplier * cost)
    partition = kernel @ marginal
    if np.any(partition <= 0):
        raise ValueError("Gibbs partition is nonpositive")
    value = -mass @ np.log2(partition)
    gradient = -(mass[:, None] * kernel / partition[:, None]).sum(0) / np.log(2)
    channel = kernel * marginal[None, :] / partition[:, None]
    return float(value), gradient, channel


def solve_scalar(observation: Observation, multiplier: float) -> Solution:
    """Minimize the strictly convex marginal objective at a fixed cost multiplier."""
    if not np.isfinite(multiplier) or multiplier < 0:
        raise ValueError("The multiplier must be finite and nonnegative")
    mass, cost = observation["mass"], observation["cost"]
    if multiplier == 0:
        marginal = np.eye(cost.shape[1])[np.argmin(mass @ cost)]
    else:
        starts = [np.array([1 / 3, 1 / 3, 1 / 3]), np.array([0.8, 0.1, 0.1]),
                  np.array([0.0, 0.0, 1.0]), np.array([0.5, 0.5, 0.0])]
        candidates = []
        for start in starts:
            result = minimize(lambda r: variational(mass, cost, multiplier, r)[:2], start,
                              jac=True, method="SLSQP", bounds=[(0, 1)] * cost.shape[1],
                              constraints={"type": "eq", "fun": lambda r: r.sum() - 1,
                                           "jac": lambda r: np.ones_like(r)},
                              options={"ftol": 1e-13, "maxiter": 1000})
            if result.success and abs(result.x.sum() - 1) < 1e-9:
                candidates.append(result.x)
        if not candidates:
            raise RuntimeError("All convex marginal optimizations failed")
        marginal = min(candidates, key=lambda r: variational(mass, cost, multiplier, r)[0])
    value, gradient, channel = variational(mass, cost, multiplier, marginal)
    actual_marginal = mass @ channel
    information = (mass[:, None] * (xlogy(channel, channel)
                                   - xlogy(channel, actual_marginal[None, :]))).sum() / np.log(2)
    certificate = max(0.0, float(gradient @ marginal - gradient.min()))
    residual = float(np.max(np.abs(actual_marginal - marginal)))
    if certificate > 2e-7 or residual > 2e-7:
        raise RuntimeError(f"Marginal optimum failed verification: {certificate=}, {residual=}")
    return {"multiplier": float(multiplier), "objective": value, "rate": float(information),
            "risk": float((mass[:, None] * channel * cost).sum()),
            "primary_risk": float((mass[:, None] * channel * observation["primary"]).sum()),
            "marginal": marginal.tolist(), "channel": channel.tolist(),
            "convex_suboptimality_bound": certificate, "marginal_residual": residual}


def solve_risk(observation: Observation, tolerance: float) -> Solution:
    """Find the multiplier whose optimum sits exactly at the tolerated additional risk."""
    mass, cost = observation["mass"], observation["cost"]
    bayes = float(mass @ cost.min(1))
    constant_risk = float((mass @ cost).min())
    if not bayes < tolerance < constant_risk:
        raise ValueError("Use a tolerance strictly between Bayes and zero-rate risks")
    high = 1.0
    while solve_scalar(observation, high)["risk"] > tolerance:
        high *= 2
        if high > 128:
            raise RuntimeError("The requested risk requires a larger numerical multiplier range")
    multiplier = brentq(lambda lam: solve_scalar(observation, lam)["risk"] - tolerance,
                        0, high, xtol=1e-10)
    return solve_scalar(observation, multiplier)


def gap_certificate(full: Observation, reduced: Observation, multiplier: float,
                    marginal: list[float] | NDArray[np.float64]) -> float:
    """Conditional KL between the lifted reduced channel and the full channel at one marginal."""
    x_value, _, x_channel = variational(full["mass"], full["cost"], multiplier, np.asarray(marginal))
    y_value, _, y_channel = variational(reduced["mass"], reduced["cost"], multiplier, np.asarray(marginal))
    lifted = y_channel[full["classes"]]
    kl = (full["mass"][:, None] * (xlogy(lifted, lifted) - xlogy(lifted, x_channel))).sum() / np.log(2)
    if not np.isfinite(kl) or abs(kl - (y_value - x_value)) > 1e-9:
        raise RuntimeError("Conditional KL identity failed")
    return float(kl)


def reduced_fano_bound(tolerance: float) -> float:
    """Lower bound every reduced-observation code by ternary Fano's inequality.

    The reduced costs are affine in the class-error indicator, so a tolerated
    additional risk fixes an error probability and the bound is
    ``H(Z) - h2(e) - e``. This is the closed form behind the quoted minimum, and
    it matches the numerical dual of :func:`solve_risk` on the same law.
    """
    _, reduced = source_law()
    base = reduced["cost"][0, 0]
    increment = reduced["cost"][0, 1] - base
    if not np.allclose(reduced["cost"], base + increment * reduced["primary"], atol=1e-12, rtol=0):
        raise ValueError("The reduced costs do not have the assumed ternary form")
    error = (tolerance - base) / increment
    if not 0 <= error <= 2 / 3:
        raise ValueError("Tolerance outside the monotonic ternary Fano range")
    entropy = -np.sum(reduced["mass"] * np.log2(reduced["mass"]))
    binary_entropy = -sum(value * math.log2(value) for value in (error, 1 - error) if value)
    return max(0.0, float(entropy - binary_entropy - error))


def reference(a: float = 0.55, gamma: float = 0.30, confidence_ratio: float = 0.85,
              beta: float = 2.5, primary_tolerance: float = PRIMARY_TOLERANCE) -> dict[str, Any]:
    """Matched-risk comparison of the two observations at one cost multiplier.

    The full observation is solved at ``beta`` and the reduced observation is
    then solved at the additional risk the full optimum attains, so the two rates
    answer the same requirement. ``dual_gap_bounds`` evaluates each dual at the
    other's marginal and ``kl_gap_bounds`` certifies the same interval through
    the conditional KL identity.
    """
    full, reduced = source_law(a, gamma, confidence_ratio)
    multiplier = beta / (a - (1 - a) / 2)
    x = solve_scalar(full, multiplier)
    y = solve_risk(reduced, x["risk"])
    if max(x["primary_risk"], y["primary_risk"]) > primary_tolerance:
        raise ValueError("Standalone optima do not meet the retained primary requirement")
    y_at_x = solve_scalar(reduced, multiplier)
    x_at_y = solve_scalar(full, y["multiplier"])
    return {"configuration": {"a": a, "gamma": gamma, "confidence_ratio": confidence_ratio, "beta": beta},
            "primary_tolerance": primary_tolerance, "additional_tolerance": x["risk"],
            "full": x, "reduced": y, "matched_rate_gap": y["rate"] - x["rate"],
            "dual_gap_bounds": [y_at_x["objective"] - x["objective"], y["objective"] - x_at_y["objective"]],
            "kl_gap_bounds": [gap_certificate(full, reduced, multiplier, y_at_x["marginal"]),
                              gap_certificate(full, reduced, y["multiplier"], x_at_y["marginal"])],
            "source_law": {name: {k: v.tolist() for k, v in observation.items()}
                           for name, observation in (("full", full), ("reduced", reduced))},
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                        "scipy": scipy.__version__}}


def matched_reference(additional_tolerance: float = CONFIRMATION_TOLERANCE, a: float = 0.55,
                      gamma: float = 0.30, confidence_ratio: float = 0.85,
                      primary_tolerance: float = PRIMARY_TOLERANCE) -> dict[str, Any]:
    """Comparison at the tolerance the finite codes are confirmed against.

    Solves the full observation for the multiplier that attains
    ``additional_tolerance`` exactly, then reports the same record as
    :func:`reference` at that operating point. This is the basis of the quoted
    reduced-observation minimum, not the earlier development multiplier.
    """
    full, _ = source_law(a, gamma, confidence_ratio)
    matched = solve_risk(full, additional_tolerance)
    beta = matched["multiplier"] * (a - (1 - a) / 2)
    record = reference(a, gamma, confidence_ratio, beta, primary_tolerance)
    record["declared_additional_tolerance"] = additional_tolerance
    return record


def bound_agreement(tolerance: float = CONFIRMATION_TOLERANCE) -> dict[str, float]:
    """Reduced-observation minimum from the closed form and from the numerical dual."""
    _, reduced = source_law()
    closed_form = reduced_fano_bound(tolerance)
    numerical = solve_risk(reduced, tolerance)
    return {"tolerance": tolerance, "closed_form_bits": closed_form,
            "numerical_dual_bits": numerical["rate"],
            "absolute_difference": abs(closed_form - numerical["rate"]),
            "numerical_dual_risk": numerical["risk"],
            "numerical_dual_primary_risk": numerical["primary_risk"]}

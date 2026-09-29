"""Numerical checks of the conditional-error-profile criterion.

Each case compares the closed-form error-allocation rate with a free
optimization over all joint binary actions, and compares the full-observation
and reduced-observation rates against membership of the constant profile in the
convex hull of the active normalized profiles.
"""

from __future__ import annotations

import itertools

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, minimize
from scipy.special import xlogy

CYCLIC = np.array([[0.9, 0.5, 0.1], [0.1, 0.9, 0.5], [0.5, 0.1, 0.9]])


def entropy(x):
    return -(xlogy(x, x) + xlogy(1 - x, 1 - x)) / np.log(2)


def common_prediction_channel(errors, count):
    """Joint-action channel that predicts the same bit for every task.

    ``errors`` are the per-state probabilities of predicting the opposite bit.
    Mass sits on the all-zeros and all-ones tuples, with a small interior
    component so that the starting point is not on its bounds.
    """
    errors = np.asarray(errors, dtype=float)
    n = errors.size
    channel = np.full((2 * n, count), 0.0)
    channel[:n, 0] = 1 - errors
    channel[:n, -1] = errors
    channel[n:, 0] = errors
    channel[n:, -1] = 1 - errors
    interior = 1e-6
    channel = (1 - interior) * channel + interior / count
    return channel


def check(name, weights, state_probabilities, normalized_allowances):
    w = np.asarray(weights, dtype=float)
    pk = np.asarray(state_probabilities, dtype=float)
    m, n = w.shape
    mean = w @ pk
    allowances = mean * np.asarray(normalized_allowances)
    weighted = w * pk
    restricted = minimize(
        lambda e: 1 - pk @ entropy(e),
        np.full(n, min(normalized_allowances) / 2),
        jac=lambda e: pk * np.log2(e / (1 - e)),
        constraints=[LinearConstraint(weighted, -np.inf, allowances)],
        bounds=Bounds(np.full(n, 1e-12), np.full(n, 0.5)),
        method="SLSQP",
        options={"ftol": 2e-13, "maxiter": 2000},
    )
    actions = np.array(list(itertools.product([0, 1], repeat=m)))
    count = len(actions)
    source = np.tile(pk / 2, 2)
    costs = np.stack([
        np.vstack([w[q, :, None] * (actions[:, q] != z) for z in [0, 1]])
        for q in range(m)
    ])
    risk_matrix = (costs * source[None, :, None]).reshape(m, -1)
    row_sum = np.kron(np.eye(2 * n), np.ones((1, count)))
    delta = min(normalized_allowances)
    initial = np.full((2 * n, count), delta / count)
    initial[:n, 0] += 1 - delta
    initial[n:, -1] += 1 - delta

    def mutual_info(flat):
        channel = flat.reshape(2 * n, count)
        output = source @ channel
        return np.sum(source[:, None] * xlogy(channel, channel / output)) / np.log(2)

    def gradient(flat):
        channel = flat.reshape(2 * n, count)
        output = source @ channel
        return (source[:, None] * np.log2(channel / output)).ravel()

    def solve(start):
        return minimize(
            mutual_info,
            start.ravel(),
            jac=gradient,
            constraints=[
                LinearConstraint(row_sum, np.ones(2 * n), np.ones(2 * n)),
                LinearConstraint(risk_matrix, -np.inf, allowances),
            ],
            bounds=Bounds(np.full(start.size, 1e-12), np.ones(start.size)),
            method="SLSQP",
            options={"ftol": 2e-12, "maxiter": 3000},
        )

    starts = {"spread": initial, "common_prediction": common_prediction_channel(restricted.x, count)}
    attempts = {label: solve(start) for label, start in starts.items()}
    if not any(result.success for result in attempts.values()):
        raise RuntimeError(f"Rate optimization failed for {name}")
    unrestricted = min((result for result in attempts.values() if result.success),
                       key=lambda result: result.fun)
    profiles = w / mean[:, None]
    hull = linprog(
        np.zeros(m), A_eq=np.vstack([profiles.T, np.ones(m)]),
        b_eq=np.ones(n + 1), bounds=[(0, None)] * m, method="highs",
    )
    active = np.flatnonzero(np.asarray(normalized_allowances) == min(normalized_allowances))
    active_hull = linprog(
        np.zeros(len(active)), A_eq=np.vstack([profiles[active].T, np.ones(len(active))]),
        b_eq=np.ones(n + 1), bounds=[(0, None)] * len(active), method="highs",
    )
    if hull.status not in (0, 2) or active_hull.status not in (0, 2):
        raise RuntimeError("Profile-hull optimization did not resolve feasibility")
    if not restricted.success:
        raise RuntimeError(f"Error-allocation optimization failed for {name}")
    output = {
        "name": name,
        "tasks": m,
        "states": n,
        "weights": w.tolist(),
        "state_probabilities": pk.tolist(),
        "normalized_allowances": list(normalized_allowances),
        "unrestricted_rate": float(unrestricted.fun),
        "formula_rate": float(restricted.fun),
        "reduced_rate": float(1 - entropy(min(0.5, min(normalized_allowances)))),
        "absolute_discrepancy": float(abs(unrestricted.fun - restricted.fun)),
        "unrestricted_starts": {
            label: {"rate": float(result.fun), "success": bool(result.success),
                    "message": result.message}
            for label, result in attempts.items()
        },
        "formula_success": bool(restricted.success),
        "max_risk_violation": float(max(0, np.max(risk_matrix @ unrestricted.x - allowances))),
        "max_normalization_violation": float(np.max(abs(row_sum @ unrestricted.x - 1))),
        "formula_errors": restricted.x.tolist(),
        "constant_profile_in_hull": bool(hull.success),
        "hull_coefficients": hull.x.tolist() if hull.success else None,
        "active_tasks": active.tolist(),
        "constant_profile_in_active_hull": bool(active_hull.success),
        "active_hull_coefficients": active_hull.x.tolist() if active_hull.success else None,
    }
    if output["absolute_discrepancy"] > 1e-7 or output["max_risk_violation"] > 1e-9:
        raise AssertionError(f"Channel and allocation rates disagree for {name}")
    gap = output["reduced_rate"] - output["formula_rate"]
    if active_hull.success and abs(gap) > 1e-7:
        raise AssertionError(f"Active-profile equality failed for {name}")
    if not active_hull.success and gap <= 1e-7:
        raise AssertionError(f"Strict profile gap unresolved for {name}")
    return output


def run_checks(progress=None):
    """Run the cyclic, nonuniform, and random cases used in the appendix."""
    records = []

    def record(*arguments):
        output = check(*arguments)
        records.append(output)
        if progress is not None:
            progress(output)

    for size in [1, 2, 3]:
        for subset in itertools.combinations(range(3), size):
            record("cyclic_" + "".join(str(x + 1) for x in subset),
                   CYCLIC[list(subset)], [1 / 3] * 3, [0.1] * size)
    for allowances in [[0.05, 0.1, 0.2], [0.2, 0.05, 0.4], [0.42, 0.36, 0.45], [0.01, 0.49, 0.49]]:
        record("cyclic_unbalanced_" + str(len(records)), CYCLIC, [1 / 3] * 3, allowances)
    pk = np.array([0.2, 0.3, 0.5])
    t = np.array([0.6, -0.4, 0.0])
    w = np.vstack([0.3 * (1 + t), 0.6 * (1 - t / 2)])
    record("nonuniform_hull", w, pk, [0.23, 0.23])
    record("nonuniform_hull_single", w[:1], pk, [0.23])
    rng = np.random.default_rng(6026)
    for m, n in [(2, 4), (3, 4), (4, 3)]:
        pk = rng.dirichlet(np.ones(n))
        w = rng.uniform(0.04, 0.98, size=(m, n))
        record(f"random_{m}_{n}", w, pk, rng.uniform(0.06, 0.36, size=m))
    return records

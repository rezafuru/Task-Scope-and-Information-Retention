"""Finite ternary codebooks for the three-class family, with transmitted indices.

Each codebook holds 8,192 distinct ternary words of length 16. Weighted Lloyd
rounds assign training blocks by total conditional loss and replace each word
position with its minimum-loss class. An encoder picks the codeword minimizing
the summed conditional loss over the block, exhaustively and with exactly
represented integer scores, and transmits the 13-bit index. Every packet carries
a 48-byte ``N3CB`` header with the codebook hash, so reported rates are complete
message lengths including framing and padding.

Two selector families run on the same frozen codebooks. The known-cost selector
uses the declared source table. The fitted-cost selector estimates the five
conditional-cost rows from 2^20 independent noisy labels, rounds them to
multiples of 2^-16, and pools them by class when the encoder observes only the
class. Two controls retain the class distribution and the Bayes prediction: one
keeps only the original deterministic task, the other removes the additional
task's confidence dependence.

Provenance of the retained evidence: ``profile_n16_b13/run.json`` records
``source_sha256`` 570b2691..., an earlier revision of the fitting script, while
the confirmation and the two replications record 459d2db4..., matching the file
this module was ported from. The seed-17 codebook that the confirmation uses is
pinned by ``codebook_sha256`` a95d4558... and re-verified by
``learned_selectors/run.json``, so the reported results follow from the stored
codebook regardless of that script drift.

The source is synthetic, so the whole pipeline runs without any dataset. Fitting
is GPU work but ``--device cpu`` is supported, and every reported number follows
from a stored codebook rather than from refitting.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linprog

from taskscope.nonbinary import (
    CONFIRMATION_TOLERANCE,
    PRIMARY_TOLERANCE,
    matched_reference,
    reduced_fano_bound,
    source_law,
)

HEADER = struct.Struct("<4sBHBQ32s")
MAGIC = b"N3CB"
SCALE = 2400
TABLE_SCALE = 65536

# One known-cost and two fitted-cost selectors per codebook, plus the two
# controls: eleven evaluations, each contributing two component means.
STUDY_EVALUATIONS = 11

# Frozen codebooks the fitted selectors are assessed on, in seed order. Paths are
# relative to the run directory the codes stage wrote.
REPLICATES = (
    (17, "confirmation_seed17", "codebook.npz", "assessment.npz",
     "assessment_risks.npy", "assessment.bin"),
    (23, "replication_seed23", "full_b13_codebook.npz", "splits.npz",
     "selected_full_risks.npy", "selected_full_b13.bin"),
    (31, "replication_seed31", "full_b13_codebook.npz", "splits.npz",
     "selected_full_risks.npy", "selected_full_b13.bin"),
)


def resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _prepare(device: torch.device, threads: int = 4) -> None:
    torch.set_num_threads(threads)
    torch.backends.cuda.matmul.allow_tf32 = False


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def codebook_hash(codebook: np.ndarray) -> bytes:
    values = np.asarray(codebook)
    if values.ndim != 2 or values.dtype.kind not in "ui" or np.any(values < 0) or np.any(values > 2):
        raise ValueError("A codebook must contain integer ternary predictions")
    return hashlib.sha256(struct.pack("<QQ", *values.shape) + values.astype("u1").tobytes()).digest()


def pack_indices(indices: np.ndarray, codebook: np.ndarray) -> bytes:
    """Frame fixed-width codeword indices behind the header that pins the codebook."""
    values = np.asarray(indices)
    count, length = codebook.shape
    bits = (count - 1).bit_length()
    if count < 1 or length < 1 or length > 65535 or bits > 24:
        raise ValueError("Invalid codebook dimensions")
    if values.ndim != 1 or values.dtype.kind not in "ui" or np.any(values < 0) or np.any(values >= count):
        raise ValueError("Invalid codeword indices")
    shifts = np.arange(bits - 1, -1, -1, dtype=np.int64)
    binary = ((values.astype(np.uint64)[:, None] >> shifts.astype(np.uint64)) & 1).astype("u1")
    header = HEADER.pack(MAGIC, 1, length, bits, len(values), codebook_hash(codebook))
    return header + np.packbits(binary.ravel()).tobytes()


def unpack_indices(message: bytes, codebook: np.ndarray) -> np.ndarray:
    if len(message) < HEADER.size:
        raise ValueError("Truncated packet header")
    magic, version, length, bits, count, digest = HEADER.unpack_from(message)
    if magic != MAGIC or version != 1 or bits > 24 or length != codebook.shape[1]:
        raise ValueError("Unsupported or mismatched packet header")
    if digest != codebook_hash(codebook) or bits != (len(codebook) - 1).bit_length():
        raise ValueError("Packet codebook identity mismatch")
    if len(message) != HEADER.size + (count * bits + 7) // 8:
        raise ValueError("Packet length mismatch")
    binary = np.unpackbits(np.frombuffer(message, dtype="u1", offset=HEADER.size))
    if binary[count * bits:].any():
        raise ValueError("Nonzero packet padding")
    shifts = np.arange(bits - 1, -1, -1, dtype=np.uint64)
    indices = (binary[:count * bits].reshape(count, bits).astype(np.uint64) << shifts).sum(1)
    if np.any(indices >= len(codebook)):
        raise ValueError("Packet contains an unused codeword index")
    return indices.astype(np.int64)


def draw_blocks(count: int, length: int, seed: int) -> np.ndarray:
    """Independent source blocks over the five full-observation states."""
    full, _ = source_law()
    return np.random.default_rng(seed).choice(5, (count, length), p=full["mass"]).astype("u1")


def score_table(access: str, condition: str, primary_weight: float = 0.0) -> np.ndarray:
    """Known conditional costs as integers the exhaustive selector represents exactly."""
    full, reduced = source_law(confidence_ratio=0 if condition == "no_confidence" else 0.85)
    observation = full if access == "full" else reduced
    costs = (observation["primary"] if condition == "primary"
             else observation["cost"] + primary_weight * observation["primary"])
    scaled = np.rint(costs * SCALE)
    if not np.allclose(scaled / SCALE, costs, atol=1e-12, rtol=0):
        raise ValueError("The exact integer selector does not represent these costs")
    return scaled.astype(np.int64)


def observed_states(blocks: np.ndarray, access: str) -> np.ndarray:
    """Map source states to what the encoder sees, pooling confidence when reduced."""
    if access not in {"full", "reduced"}:
        raise ValueError("Unknown observation")
    return blocks if access == "full" else np.array([0, 0, 1, 1, 2], dtype="u1")[blocks]


def search(states: torch.Tensor, codebook: torch.Tensor, costs: torch.Tensor,
           chunk: int = 512) -> torch.Tensor:
    """Return the first conditional-loss minimizer using exactly represented integer scores."""
    if chunk < 1 or states.ndim != 2 or codebook.ndim != 2 or states.shape[1] != codebook.shape[1]:
        raise ValueError("Invalid search dimensions")
    if states.shape[1] * int(costs.max() - costs.min()) >= 2 ** 23:
        raise ValueError("Scores exceed the exact float32 integer range")
    features = torch.nn.functional.one_hot(codebook.long(), 3)[..., :2].float().flatten(1).T.contiguous()
    answers = []
    for start in range(0, len(states), chunk):
        rows = costs[states[start:start + chunk].long()]
        contrasts = (rows[..., :2] - rows[..., 2:]).flatten(1)
        answers.append((contrasts @ features).argmin(1))
    return torch.cat(answers) if answers else torch.empty(0, dtype=torch.int64, device=states.device)


def fit_codebook(train: np.ndarray, access: str, condition: str, bits: int, seed: int,
                 rounds: int, device: torch.device | str,
                 primary_weight: float = 0.0) -> tuple[np.ndarray, list[dict]]:
    """Weighted Lloyd from distinct random ternary words, checked for monotone descent."""
    if not 0 <= bits <= 24 or rounds < 0 or len(train) < 1 or 2 ** bits > 3 ** train.shape[1]:
        raise ValueError("Invalid fit dimensions")
    rng = np.random.default_rng(seed)
    length = train.shape[1]
    initial = rng.choice(3 ** length, 2 ** bits, replace=False)
    book = torch.as_tensor(((initial[:, None] // (3 ** np.arange(length))) % 3), device=device)
    states = torch.as_tensor(observed_states(train, access), device=device)
    costs = torch.as_tensor(score_table(access, condition, primary_weight),
                            dtype=torch.float32, device=device)
    history = []
    for iteration in range(rounds + 1):
        assignments = search(states, book, costs)
        losses = costs[states.long(), book[assignments]].double().mean().item() / SCALE
        history.append({"iteration": iteration, "training_objective": losses,
                        "occupied_codewords": int(torch.unique(assignments).numel())})
        if len(history) > 1 and losses > history[-2]["training_objective"] + 1e-10:
            raise RuntimeError("Lloyd iteration increased its training objective")
        if iteration == rounds:
            break
        sums = torch.zeros((len(book), length, 3), dtype=torch.float64, device=device)
        sums.index_add_(0, assignments, costs[states.long()].double())
        update = sums.argmin(2)
        current = sums.gather(2, book[..., None]).squeeze(2)
        book = torch.where(current == sums.min(2).values, book, update)
    return book.cpu().numpy().astype("u1"), history


def measure(blocks: np.ndarray, predictions: np.ndarray, condition: str) -> tuple[dict, np.ndarray]:
    """Average both conditional target losses within blocks, under the true law."""
    full, _ = source_law(confidence_ratio=0 if condition == "no_confidence" else 0.85)
    primary = full["primary"][blocks, predictions]
    additional = full["cost"][blocks, predictions]
    risks = np.stack((primary.mean(1), additional.mean(1)), 1)
    state_count = np.bincount(blocks.ravel(), minlength=5)
    errors = np.bincount(blocks.ravel(), weights=primary.ravel(), minlength=5)
    return {"risks": risks.mean(0).tolist(),
            "risk_se": (risks.std(0, ddof=1) / math.sqrt(len(risks))).tolist(),
            "prediction_frequency": (np.bincount(predictions.ravel(), minlength=3)
                                     / predictions.size).tolist(),
            "state_count": state_count.tolist(),
            "state_error": [float(errors[i] / state_count[i]) if state_count[i] else None
                            for i in range(5)]}, risks


def risk_upper_bound(risks: np.ndarray, condition: str,
                     failure_probability: float = 0.01) -> np.ndarray:
    """One-sided Hoeffding allowance for independent-block means of both components.

    The original loss range is one and the additional range comes from the source
    table. Passing ``failure_probability`` divided by the number of evaluations
    gives the union bound over every reported component mean.
    """
    if risks.ndim != 2 or risks.shape[1] != 2 or len(risks) < 1 or not 0 < failure_probability < 1:
        raise ValueError("Invalid risk-bound inputs")
    full, _ = source_law(confidence_ratio=0 if condition == "no_confidence" else 0.85)
    ranges = np.array([1.0, np.ptp(full["cost"])])
    allowance = ranges * math.sqrt(math.log(2 / failure_probability) / (2 * len(risks)))
    return risks.mean(0) + allowance


def _encode(blocks: np.ndarray, codebook: np.ndarray, table: np.ndarray, access: str,
            device: torch.device | str) -> tuple[np.ndarray, bytes]:
    """Select codewords from the observation alone and check the packet roundtrip."""
    states = torch.as_tensor(observed_states(blocks, access), device=device)
    book = torch.as_tensor(codebook.astype(np.int64), device=device)
    costs = torch.as_tensor(table, dtype=torch.float32, device=device)
    indices = search(states, book, costs).cpu().numpy()
    packet = pack_indices(indices, codebook)
    decoded = unpack_indices(packet, codebook)
    if not np.array_equal(indices, decoded):
        raise RuntimeError("Packet roundtrip changed codeword indices")
    return decoded, packet


def evaluate(blocks: np.ndarray, codebook: np.ndarray, access: str, condition: str,
             device: torch.device | str, primary_weight: float = 0.0) -> tuple[dict, np.ndarray, bytes]:
    """Known-cost selection on one codebook, with its complete transmitted rate."""
    decoded, packet = _encode(blocks, codebook, score_table(access, condition, primary_weight),
                              access, device)
    record, risks = measure(blocks, codebook[decoded], condition)
    record.update(message_bytes=len(packet), header_bytes=HEADER.size,
                  transmitted_rate=8 * len(packet) / blocks.size,
                  index_bits=(len(codebook) - 1).bit_length(),
                  used_codewords=int(len(np.unique(decoded))))
    return record, risks, packet


def evaluate_table(blocks: np.ndarray, codebook: np.ndarray, table: np.ndarray, access: str,
                   condition: str, device: torch.device | str) -> tuple[dict, np.ndarray, bytes]:
    """Encode with a supplied cost table, then assess decoded predictions under the true law."""
    decoded, packet = _encode(blocks, codebook, table, access, device)
    stats, risks = measure(blocks, codebook[decoded], condition)
    stats.update(message_bytes=len(packet), header_bytes=HEADER.size,
                 transmitted_rate=8 * len(packet) / blocks.size,
                 packet_sha256=hashlib.sha256(packet).hexdigest(),
                 assessment_blocks=len(blocks), source_symbols=int(blocks.size))
    return stats, risks, packet


def draw_calibration(count: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Independent labelled examples from the declared source law, for cost fitting."""
    if count < 1:
        raise ValueError("Calibration count must be positive")
    full, _ = source_law()
    generator = np.random.default_rng(seed)
    states = generator.choice(5, count, p=full["mass"]).astype("u1")
    cumulative = (1 - full["cost"]).cumsum(1)
    targets = (generator.random(count)[:, None] >= cumulative[states, :2]).sum(1).astype("u1")
    return states, targets


def fit_costs(states: np.ndarray, targets: np.ndarray, access: str) -> tuple[np.ndarray, np.ndarray]:
    """Estimate a loss table exclusively from observed calibration states and labels."""
    if states.ndim != 1 or states.shape != targets.shape or len(states) < 1:
        raise ValueError("Calibration states and labels must be aligned vectors")
    if np.any(states > 4) or np.any(targets > 2) or np.any(states < 0) or np.any(targets < 0):
        raise ValueError("Invalid calibration state or label")
    observed = observed_states(states, access)
    cells = 5 if access == "full" else 3
    counts = np.bincount(observed.astype(np.int64) * 3 + targets, minlength=cells * 3).reshape(cells, 3)
    totals = counts.sum(1)
    if np.any(totals == 0):
        raise ValueError("Every observation cell needs calibration examples")
    return 1 - counts / totals[:, None], counts


def quantized_costs(costs: np.ndarray) -> np.ndarray:
    """Round fitted costs to multiples of 2^-16 so selection scores stay exact integers."""
    if costs.ndim != 2 or costs.shape[1] != 3 or not np.isfinite(costs).all():
        raise ValueError("Expected a finite three-prediction cost table")
    if np.any(costs < 0) or np.any(costs > 1):
        raise ValueError("Conditional zero-one risks must lie in [0,1]")
    return np.rint(costs * TABLE_SCALE).astype(np.int64)


def add_bounds(stats: dict, risks: np.ndarray, condition: str,
               tolerances: tuple[float, float] = (PRIMARY_TOLERANCE, CONFIRMATION_TOLERANCE)) -> None:
    """Attach the union-bounded one-sided 99% upper risks and the pass decision."""
    upper = risk_upper_bound(risks, condition, failure_probability=0.01 / STUDY_EVALUATIONS)
    stats["risk_upper_study_99_hoeffding"] = upper.tolist()
    stats["meets_expanded_requirements_study_99"] = bool(np.all(upper <= np.array(tolerances)))


def fit(output: Path, *, device_name: str, length: int, bits, access, condition: str, seed: int,
        rounds: int, train_count: int, dev_count: int, assessment_count: int,
        primary_weight: float, additional_tolerance: float, safety_margin: float,
        threads: int, source_file: Path) -> None:
    """Fit codebooks, select a development mixture, then assess on fresh blocks.

    Selection is frozen before any assessment index or risk is computed. With one
    candidate per observation the linear program reduces to a feasibility check.
    """
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "run.json").exists():
        raise FileExistsError(f"Refusing to overwrite an existing run: {output}")
    device = resolve_device(device_name)
    _prepare(device, threads)
    started = time.perf_counter()
    configuration = dict(
        output=str(output), length=length, bits=list(bits), access=list(access),
        condition=condition, seed=seed, rounds=rounds, train_count=train_count,
        dev_count=dev_count, assessment_count=assessment_count, primary_weight=primary_weight,
        additional_tolerance=additional_tolerance, safety_margin=safety_margin,
        device=str(device), threads=threads,
        source_sha256=hashlib.sha256(source_file.read_bytes()).hexdigest(),
        python=sys.version.split()[0], torch=torch.__version__, cuda=torch.version.cuda,
        split_seeds={"training": seed + 100000, "development": seed + 200000,
                     "assessment": seed + 300000},
        prediction_tuples=[[0, 0], [1, 1], [2, 2]],
        model_accounting="Codebooks are shared models. Every packet includes its 32-byte model hash.",
    )
    write_json(output / "run.json", configuration)
    write_json(output / "matched_reference.json", matched_reference(additional_tolerance))
    train = draw_blocks(train_count, length, seed + 100000)
    dev = draw_blocks(dev_count, length, seed + 200000)
    assessment = draw_blocks(assessment_count, length, seed + 300000)
    np.savez_compressed(output / "splits.npz", training=train, development=dev, assessment=assessment)
    records = []
    for width in bits:
        for observation in access:
            fit_start = time.perf_counter()
            book, history = fit_codebook(train, observation, condition, width, seed + width * 1000,
                                         rounds, device, primary_weight)
            if device.type == "cuda":
                torch.cuda.synchronize()
            fit_seconds = time.perf_counter() - fit_start
            name = f"{observation}_b{width}"
            np.savez_compressed(output / f"{name}_codebook.npz", codebook=book)
            dev_record, dev_risks, dev_packet = evaluate(dev, book, observation, condition,
                                                         device, primary_weight)
            (output / f"{name}_development.bin").write_bytes(dev_packet)
            np.save(output / f"{name}_development_risks.npy", dev_risks)
            record = {"name": name, "access": observation, "bits": width, "length": length,
                      "condition": condition, "seed": seed, "fit_seconds": fit_seconds,
                      "history": history, "development": dev_record,
                      "codebook_bytes": (output / f"{name}_codebook.npz").stat().st_size,
                      "codebook_sha256": codebook_hash(book).hex()}
            write_json(output / f"{name}.json", record)
            records.append(record)
            print(json.dumps({"name": name, "fit_seconds": fit_seconds,
                              "development": dev_record}), flush=True)
    tolerance = np.array([PRIMARY_TOLERANCE, additional_tolerance])
    selections = {}
    for observation in access:
        available = [r for r in records if r["access"] == observation]
        risk = np.array([r["development"]["risks"] for r in available])
        rate = np.array([r["bits"] / r["length"] for r in available])
        active = [0] if condition == "primary" else [0, 1]
        result = linprog(rate, A_ub=risk[:, active].T, b_ub=(tolerance - safety_margin)[active],
                         A_eq=np.ones((1, len(available))), b_eq=[1], bounds=(0, None), method="highs")
        if result.success:
            counts = np.floor(result.x * assessment_count).astype(int)
            remainder = assessment_count - counts.sum()
            for index in np.argsort(-(result.x * assessment_count - counts))[:remainder]:
                counts[index] += 1
            selections[observation] = {
                "names": [r["name"] for r in available], "block_counts": counts.tolist(),
                "development_risks_after_rounding": (counts @ risk / counts.sum()).tolist(),
                "public_schedule": "Consecutive block groups in the listed order, fixed before assessment."}
        else:
            selections[observation] = {"feasible": False, "reason": result.message}
    write_json(output / "selection.json", {"tolerances": tolerance.tolist(),
                                           "safety_margin": safety_margin,
                                           "selections": selections})
    selected = {}
    for observation, selection in selections.items():
        if "block_counts" not in selection:
            continue
        offset = 0
        total_bytes = 0
        parts = []
        all_predictions = []
        for name, count in zip(selection["names"], selection["block_counts"]):
            if not count:
                continue
            book = np.load(output / f"{name}_codebook.npz")["codebook"]
            blocks = assessment[offset:offset + count]
            _, risk, packet = evaluate(blocks, book, observation, condition, device, primary_weight)
            (output / f"selected_{name}.bin").write_bytes(packet)
            parts.append(risk)
            all_predictions.append(book[unpack_indices(packet, book)])
            total_bytes += len(packet)
            offset += count
        risks = np.concatenate(parts)
        stats, _ = measure(assessment, np.concatenate(all_predictions), condition)
        stats.update(transmitted_rate=8 * total_bytes / assessment.size, message_bytes=total_bytes,
                     source_symbols=int(assessment.size), packet_count=len(parts),
                     framing_bytes=len(parts) * HEADER.size,
                     risk_upper_99_normal=(risks.mean(0) + 2.576 * risks.std(0, ddof=1)
                                           / math.sqrt(len(risks))).tolist(),
                     risk_upper_99_hoeffding=risk_upper_bound(risks, condition).tolist())
        np.save(output / f"selected_{observation}_risks.npy", risks)
        selected[observation] = stats
    for record in records:
        book = np.load(output / f"{record['name']}_codebook.npz")["codebook"]
        stats, risks, packet = evaluate(assessment, book, record["access"], condition,
                                        device, primary_weight)
        record["assessment"] = stats
        (output / f"{record['name']}_assessment.bin").write_bytes(packet)
        np.save(output / f"{record['name']}_assessment_risks.npy", risks)
        write_json(output / f"{record['name']}.json", record)
    summary = {"selected": selected, "candidates": records,
               "elapsed_seconds": time.perf_counter() - started,
               "peak_cuda_memory_bytes": (torch.cuda.max_memory_allocated(device)
                                          if device.type == "cuda" else 0)}
    write_json(output / "summary.json", summary)
    print(json.dumps({key: summary[key] for key in
                      ("selected", "elapsed_seconds", "peak_cuda_memory_bytes")}), flush=True)


def confirm(output: Path, *, codebook_path: Path, device_name: str, access: str, condition: str,
            seed: int, assessment_count: int, primary_weight: float, additional_tolerance: float,
            threads: int, source_file: Path) -> None:
    """Assess a frozen codebook and risk requirement on an independent source draw.

    Every decision is recorded before the confirmation blocks are generated. The
    reported rate is the complete packet length, compared against the
    reduced-observation minimum from ternary Fano at the same tolerance.
    """
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "run.json").exists():
        raise FileExistsError(f"Refusing to overwrite an existing confirmation: {output}")
    device = resolve_device(device_name)
    _prepare(device, threads)
    started = time.perf_counter()
    book = np.load(codebook_path)["codebook"]
    length = book.shape[1]
    configuration = dict(
        output=str(output), length=length, access=[access], condition=condition, seed=seed,
        assessment_count=assessment_count, primary_weight=primary_weight,
        additional_tolerance=additional_tolerance, confirm_codebook=str(codebook_path),
        device=str(device), threads=threads,
        source_sha256=hashlib.sha256(source_file.read_bytes()).hexdigest(),
        codebook_sha256=codebook_hash(book).hex(), assessment_seed=seed + 300000,
        tolerances=[PRIMARY_TOLERANCE, additional_tolerance],
        statistical_bound="Simultaneous one-sided Hoeffding at failure probability 0.01.",
    )
    write_json(output / "run.json", configuration)
    np.savez_compressed(output / "codebook.npz", codebook=book)
    write_json(output / "matched_reference.json", matched_reference(additional_tolerance))
    blocks = draw_blocks(assessment_count, length, seed + 300000)
    np.savez_compressed(output / "assessment.npz", assessment=blocks)
    stats, risks, packet = evaluate(blocks, book, access, condition, device, primary_weight)
    (output / "assessment.bin").write_bytes(packet)
    np.save(output / "assessment_risks.npy", risks)
    upper = risk_upper_bound(risks, condition)
    tolerance = np.array([PRIMARY_TOLERANCE, additional_tolerance])
    fano_bound = reduced_fano_bound(additional_tolerance)
    stats.update(risk_upper_99_hoeffding=upper.tolist(),
                 requirements_certified_99=bool(np.all(upper <= tolerance)),
                 reduced_rate_lower_bound=fano_bound,
                 rate_below_reduced_bound=fano_bound - stats["transmitted_rate"],
                 elapsed_seconds=time.perf_counter() - started,
                 peak_cuda_memory_bytes=(torch.cuda.max_memory_allocated(device)
                                         if device.type == "cuda" else 0))
    write_json(output / "summary.json", stats)
    print(json.dumps(stats), flush=True)


def learned_selectors(source: Path, output: Path, *, device_name: str, calibration_count: int,
                      control_blocks: int, threads: int, source_file: Path) -> None:
    """Assess fitted-cost selectors against known-cost selection on frozen codebooks.

    One independent calibration draw per codebook, no tuning and no refitting.
    Both selectors read the same frozen codebook and the same assessment blocks,
    so the paired difference isolates the fitted costs. ``source`` is the run
    directory the codes stage wrote, holding the three replicate subdirectories
    named in :data:`REPLICATES`.
    """
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "run.json").exists():
        raise FileExistsError(f"Run already exists: {output}")
    device = resolve_device(device_name)
    _prepare(device, threads)
    started = time.perf_counter()
    model_records = []
    for seed, directory, book_file, blocks_file, risks_file, packet_file in REPLICATES:
        book = np.load(source / directory / book_file)["codebook"]
        model_records.append({"seed": seed, "directory": directory, "codebook": book_file,
                              "assessment": blocks_file, "known_risks": risks_file,
                              "known_packet": packet_file,
                              "codebook_sha256": codebook_hash(book).hex(),
                              "calibration_seed": 73000000 + seed})
    configuration = {
        "calibration_count_per_fit": calibration_count, "replicates": model_records,
        "tolerances": [PRIMARY_TOLERANCE, CONFIRMATION_TOLERANCE], "table_scale": TABLE_SCALE,
        "device": str(device),
        "calibration_policy": "One independent labeled calibration draw per frozen codebook. No tuning or refitting.",
        "comparison": "Known and fitted selectors use the same frozen codebook and assessment source blocks.",
        "study_failure_probability": 0.01, "risk_means_in_union": 2 * STUDY_EVALUATIONS,
        "control_blocks": control_blocks,
        "source_sha256": hashlib.sha256(source_file.read_bytes()).hexdigest(),
        "python": sys.version.split()[0], "torch": torch.__version__, "cuda": torch.version.cuda,
    }
    write_json(output / "run.json", configuration)
    results = []
    for model in model_records:
        path = source / model["directory"]
        book = np.load(path / model["codebook"])["codebook"]
        blocks = np.load(path / model["assessment"])["assessment"]
        known_risks = np.load(path / model["known_risks"])
        known_packet = (path / model["known_packet"]).read_bytes()
        known_predictions = book[unpack_indices(known_packet, book)]
        known, reconstructed_risks = measure(blocks, known_predictions, "expanded")
        if not np.array_equal(known_risks, reconstructed_risks):
            raise RuntimeError("Stored known-selector risks disagree with decoded packets")
        known.update(message_bytes=len(known_packet), header_bytes=HEADER.size,
                     transmitted_rate=8 * len(known_packet) / blocks.size,
                     packet_sha256=hashlib.sha256(known_packet).hexdigest())
        add_bounds(known, known_risks, "expanded")
        states, labels = draw_calibration(calibration_count, model["calibration_seed"])
        seed_output = output / f"seed{model['seed']}"
        seed_output.mkdir()
        np.savez_compressed(seed_output / "calibration.npz", states=states, targets=labels)
        record = {"seed": model["seed"], "known_full": known, "learned": {}}
        for access in ("full", "reduced"):
            estimates, counts = fit_costs(states, labels, access)
            table = quantized_costs(estimates)
            np.savez_compressed(seed_output / f"{access}_table.npz", estimated_cost=estimates,
                                target_counts=counts, integer_cost=table)
            stats, risks, packet = evaluate_table(blocks, book, table, access, "expanded", device)
            add_bounds(stats, risks, "expanded")
            paired = risks - known_risks
            changed = np.mean(unpack_indices(packet, book) != unpack_indices(known_packet, book))
            stats.update(paired_risk_change=paired.mean(0).tolist(),
                         paired_risk_change_se=(paired.std(0, ddof=1) / np.sqrt(len(paired))).tolist(),
                         max_cost_rounding_error=float(np.max(np.abs(estimates - table / TABLE_SCALE))),
                         changed_index_fraction=float(changed))
            (seed_output / f"learned_{access}.bin").write_bytes(packet)
            np.save(seed_output / f"learned_{access}_risks.npy", risks)
            record["learned"][access] = stats
        write_json(seed_output / "summary.json", record)
        results.append(record)
        print(json.dumps(record), flush=True)
    first = model_records[0]
    first_path = source / first["directory"]
    book = np.load(first_path / first["codebook"])["codebook"]
    blocks = np.load(first_path / first["assessment"])["assessment"][:control_blocks]
    controls = {}
    for condition in ("primary", "no_confidence"):
        packets = []
        for access in ("full", "reduced"):
            stats, risks, packet = evaluate_table(blocks, book, score_table(access, condition),
                                                  access, condition, device)
            packets.append(packet)
        if packets[0] != packets[1]:
            raise RuntimeError(f"The {condition} equality control produced different packets")
        add_bounds(stats, risks, condition)
        stats["full_and_reduced_packets_identical"] = True
        stats["interpretation"] = ("Equality of these fixed-code outputs under the control law. "
                                   "No claim of expanded-family feasibility.")
        (output / f"{condition}_control.bin").write_bytes(packets[0])
        np.save(output / f"{condition}_control_risks.npy", risks)
        controls[condition] = stats
    summary = {"replicates": results, "controls": controls,
               "reduced_rate_lower_bound": reduced_fano_bound(CONFIRMATION_TOLERANCE),
               "elapsed_seconds": time.perf_counter() - started,
               "peak_cuda_memory_bytes": (torch.cuda.max_memory_allocated(device)
                                          if device.type == "cuda" else 0)}
    write_json(output / "summary.json", summary)
    print(json.dumps({key: value for key, value in summary.items()
                      if key != "replicates"}), flush=True)

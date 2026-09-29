"""Finite binary vector quantizers with transmitted block indices.

Decision bits are fair and independent, with independent equiprobable
confidence states. A codeword index is sent from a shared binary codebook and
every task decoder recovers that codeword. Rates count indices, headers, and
padding, so they are complete message lengths rather than entropy estimates.

The source is synthetic, so the whole pipeline runs without any dataset.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import struct
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import LinearConstraint, linprog, minimize
from scipy.special import xlogy

HEADER = struct.Struct("<IHBx")
THREE_STATE_PROFILES = [[0.9, 0.5, 0.1], [0.1, 0.9, 0.5], [0.5, 0.1, 0.9]]


def entropy(p):
    p = np.asarray(p)
    return -(xlogy(p, p) + xlogy(1 - p, 1 - p)) / np.log(2)


def exact_rate(weights, tolerances):
    """Asymptotic full- and reduced-observation rates for one requirement family."""
    weights = np.asarray(weights, dtype=float)
    base = (1 - weights.mean(axis=1)) / 2
    allowance = np.asarray(tolerances) - base
    states = weights.shape[1]
    reduced_error = min(0.5, float(np.min(allowance / weights.mean(axis=1))))
    if reduced_error < 0:
        raise ValueError("Risk below the Bayes risk")
    result = minimize(
        lambda e: 1 - entropy(e).mean(),
        np.full(states, reduced_error * 0.99),
        jac=lambda e: np.log2(e / (1 - e)) / states,
        bounds=[(1e-10, 0.5)] * states,
        constraints=[LinearConstraint(weights / states, -np.inf, allowance)],
        method="SLSQP", options={"ftol": 1e-12, "maxiter": 1000},
    )
    if not result.success or np.max(weights @ result.x / states - allowance) > 1e-8:
        raise RuntimeError(f"Exact allocation failed: {result.message}")
    return {
        "full_rate": float(result.fun), "reduced_rate": float(1 - entropy(reduced_error)),
        "errors": result.x.tolist(), "risks": (base + weights @ result.x / states).tolist(),
        "active": np.flatnonzero(np.abs(weights @ result.x / states - allowance) < 1e-7).tolist(),
    }


def conditions():
    """Requirement families: single, duplicate, complementary, and three-state."""
    result = []
    for tau in [0.0, 0.125, 0.5]:
        first = [0.9 - 0.8 * tau, 0.1 + 0.8 * tau]
        second = first[::-1]
        law = f"two_tau_{tau:g}"
        specs = [("single", [first], [0.30]), ("pair", [first, second], [0.30, 0.30])]
        if tau == 0:
            specs.extend([
                ("duplicate", [first, first], [0.30, 0.30]),
                ("loose", [first, second], [0.30, 0.46]),
                ("tolerance_034", [first, second], [0.30, 0.34]),
                ("tolerance_038", [first, second], [0.30, 0.38]),
                ("tolerance_042", [first, second], [0.30, 0.42]),
            ])
        for name, weights, tolerances in specs:
            result.append(dict(name=f"{law}_{name}", law=law, family=name, tau=tau,
                               weights=weights, tolerances=tolerances,
                               exact=exact_rate(weights, tolerances)))
    three = THREE_STATE_PROFILES
    for size, name in [(1, "single"), (2, "partial"), (3, "cyclic")]:
        result.append(dict(name=f"three_{name}", law="three", family=name, tau=0.0,
                           weights=three[:size], tolerances=[0.30] * size,
                           exact=exact_rate(three[:size], [0.30] * size)))
    return result


def pack_indices(indices, block_length, bits):
    indices = np.asarray(indices, dtype=np.uint32)
    if bits < 1 or bits > 24 or np.any(indices >= 2 ** bits):
        raise ValueError("Invalid fixed-width indices")
    shifts = np.arange(bits - 1, -1, -1, dtype=np.uint32)
    payload = np.packbits(((indices[:, None] >> shifts) & 1).astype(np.uint8).ravel())
    return HEADER.pack(len(indices), block_length, bits) + payload.tobytes()


def unpack_indices(message):
    count, length, bits = HEADER.unpack_from(message)
    if not 1 <= bits <= 24 or len(message) != HEADER.size + math.ceil(count * bits / 8):
        raise ValueError("Malformed message")
    binary = np.unpackbits(np.frombuffer(message, dtype=np.uint8, offset=HEADER.size))[:count * bits]
    indices = (binary.reshape(count, bits).astype(np.uint32) <<
               np.arange(bits - 1, -1, -1, dtype=np.uint32)).sum(axis=1)
    return indices, length, bits


def draw(count, length, states, generator, device):
    z = torch.randint(2, (count, length), generator=generator, device=device).float()
    observed = torch.randint(states, (count, length), generator=generator, device=device)
    dummy = torch.randint(states, (count, length), generator=generator, device=device)
    return z, observed, dummy


def search(z, observed, codebook, profile, chunk=512):
    indices = []
    for start in range(0, len(z), chunk):
        values = z[start:start + chunk]
        costs = profile[observed[start:start + chunk]]
        scores = ((1 - 2 * values) * costs) @ codebook.T
        indices.append(scores.argmin(dim=1))
    return torch.cat(indices)


def fit_codebook(length, bits, profile, seed, device, rounds, train_count):
    """Weighted Lloyd updates from distinct random binary words."""
    generator = torch.Generator(device=device).manual_seed(seed)
    integers = torch.randperm(2 ** length, generator=generator, device=device)[:2 ** bits]
    shifts = torch.arange(length, device=device)
    codebook = ((integers[:, None] >> shifts) & 1).float()
    for _ in range(rounds):
        z, observed, _ = draw(train_count, length, len(profile), generator, device)
        indices = search(z, observed, codebook, profile)
        weighted = profile[observed]
        ones = torch.zeros_like(codebook).index_add_(0, indices, z * weighted)
        totals = torch.zeros_like(codebook).index_add_(0, indices, weighted)
        update = (2 * ones >= totals).float()
        codebook = torch.where(totals > 0, update, codebook)
    return codebook


def fit_score_profile(profile, seed, device, steps):
    """Learn positive state scores from soft known-cost assignments on separate blocks."""
    generator = torch.Generator(device=device).manual_seed(seed)
    codebook = torch.randint(2, (64, 12), generator=generator, device=device).float()
    raw = torch.nn.Parameter(torch.zeros(len(profile), device=device))
    optimizer = torch.optim.Adam([raw], lr=0.04)
    for _ in range(steps):
        z, observed, _ = draw(512, 12, len(profile), generator, device)
        learned = torch.softmax(raw, dim=0) * profile.sum()
        distances = ((1 - 2 * z) * learned[observed]) @ codebook.T
        logits = -distances / 0.05
        target_scores = ((1 - 2 * z) * profile[observed]) @ codebook.T
        target_prob = torch.softmax(-target_scores / 0.05, dim=1)
        loss = -(target_prob * torch.log_softmax(logits, dim=1)).sum(dim=1).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return (torch.softmax(raw, dim=0) * profile.sum()).detach()


def rounded_scores(profile, decimals):
    """Fixed precision keeps exactly tied codeword decisions from turning on 1e-7 asymmetries."""
    scale = 10 ** decimals
    return torch.round(profile * scale) / scale


def profiles_for(law):
    if law == "three":
        w = np.array(THREE_STATE_PROFILES)
        return [w[0], 0.75 * w[0] + 0.25 * w[1], (w[0] + w[1]) / 2,
                0.25 * w[0] + 0.75 * w[1], w[1], w.mean(axis=0)]
    tau = float(law.split("_")[-1])
    w = np.array([0.9 - 0.8 * tau, 0.1 + 0.8 * tau])
    if tau == 0.5:
        return [w]
    return [alpha * w + (1 - alpha) * w[::-1]
            for alpha in [1, 0.875, 0.75, 0.625, 0.5, 0.375, 0.25, 0]]


def block_errors(z, observed, reconstructed, states):
    mismatch = (z != reconstructed).float()
    return torch.stack([(mismatch * (observed == state)).mean(dim=1)
                        for state in range(states)], dim=1).cpu().numpy()


def evaluate(codebook, profile, seed, count, device, access, learned_profile=None):
    """Encode fresh blocks under one observation condition and return per-block errors."""
    generator = torch.Generator(device=device).manual_seed(seed)
    z, observed, dummy = draw(count, codebook.shape[1], len(profile), generator, device)
    if access == "reduced":
        supplied = dummy
        used = torch.full_like(profile, profile.mean())
    else:
        supplied = observed
        used = profile if learned_profile is None else learned_profile
    indices = search(z, supplied, codebook, used)
    reconstructed = codebook[indices]
    errors = block_errors(z, observed, reconstructed, len(profile))
    return errors, indices.cpu().numpy(), z[:16].cpu().numpy(), observed[:16].cpu().numpy()


def measured(errors, weights):
    weights = np.asarray(weights)
    risk_blocks = errors @ weights.T + (1 - weights.mean(axis=1)) / 2
    means = risk_blocks.mean(axis=0)
    se = risk_blocks.std(axis=0, ddof=1) / np.sqrt(len(errors))
    return {"risks": means.tolist(), "standard_errors": se.tolist(),
            "conditional_errors": (errors.mean(axis=0) * weights.shape[1]).tolist()}


def resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _prepare(device: torch.device) -> None:
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False


def candidate_record(path, base, meta, validation, assessment):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, validation=validation, assessment=assessment)
    return dict(meta, error_file=str(path.relative_to(base)))


def run(output: Path, *, device_name, seeds, lengths, laws, min_rate, max_rate, rounds,
        train_count, validation_count, assessment_count, score_steps, sample_seed_offset,
        source_file: Path):
    """Fit codebooks and encoders across seeds, block lengths, and index widths."""
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    device = resolve_device(device_name)
    _prepare(device)
    started = time.time()
    configuration = dict(
        seeds=list(seeds), lengths=list(lengths), laws=laws, min_rate=min_rate, max_rate=max_rate,
        rounds=rounds, train_count=train_count, validation_count=validation_count,
        assessment_count=assessment_count, score_steps=score_steps,
        sample_seed_offset=sample_seed_offset, device=str(device),
        source_sha256=hashlib.sha256(source_file.read_bytes()).hexdigest(),
        python=sys.version.split()[0], torch=torch.__version__, numpy=np.__version__,
        started_unix=started, command="run",
        claim=("Learned binary vector quantizers and structured score encoders "
               "with known conditional costs"),
    )
    (output / "configuration.json").write_text(json.dumps(configuration, indent=2) + "\n")
    all_conditions = conditions()
    (output / "predictions.json").write_text(json.dumps(all_conditions, indent=2) + "\n")
    records = []
    score_cache = {}
    for seed in seeds:
        for length in lengths:
            bits_grid = range(max(2, round(length * min_rate)),
                              min(length, round(length * max_rate)) + 1)
            for bits in bits_grid:
                reduced_profile = torch.tensor([0.5, 0.5], device=device)
                reduced_book = fit_codebook(length, bits, reduced_profile,
                                            100000 * seed + length * 100 + bits,
                                            device, rounds, train_count)
                for law in laws.split(","):
                    raw_profiles = profiles_for(law)
                    fit_profiles = [raw_profiles[0]]
                    if law == "three":
                        fit_profiles.append(raw_profiles[2])
                    books = [("reduced_fit", reduced_book)]
                    for fit_id, fit in enumerate(fit_profiles):
                        tensor = torch.tensor(fit, device=device, dtype=torch.float32)
                        books.append((f"full_fit_{fit_id}",
                                      fit_codebook(length, bits, tensor,
                                                   100000 * seed + length * 100 + bits,
                                                   device, rounds, train_count)))
                    for book_name, book in books:
                        book_path = (output / "codebooks" /
                                     f"s{seed}_n{length}_b{bits}_{law}_{book_name}.npz")
                        book_path.parent.mkdir(exist_ok=True)
                        np.savez_compressed(book_path,
                                            codebook=book.cpu().numpy().astype(np.uint8))
                        for profile_id, raw_profile in enumerate(raw_profiles):
                            profile = torch.tensor(raw_profile, device=device, dtype=torch.float32)
                            score_key = (seed, law, profile_id)
                            if score_key not in score_cache:
                                score_cache[score_key] = fit_score_profile(
                                    profile, 80000 + seed * 100 + profile_id, device, score_steps)
                            learned = score_cache[score_key]
                            variants = [("full", "reference", None), ("full", "learned", learned)]
                            if profile_id == 0:
                                variants.append(("reduced", "reference", None))
                            for access, method, score in variants:
                                name = (f"s{seed}_n{length}_b{bits}_{law}_{book_name}"
                                        f"_p{profile_id}_{access}_{method}")
                                validation, _, _, _ = evaluate(
                                    book, profile, sample_seed_offset + 31000 + length,
                                    validation_count, device, access, score)
                                assessment, indices, _, _ = evaluate(
                                    book, profile, sample_seed_offset + 91000 + length,
                                    assessment_count, device, access, score)
                                message = pack_indices(indices, length, bits)
                                restored, restored_n, restored_b = unpack_indices(message)
                                if (restored_n != length or restored_b != bits
                                        or not np.array_equal(indices, restored)):
                                    raise AssertionError("Index serialization changed the message")
                                message_path = output / "messages" / f"{name}.bin"
                                message_path.parent.mkdir(exist_ok=True)
                                message_path.write_bytes(message)
                                meta = dict(
                                    name=name, seed=seed, length=length, bits=bits, law=law,
                                    book=book_name, profile_id=profile_id,
                                    profile=list(raw_profile),
                                    learned_profile=learned.cpu().tolist(),
                                    access=access, method=method, payload_rate=bits / length,
                                    complete_rate=len(message) * 8 / (len(indices) * length),
                                    complete_bytes=len(message), header_bytes=HEADER.size,
                                    public_codebook_bits=int(book.numel()),
                                    codebook_file=str(book_path.relative_to(output)),
                                    message_file=str(message_path.relative_to(output)))
                                records.append(candidate_record(
                                    output / "errors" / f"{name}.npz", output,
                                    meta, validation, assessment))
                    (output / "candidates.json").write_text(json.dumps(records, indent=2) + "\n")
                print(json.dumps(dict(seed=seed, length=length, bits=bits,
                                      candidates=len(records),
                                      elapsed_seconds=time.time() - started)), flush=True)
    (output / "runtime.json").write_text(
        json.dumps(dict(elapsed_seconds=time.time() - started), indent=2) + "\n")


def rescore(source: Path, output: Path, *, device_name, sample_seed_offset, score_decimals,
            source_file: Path):
    """Repeat validation and assessment with fitted state scores rounded to fixed precision."""
    source = source.resolve()
    output = output.resolve()
    if source == output:
        raise ValueError("Corrected assessment must preserve the original output directory")
    output.mkdir(parents=True, exist_ok=True)
    configuration = json.loads((source / "configuration.json").read_text())
    configuration.update(
        source_sha256=hashlib.sha256(source_file.read_bytes()).hexdigest(),
        original_results=str(source), sample_seed_offset=sample_seed_offset,
        score_decimals=score_decimals, command="rescore",
        correction="Round every fitted score vector at a fixed precision before hard selection")
    (output / "configuration.json").write_text(json.dumps(configuration, indent=2) + "\n")
    (output / "predictions.json").write_bytes((source / "predictions.json").read_bytes())
    device = resolve_device(device_name)
    _prepare(device)
    candidates = json.loads((source / "candidates.json").read_text())
    records = []
    reference_records = {}
    codebooks = {}
    started = time.time()
    reused = 0
    for index, original in enumerate(candidates):
        candidate = dict(original)
        book_path = candidate["codebook_file"]
        if book_path not in codebooks:
            codebooks[book_path] = torch.tensor(np.load(source / book_path)["codebook"],
                                                dtype=torch.float32, device=device)
        book = codebooks[book_path]
        profile = torch.tensor(candidate["profile"], dtype=torch.float32, device=device)
        raw_score = torch.tensor(candidate["learned_profile"], dtype=torch.float32, device=device)
        score = rounded_scores(raw_score, score_decimals)
        candidate["unrounded_learned_profile"] = candidate["learned_profile"]
        candidate["learned_profile"] = score.cpu().tolist()
        reference_name = candidate["name"].replace("_full_learned", "_full_reference")
        if candidate["method"] == "learned" and torch.equal(score, profile):
            reference = reference_records[reference_name]
            for field in ["error_file", "message_file", "complete_rate", "complete_bytes"]:
                candidate[field] = reference[field]
            candidate["identical_reference"] = reference_name
            reused += 1
        else:
            supplied_score = score if candidate["method"] == "learned" else None
            length = candidate["length"]
            validation, _, _, _ = evaluate(book, profile, sample_seed_offset + 31000 + length,
                                           configuration["validation_count"], device,
                                           candidate["access"], supplied_score)
            assessment, indices, _, _ = evaluate(book, profile, sample_seed_offset + 91000 + length,
                                                 configuration["assessment_count"], device,
                                                 candidate["access"], supplied_score)
            message = pack_indices(indices, length, candidate["bits"])
            restored, _, _ = unpack_indices(message)
            if not np.array_equal(restored, indices):
                raise AssertionError("Corrected encoder message did not round trip")
            message_path = output / "messages" / f"{candidate['name']}.bin"
            message_path.parent.mkdir(exist_ok=True)
            message_path.write_bytes(message)
            candidate.update(message_file=str(message_path.relative_to(output)),
                             complete_bytes=len(message),
                             complete_rate=len(message) * 8 / (len(indices) * length))
            candidate = candidate_record(output / "errors" / f"{candidate['name']}.npz",
                                         output, candidate, validation, assessment)
        records.append(candidate)
        if candidate["method"] == "reference" and candidate["access"] == "full":
            reference_records[candidate["name"]] = candidate
        if (index + 1) % 113 == 0:
            print(json.dumps(dict(candidates=index + 1, identical_references=reused,
                                  elapsed_seconds=time.time() - started)), flush=True)
    (output / "candidates.json").write_text(json.dumps(records, indent=2) + "\n")
    (output / "runtime.json").write_text(
        json.dumps(dict(elapsed_seconds=time.time() - started,
                        identical_reference_encoders=reused), indent=2) + "\n")
    _copy_codebooks(source, output, records)


def _copy_codebooks(source: Path, output: Path, records) -> None:
    """Keep rescored candidates self-contained by carrying their codebooks across."""
    for name in sorted({record["codebook_file"] for record in records}):
        destination = output / name
        if destination.exists():
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((source / name).read_bytes())


def select(candidates, base: Path, condition, output: Path, assessment_count, safety_margin):
    """Choose a source-independent mixture of fitted codes meeting every component risk."""
    weights = np.asarray(condition["weights"])
    records = []
    val_risks = []
    for candidate in candidates:
        data = np.load(base / candidate["error_file"])
        records.append((candidate, data["assessment"]))
        val_risks.append(np.array(measured(data["validation"], weights)["risks"]))
    rates = np.array([item[0]["payload_rate"] for item in records])
    matrix = np.array(val_risks).T
    result = linprog(rates, A_ub=matrix,
                     b_ub=np.asarray(condition["tolerances"]) - safety_margin,
                     A_eq=np.ones((1, len(rates))), b_eq=[1], bounds=(0, None), method="highs")
    if not result.success:
        return {"feasible_validation": False, "solver_message": result.message}
    active = np.flatnonzero(result.x > 1e-9)
    fractions = result.x[active]
    counts = np.floor(fractions * assessment_count).astype(int)
    counts[np.argmax(fractions)] += assessment_count - counts.sum()
    schedule = np.repeat(active, counts)
    np.random.default_rng(617).shuffle(schedule)
    errors = np.zeros_like(records[0][1])
    total_bytes = 0
    transmitted = []
    for index in active:
        selected = schedule == index
        candidate, source_errors = records[index]
        errors[selected] = source_errors[selected]
        indices, length, bits = unpack_indices((base / candidate["message_file"]).read_bytes())
        component = pack_indices(indices[selected], length, bits)
        packet_path = output / f"component_{index}.bin"
        packet_path.write_bytes(component)
        restored, _, _ = unpack_indices(component)
        if not np.array_equal(restored, indices[selected]):
            raise AssertionError("Selected mixture does not decode")
        total_bytes += len(component)
        transmitted.append(dict(candidate=candidate["name"], blocks=int(selected.sum()),
                                fraction=float(selected.mean()), bytes=len(component),
                                message_file=str(packet_path.relative_to(base)),
                                codebook_file=candidate["codebook_file"]))
    np.save(output / "public_schedule.npy", schedule)
    np.savez_compressed(output / "assessment_errors.npz", errors=errors)
    statistics = measured(errors, weights)
    z_joint = 2.394 if len(weights) > 1 else 1.96
    upper = np.array(statistics["risks"]) + z_joint * np.array(statistics["standard_errors"])
    lower = np.array(statistics["risks"]) - z_joint * np.array(statistics["standard_errors"])
    return dict(feasible_validation=True, ideal=condition["exact"],
                validation_risks=(matrix @ result.x).tolist(),
                validation_expected_payload_rate=float(result.fun),
                payload_rate=float(sum(item["fraction"] * records[index][0]["payload_rate"]
                                       for item, index in zip(transmitted, active))),
                complete_rate=total_bytes * 8 / (assessment_count * records[0][0]["length"]),
                complete_bytes=total_bytes, components=transmitted,
                joint_interval_lower=lower.tolist(), joint_interval_upper=upper.tolist(),
                assessed_feasible=bool(np.all(upper <= condition["tolerances"])), **statistics)


def _origin_audit(source: Path, candidates) -> list[dict]:
    """Compare every confidence-fitted codebook with the matching reduced-fitted one."""
    books = {}
    for candidate in candidates:
        key = (candidate["seed"], candidate["length"], candidate["bits"],
               candidate["law"], candidate["book"])
        books[key] = candidate["codebook_file"]
    audit = []
    for (seed, length, bits, law, book), path in sorted(books.items()):
        if not book.startswith("full_fit"):
            continue
        reduced = books.get((seed, length, bits, law, "reduced_fit"))
        if reduced is None:
            continue
        full_bits = np.load(source / path)["codebook"]
        reduced_bits = np.load(source / reduced)["codebook"]
        audit.append({"codebook_file": path, "reduced_codebook_file": reduced,
                      "changed_bits": int(np.sum(full_bits != reduced_bits)),
                      "bits": int(full_bits.size)})
    return audit


def fixed_codebook_rows(paired, candidates):
    """Aggregate the paired full/reduced comparisons made on each frozen codebook.

    One row per requirement component, holding the range across fits of the
    component risks under both observations, their paired difference, and the
    block standard error. This keeps the per-block error grids out of the
    release while retaining every quantity the appendix reports.
    """
    meta = {candidate["name"]: candidate for candidate in candidates}
    groups: dict[tuple, list[dict]] = {}
    for record in paired:
        candidate = meta[record["candidate"]]
        key = (candidate["length"], candidate["bits"], candidate["law"], candidate["book"],
               candidate["profile_id"], candidate["method"], record["condition"])
        groups.setdefault(key, []).append(record)
    rows = []
    for key, records in sorted(groups.items()):
        length, bits, law, book, profile_id, method, condition = key
        for component in range(len(records[0]["full"]["risks"])):
            full = [r["full"]["risks"][component] for r in records]
            reduced = [r["reduced"]["risks"][component] for r in records]
            difference = [r["full_minus_reduced_risk"][component] for r in records]
            errors = [r["paired_standard_errors"][component] for r in records]
            rows.append({
                "length": length, "bits": bits, "payload_rate": bits / length, "law": law,
                "book": book, "profile_id": profile_id, "method": method,
                "condition": condition, "component": component, "comparisons": len(records),
                "full_risk_min": min(full), "full_risk_max": max(full),
                "reduced_risk_min": min(reduced), "reduced_risk_max": max(reduced),
                "full_minus_reduced_min": min(difference), "full_minus_reduced_max": max(difference),
                "paired_standard_error_min": min(errors), "paired_standard_error_max": max(errors),
            })
    return rows


FIXED_CODEBOOK_FIELDS = (
    "length", "bits", "payload_rate", "law", "book", "profile_id", "method", "condition",
    "component", "comparisons", "full_risk_min", "full_risk_max", "reduced_risk_min",
    "reduced_risk_max", "full_minus_reduced_min", "full_minus_reduced_max",
    "paired_standard_error_min", "paired_standard_error_max",
)


def write_fixed_codebook_csv(path: Path, rows) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(FIXED_CODEBOOK_FIELDS))
        writer.writeheader()
        writer.writerows(rows)


def write_selection_table(rows, path: Path) -> None:
    """Flat view of the selected policies: complete rate, asymptotic rate, excess, and risks."""
    fields = ["condition", "length", "seed", "access", "method", "complete_rate", "ideal_rate",
              "excess_rate", "risk_1", "risk_2", "risk_3", "assessed_feasible"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            if not row["feasible_validation"]:
                continue
            ideal = row["ideal"][f"{row['access']}_rate"]
            entry = {key: row[key] for key in fields[:6]}
            entry.update(ideal_rate=ideal, excess_rate=row["complete_rate"] - ideal,
                         assessed_feasible=row["assessed_feasible"])
            entry.update({f"risk_{i + 1}": risk for i, risk in enumerate(row["risks"])})
            writer.writerow(entry)


def analyze(source: Path, *, safety_margin, keep_paired_records=True):
    """Select mixtures per condition, then compare observations on frozen codebooks."""
    source = source.resolve()
    candidates = json.loads((source / "candidates.json").read_text())
    configuration = json.loads((source / "configuration.json").read_text())
    all_conditions = json.loads((source / "predictions.json").read_text())
    result = []
    for condition in all_conditions:
        for seed in configuration["seeds"]:
            for length in configuration["lengths"]:
                for access, method in [("full", "reference"), ("full", "learned"),
                                       ("reduced", "reference")]:
                    chosen = [c for c in candidates
                              if c["law"] == condition["law"] and c["seed"] == seed
                              and c["length"] == length and c["access"] == access
                              and c["method"] == method]
                    if not chosen:
                        continue
                    folder = (source / "selected" /
                              f"{condition['name']}_s{seed}_n{length}_{access}_{method}")
                    folder.mkdir(parents=True, exist_ok=True)
                    selected = select(chosen, source, condition, folder,
                                      configuration["assessment_count"], safety_margin)
                    row = dict(condition=condition["name"], family=condition["family"],
                               law=condition["law"], seed=seed, length=length, access=access,
                               method=method, tolerances=condition["tolerances"], **selected)
                    result.append(row)
                    (folder / "selection.json").write_text(json.dumps(row, indent=2) + "\n")
    (source / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    write_selection_table(result, source / "manuscript_table.csv")

    paired = []
    lookup = {(c["seed"], c["length"], c["bits"], c["law"], c["book"]): c
              for c in candidates if c["access"] == "reduced"}
    for candidate in candidates:
        if candidate["access"] != "full":
            continue
        reference = lookup[tuple(candidate[k] for k in ["seed", "length", "bits", "law", "book"])]
        full_errors = np.load(source / candidate["error_file"])["assessment"]
        reduced_errors = np.load(source / reference["error_file"])["assessment"]
        for condition in all_conditions:
            if condition["law"] != candidate["law"]:
                continue
            weights = np.asarray(condition["weights"])
            differences = (full_errors - reduced_errors) @ weights.T
            paired.append(dict(
                candidate=candidate["name"], reduced_candidate=reference["name"],
                condition=condition["name"],
                full_minus_reduced_risk=differences.mean(axis=0).tolist(),
                paired_standard_errors=(differences.std(axis=0, ddof=1)
                                        / np.sqrt(len(differences))).tolist(),
                full=measured(full_errors, weights), reduced=measured(reduced_errors, weights)))
    if keep_paired_records:
        (source / "paired_comparisons.json").write_text(json.dumps(paired, indent=2) + "\n")
    write_fixed_codebook_csv(source / "fixed_codebook_comparisons.csv",
                             fixed_codebook_rows(paired, candidates))
    (source / "codebook_origin_audit.json").write_text(
        json.dumps(_origin_audit(source, candidates), indent=2) + "\n")
    print(json.dumps(dict(selections=len(result),
                          assessed_feasible=sum(r.get("assessed_feasible", False) for r in result),
                          paired_comparisons=len(paired))))


def forecast(source: Path, *, device_name):
    """Predict additional-task risks from calibration blocks, then assess without refitting."""
    source = source.resolve()
    configuration = json.loads((source / "configuration.json").read_text())
    candidates = {c["name"]: c for c in json.loads((source / "candidates.json").read_text())}
    selections = json.loads((source / "summary.json").read_text())
    all_conditions = json.loads((source / "predictions.json").read_text())
    device = resolve_device(device_name)
    _prepare(device)
    offset = configuration.get("sample_seed_offset", 0)
    result = []
    for selection in selections:
        if selection["family"] != "single" or not selection["feasible_validation"]:
            continue
        folder = (source / "selected" /
                  f"{selection['condition']}_s{selection['seed']}_n{selection['length']}_"
                  f"{selection['access']}_{selection['method']}")
        schedule = np.load(folder / "public_schedule.npy")
        first_candidate = candidates[selection["components"][0]["candidate"]]
        calibration = np.zeros((len(schedule), len(first_candidate["profile"])), dtype=np.float32)
        frozen_encoders = []
        for component in selection["components"]:
            candidate = candidates[component["candidate"]]
            index = int(Path(component["message_file"]).stem.split("_")[-1])
            selected = schedule == index
            book = torch.tensor(np.load(source / candidate["codebook_file"])["codebook"],
                                dtype=torch.float32, device=device)
            profile = torch.tensor(candidate["profile"], dtype=torch.float32, device=device)
            score = torch.tensor(candidate["learned_profile"], dtype=torch.float32, device=device)
            errors, _, _, _ = evaluate(
                book, profile, offset + 51000 + candidate["length"], len(schedule), device,
                candidate["access"], score if candidate["method"] == "learned" else None)
            calibration[selected] = errors[selected]
            frozen_encoders.append((selected, book, profile, score, candidate))
        np.savez_compressed(folder / "calibration_errors.npz", errors=calibration)
        forecasts = {condition["name"]: measured(calibration, condition["weights"])
                     for condition in all_conditions if condition["law"] == selection["law"]}
        (folder / "frozen_forecasts.json").write_text(json.dumps(forecasts, indent=2) + "\n")
        assessment = np.zeros_like(calibration)
        for selected, book, profile, score, candidate in frozen_encoders:
            errors, _, _, _ = evaluate(
                book, profile, offset + 191000 + candidate["length"], len(schedule), device,
                candidate["access"], score if candidate["method"] == "learned" else None)
            assessment[selected] = errors[selected]
        np.savez_compressed(folder / "forecast_assessment_errors.npz", errors=assessment)
        for condition in all_conditions:
            if condition["law"] != selection["law"]:
                continue
            predicted = forecasts[condition["name"]]
            assessed = measured(assessment, condition["weights"])
            combined_se = np.sqrt(np.array(predicted["standard_errors"]) ** 2
                                  + np.array(assessed["standard_errors"]) ** 2)
            delta = np.array(assessed["risks"]) - np.array(predicted["risks"])
            result.append(dict(
                source_condition=selection["condition"], condition=condition["name"],
                seed=selection["seed"], length=selection["length"], access=selection["access"],
                method=selection["method"], predicted=predicted, assessed=assessed,
                assessment_minus_forecast=delta.tolist(),
                combined_standard_errors=combined_se.tolist(),
                predicted_feasible=bool(np.all(np.array(predicted["risks"])
                                               <= condition["tolerances"])),
                observed_feasible=bool(np.all(np.array(assessed["risks"])
                                              <= condition["tolerances"]))))
    (source / "forecasts.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(forecasts=len(result))))

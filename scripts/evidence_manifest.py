#!/usr/bin/env python3
"""Record what every retained evidence file supports, with its size and hash.

Fails when a retained file has no entry, so the manifest cannot drift behind the
evidence tree.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from taskscope.paths import ASSETS, REPO_ROOT, RESULTS

SUPPORTS = {
    "results/exact/rate_curves.csv":
        "Rate grids for the overlapping-threshold and confidence families. Input to the exact-rate figure.",
    "results/exact/summary.json":
        "Threshold baseline gap, confidence rates at tolerated risk 0.30, and the single-task channel's "
        "second-task risk, with the free-channel verification.",
    "results/exact/profile_numerical_checks.json":
        "Conditional-error-profile criterion checked against a free joint-action optimization, including "
        "the successive three-state rates.",

    "results/block_coding/summary.json":
        "Selected policy per requirement family, fit, block length, observation, and encoder, with complete "
        "rates, component risks, and joint assessment intervals.",
    "results/block_coding/predictions.json":
        "Requirement families and their asymptotic full- and reduced-observation rates.",
    "results/block_coding/forecasts.json":
        "Additional-task risks predicted on calibration blocks and assessed on independent blocks without "
        "refitting.",
    "results/block_coding/manuscript_table.csv":
        "Flat view of the selected policies: complete rate, asymptotic rate, excess, and component risks.",
    "results/block_coding/fixed_codebook_comparisons.csv":
        "Range across fits of the component risks under both observations, their paired difference, and the "
        "block standard error, aggregated from the per-block comparison grid.",
    "results/block_coding/codebook_origin_audit.json":
        "Bit differences between each confidence-fitted codebook and the matching reduced-fitted codebook.",
    "results/block_coding/configuration.json":
        "Parameters of the run that produced this evidence, including the score-rounding correction.",
    "results/block_coding/correction_timing.json":
        "When the corrected configuration was written and how many reference encoders were identical.",
    "results/block_coding/before_score_rounding/precision_audit.json":
        "Validation assignment changes and risk shifts caused by near-tied fitted scores before rounding.",
    "results/block_coding/before_score_rounding/manuscript_table.csv":
        "Selected policies from the run before score rounding, including the original cyclic-family rates.",

    "results/nonbinary/matched_confirmation_reference.json":
        "Matched-risk comparison of the two observations at the confirmation tolerance 0.499: full and "
        "reduced rates, the dual gap interval, and the conditional-KL certificates.",
    "results/nonbinary/learned_selectors/summary.json":
        "Known-cost and fitted-cost selectors on the three frozen codebooks under both observations, "
        "with union-bounded one-sided 99% upper risks, the two controls, and the reduced-observation "
        "minimum. Input to the ternary-code figure.",
    "results/nonbinary/learned_selectors/run.json":
        "Calibration policy, codebook hashes, union size, and tolerances fixed before the fitted-cost "
        "selectors were assessed.",
    "results/nonbinary/learned_selectors/seed17/summary.json":
        "Known and fitted selector records on the seed-17 codebook, with paired risk changes, "
        "cost-rounding error, and the fraction of indices the fitted costs move.",
    "results/nonbinary/learned_selectors/seed17/full_table.npz":
        "Five conditional-cost rows fitted from 2^20 independent noisy labels, with their target "
        "counts and the integer table rounded to multiples of 2^-16, for seed 17.",
    "results/nonbinary/learned_selectors/seed17/reduced_table.npz":
        "The same fitted costs pooled into three class rows for the reduced-observation selector, "
        "for seed 17.",
    "results/nonbinary/learned_selectors/seed23/summary.json":
        "Known and fitted selector records on the seed-23 codebook, with paired risk changes, "
        "cost-rounding error, and the fraction of indices the fitted costs move.",
    "results/nonbinary/learned_selectors/seed23/full_table.npz":
        "Five conditional-cost rows fitted from 2^20 independent noisy labels, with their target "
        "counts and the integer table rounded to multiples of 2^-16, for seed 23.",
    "results/nonbinary/learned_selectors/seed23/reduced_table.npz":
        "The same fitted costs pooled into three class rows for the reduced-observation selector, "
        "for seed 23.",
    "results/nonbinary/learned_selectors/seed31/summary.json":
        "Known and fitted selector records on the seed-31 codebook, with paired risk changes, "
        "cost-rounding error, and the fraction of indices the fitted costs move.",
    "results/nonbinary/learned_selectors/seed31/full_table.npz":
        "Five conditional-cost rows fitted from 2^20 independent noisy labels, with their target "
        "counts and the integer table rounded to multiples of 2^-16, for seed 31.",
    "results/nonbinary/learned_selectors/seed31/reduced_table.npz":
        "The same fitted costs pooled into three class rows for the reduced-observation selector, "
        "for seed 31.",
    "results/nonbinary/confirmation_seed17/summary.json":
        "Independent confirmation of the frozen seed-17 codebook: complete packet length, both component "
        "risks, one-sided 99% bounds, and the margin below the reduced-observation minimum.",
    "results/nonbinary/confirmation_seed17/run.json":
        "Codebook hash, assessment seed, and tolerances recorded before the confirmation blocks were drawn.",
    "results/nonbinary/confirmation_seed17/codebook.npz":
        "The 8,192-word ternary codebook of length 16 that the confirmation and the seed-17 selectors use. "
        "Byte-identical to the development copy under profile_n16_b13/.",
    "results/nonbinary/replication_seed23/summary.json":
        "Seed-23 replication: selected assessment risks, complete rate, and the candidate record.",
    "results/nonbinary/replication_seed23/full_b13.json":
        "Lloyd fitting history, development record, and assessment of the full-observation codebook "
        "for seed 23.",
    "results/nonbinary/replication_seed23/selection.json":
        "Development selection for seed 23, frozen before assessment at tolerances 0.35 and 0.499.",
    "results/nonbinary/replication_seed23/run.json":
        "Configuration, split seeds, and script hash of the seed-23 replication.",
    "results/nonbinary/replication_seed23/full_b13_codebook.npz":
        "The seed-23 ternary codebook the known and fitted selectors are assessed on.",
    "results/nonbinary/replication_seed31/summary.json":
        "Seed-31 replication: selected assessment risks, complete rate, and the candidate record.",
    "results/nonbinary/replication_seed31/full_b13.json":
        "Lloyd fitting history, development record, and assessment of the full-observation codebook "
        "for seed 31.",
    "results/nonbinary/replication_seed31/selection.json":
        "Development selection for seed 31, frozen before assessment at tolerances 0.35 and 0.499.",
    "results/nonbinary/replication_seed31/run.json":
        "Configuration, split seeds, and script hash of the seed-31 replication.",
    "results/nonbinary/replication_seed31/full_b13_codebook.npz":
        "The seed-31 ternary codebook the known and fitted selectors are assessed on.",
    "results/nonbinary/profile_n16_b13/summary.json":
        "First development fit at length 16 and 13 index bits, under both observations.",
    "results/nonbinary/profile_n16_b13/run.json":
        "Configuration of the first development fit. Its source_sha256 570b2691 is an earlier revision of "
        "the fitting code than the 459d2db4 the confirmation and replications record, so the codebook is "
        "pinned by its own hash a95d4558 instead.",
    "results/nonbinary/profile_n16_b13/selection.json":
        "Development selection for the first fit, and the reduced-observation infeasibility at these "
        "tolerances. Its additional tolerance 0.51197 is the dual optimum at the earlier development "
        "multiplier, before the 0.499 tolerance was declared and independently confirmed.",
    "results/nonbinary/profile_n16_b13/full_b13_codebook.npz":
        "The development codebook the seed-17 confirmation freezes.",
    "results/cifar/validation_selection_frozen.json":
        "Validation candidates, paired means, and the six frozen selections. Bound by hash to the test "
        "evidence.",
    "results/cifar/test_extension_summary.json":
        "Test assessment of the frozen selections and the three lower-rate neighbours, with paired image "
        "intervals, complete byte breakdowns, and uncompressed references.",
    "results/cifar/validation_extension_rows.csv":
        "Flat per-codec-fit view of the validation evidence.",
    "results/cifar/test_extension_rows.csv":
        "Flat per-codec-fit view of the test evidence, including header, hyperlatent, and main-latent bytes.",
    "results/cifar/earlier_round/focal_uncertainty.json":
        "Observation and cold-fit controls from the preceding study: refitted fine accuracy and rates for the "
        "early and logit observations and the near-rate preservation settings.",
    "results/cifar/earlier_round/test_fidelity_group_means.json":
        "Normalized early-feature error of the cheaper logit-preserving reconstruction against the "
        "early-preserving one.",
    "results/cifar/earlier_round/coarse_risk_decomposition_validation.json":
        "Validation coarse accuracy of the less favourable cold logit-loss fit against its reference.",

    "results/taskonomy/candidate_pool.csv":
        "Complete twelve-candidate test pool with validation-selected heads, object mIoU, and depth RMSE.",
    "results/taskonomy/focal_means.json":
        "Three-fit means for the focal settings, including complete message lengths and the header fraction.",
    "results/taskonomy/focal_seed_metrics.json":
        "Per-fit test metrics for the focal settings.",
    "results/taskonomy/focal_seed_metrics.csv":
        "Per-fit test metrics for the focal settings, flat view.",
    "results/taskonomy/focal_seed_building_metrics.json":
        "Per-fit, per-building test metrics for the focal settings.",
    "results/taskonomy/focal_seed_building_metrics.csv":
        "Per-fit, per-building test metrics for the focal settings, flat view.",
    "results/taskonomy/focal_building_means.csv":
        "Focal-setting risks after averaging fits within each test building.",
    "results/taskonomy/mixture_selections.json":
        "Validation-selected single codes and input-independent mixtures across family and tolerance, with "
        "their transferred test outcomes.",
    "results/taskonomy/mixture_selections.csv":
        "Validation-selected single codes and mixtures, flat view.",
    "results/taskonomy/semantic_support.json":
        "Semantic class support across validation and test, and the retrospective fixed-class mIoU diagnostic.",
    "results/taskonomy/semantic_support.csv":
        "Per-class semantic support and IoU, flat view.",
    "results/taskonomy/decoder_comparison.json":
        "Matched task decoders against the reconstructed-RGB route at unchanged transmitted bits.",
    "results/taskonomy/retained_paired_contrasts.json":
        "Paired image-bootstrap contrasts retained for the reported candidate comparisons.",
    "results/taskonomy/summary.json":
        "Aggregate of the Taskonomy analysis outputs.",
    "results/taskonomy/provenance.json":
        "Inputs and hashes of the run that produced these tables, naming the retained copy where "
        "one is released and marking the rest external.",
    "results/taskonomy/pool/full_pool_matched_test_summary.json":
        "Test risks and rates for the twelve candidates under matched task decoders.",
    "results/taskonomy/pool/full_pool_matched_test_paired.json":
        "Conditional image-bootstrap intervals for the matched-decoder pool.",
    "results/taskonomy/pool/full_pool_available_test_summary.json":
        "Test risks and rates for the twelve candidates under validation-selected available heads.",
    "results/taskonomy/pool/full_pool_available_validation_summary.json":
        "Validation risks and rates under validation-selected available heads.",
    "results/taskonomy/pool/full_pool_available_validation_choices.json":
        "Validation choice per family and tolerance under available heads.",
    "results/taskonomy/pool/development_tolerances.json":
        "Reference and constant-predictor losses that fix the absolute component requirements.",
    "results/taskonomy/pool/focal_available_validation_selected_heads_test.json":
        "Three-fit focal test results under validation-selected available heads.",
    "results/taskonomy/pool/capacity_sensitivity_summary.json":
        "Validation losses when the fresh decoder width doubles from 48 to 96.",
    "results/taskonomy/pool/capacity_sensitivity_test_summary.json":
        "Test point decisions at the doubled decoder width against the matched width-48 bank.",
    "results/taskonomy/pool/capacity_de_r3_retained_selection.json":
        "Selection record for the wider decoder bank, fixed before test access.",
    "results/taskonomy/reference/validation_evaluation.json":
        "Uncompressed reference predictor on validation.",
    "results/taskonomy/reference/test_evaluation.json":
        "Uncompressed reference predictor on test.",
    "results/taskonomy/available_heads/d_r30/evaluation.json":
        "Depth-preserving focal code at rate multiplier 30, validation evaluation with its selected heads.",
    "results/taskonomy/available_heads/ds_r3/evaluation.json":
        "Depth-semantic focal code at rate multiplier 3, validation evaluation with its selected heads.",
    "results/taskonomy/available_heads/de_r3/evaluation.json":
        "Depth-edge focal code at rate multiplier 3, validation evaluation with its selected heads.",
    "results/taskonomy/examples/focal_readout_example_predictions.npz":
        "Decoded semantic and edge predictions on the five preselected validation images for the three focal "
        "codes. Input to the qualitative figures.",
    "results/taskonomy/examples/focal_readout_example_predictions.json":
        "Image keys, rates, and sources for the decoded example arrays.",

    "results/pilot/pair/heldout_evaluation.json":
        "Species-pair accuracy for each of the two readers fitted per observation, from the 200-way CUB "
        "output restricted to the two species on thirty official test photographs each.",
    "results/pilot/pair/asset_evidence.json":
        "Binds the figure to its evidence: the evaluation it reads by hash, the copied accuracy block, "
        "the verified illustration images, and the file the wing boxes come from.",
    "results/pilot/pair/figure_numbers.json":
        "Accuracies, wing boxes and mark meanings as the motivating figure reports them.",
    "results/pilot/pair/cue_retention.json":
        "Per-photograph measurements inside the marked region. Only the wing boxes are still reported.",
    "results/pilot/pair/sources.json":
        "Bird boxes and provenance for the retained photograph and inversion images.",
}

ASSET_SUPPORT = ("Photograph or RGB inversion shown in the motivating figure. Retained because the "
                 "intermediate inversion renders are not recoverable.")

# The ImageNet strand keeps one directory per fitted candidate and per assessment, so its
# entries are described by what each file is rather than one by one.
IMAGENET_SUPPORT = (
    ("selection_frozen/selection.json",
     "Validation candidates, their paired group means, the frozen decoding route per task, and "
     "the rate-minimal selection per fitting objective. Bound by hash to every assessment."),
    ("assessment_artifact_hashes.json",
     "Digests of the selection, the assessed seals, summaries, and prediction arrays behind the "
     "held-out numbers, including the packet streams that stay external."),
    ("matched_fit_check.json",
     "Confirmation that each coarse-only and added-fine pair shares one initializer state, one "
     "training exposure, and identical fitting arguments apart from the objective."),
    ("data_audit.json",
     "Archive identity, split seed, image counts, and membership digests of the training, "
     "validation, and test splits."),
    ("upstream_mapping_check.json",
     "Comparison of the rebuilt ENTITY-30 mapping against the pinned upstream generator, over "
     "superclass order, every fine index, and the class names."),
    ("hierarchy/",
     "BREEDS hierarchy input mirrored so that mapping.json can be rebuilt without the dataset."),
    ("/test_summary.json",
     "Held-out coarse and fine accuracy of one assessed codec through its frozen route, with the "
     "inherited seal and both adapted readouts."),
    ("/test/sealed.json",
     "Held-out seal: complete byte components, inherited accuracies, and the digests of what the "
     "coding stage wrote."),
    ("/test/predictions.npz",
     "Per-image identifiers, targets, inherited and uncompressed-reference predictions, and the "
     "four byte components of each held-out message."),
    ("_predictions.npy",
     "Per-image adapted readout predictions on the held-out split, one file per readout seed."),
    ("/readouts.json",
     "Selected epoch, validation accuracies, and fitting budget of both adapted readouts for one "
     "candidate."),
    ("_validation.npy",
     "Per-image adapted readout predictions on validation, one file per readout seed."),
    ("/validation/sealed.json",
     "Validation seal: complete byte components, inherited accuracies, and the digests of what "
     "the coding stage wrote."),
    ("/validation/predictions.npz",
     "Per-image identifiers, targets, inherited and uncompressed-reference predictions, and the "
     "four byte components of each validation message."),
    ("/fit.json",
     "Fitting record of one codec: objective, rate weight, seed, initializer identity, image "
     "exposures, and the normalization taken from training images."),
)


def imagenet_support(name: str) -> str | None:
    """Describe one retained file of the ImageNet strand by what kind of record it is."""
    tail = name[len("results/imagenet/"):]
    for pattern, support in IMAGENET_SUPPORT:
        if tail.startswith(pattern) or tail.endswith(pattern):
            return support
    return None


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            sha.update(chunk)
    return sha.hexdigest()


def collect(root: Path, skip: set[str]) -> list[dict]:
    entries = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        name = str(path.relative_to(REPO_ROOT))
        if name in skip:
            continue
        if name.startswith("assets/"):
            support = ASSET_SUPPORT
        elif name.startswith("results/imagenet/"):
            support = imagenet_support(name)
        else:
            support = SUPPORTS.get(name)
        if support is None:
            raise SystemExit(f"No manifest entry for retained file {name}")
        entries.append({"path": name, "bytes": path.stat().st_size,
                        "sha256": digest(path), "supports": support})
    return entries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=RESULTS / "MANIFEST.json")
    arguments = parser.parse_args()
    skip = {str(arguments.output.relative_to(REPO_ROOT))}
    entries = collect(RESULTS, skip) + collect(ASSETS, skip)
    unused = sorted(set(SUPPORTS) - {entry["path"] for entry in entries})
    if unused:
        raise SystemExit(f"Manifest entries with no file: {unused}")
    arguments.output.write_text(json.dumps(
        {"files": len(entries), "bytes": sum(entry["bytes"] for entry in entries),
         "entries": entries}, indent=2) + "\n")
    print(f"{len(entries)} files, {sum(e['bytes'] for e in entries) / 1e6:.2f} MB")


if __name__ == "__main__":
    main()

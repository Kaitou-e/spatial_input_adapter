#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.io import loadmat, savemat
from scipy.ndimage import binary_erosion

"""
python datautils/make_double_block_split.py \
  --input data/IndianPine.mat \
  --output data/IndianPines_patch5_double_split_8020_seed0.mat \
  --block-size 8 \
  --patch-size 5 \
  --train-ratio 0.8 \
  --eval-ratio 0.2 \
  --min-train-per-class 1 \
  --min-eval-per-class 1 \
  --attempts 30000 \
  --seed 0

python datautils/make_double_block_split.py \
  --input data/Houston2013_full.mat \
  --output data/Houston2013_blocks30_seed0.mat \
  --block-size 30 \
  --patch-size 7 \
  --train-ratio 0.8 \
  --eval-ratio 0.2 \
  --min-train-per-class 5 \
  --min-eval-per-class 5 \
  --attempts 20000 \
  --seed 0

python datautils/make_double_block_split.py \
  --input data/Houston2018_comb_full.mat \
  --output data/Houston2013_blocks30_seed0.mat \
  --block-size 30 \
  --patch-size 7 \
  --train-ratio 0.8 \
  --eval-ratio 0.2 \
  --min-train-per-class 5 \
  --min-eval-per-class 5 \
  --attempts 20000 \
  --seed 0

python datautils/make_double_block_split.py \
  --input data/WHU_Hi_HongHu_full.mat \
  --output data/WHU_Hi_HongHu_patch7_seed0.mat \
  --block-size 20 \
  --patch-size 7 \
  --train-ratio 0.8 \
  --eval-ratio 0.2 \
  --min-train-per-class 10 \
  --min-eval-per-class 10 \
  --attempts 50000 \
  --seed 0
"""


def class_counts(
    mask: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    return np.array(
        [
            int((mask == class_id).sum())
            for class_id in range(1, num_classes + 1)
        ],
        dtype=np.int64,
    )


def load_cube_and_ground_truth(
    input_path: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Accept either:
        input + GT
    or:
        input + TR + TE

    In the second case, reconstruct GT from the original masks.
    """

    mat = loadmat(input_path)

    available_keys = sorted(
        key
        for key in mat
        if not key.startswith("__")
    )

    if "input" not in mat:
        raise KeyError(
            "Missing required key 'input'. "
            f"Available keys: {available_keys}"
        )

    cube = np.asarray(mat["input"])

    if "GT" in mat:
        ground_truth = np.asarray(
            mat["GT"],
        ).astype(np.int64)

        print("Using complete ground truth from key 'GT'.")

    elif "TR" in mat and "TE" in mat:
        original_train = np.asarray(
            mat["TR"],
        ).astype(np.int64)

        original_evaluation = np.asarray(
            mat["TE"],
        ).astype(np.int64)

        if original_train.shape != original_evaluation.shape:
            raise ValueError(
                f"Original TR shape {original_train.shape} "
                f"does not match TE shape "
                f"{original_evaluation.shape}."
            )

        exact_overlap = (
            (original_train > 0)
            & (original_evaluation > 0)
        )

        if exact_overlap.any():
            raise ValueError(
                "Original TR and TE masks overlap at "
                f"{int(exact_overlap.sum())} centers."
            )

        ground_truth = np.maximum(
            original_train,
            original_evaluation,
        )

        print(
            "Reconstructed complete ground truth from "
            "the original TR and TE masks."
        )

    else:
        raise KeyError(
            "Input file must contain either 'GT' or both "
            f"'TR' and 'TE'. Available keys: {available_keys}"
        )

    if cube.ndim != 3:
        raise ValueError(
            "'input' must have shape [height, width, bands], "
            f"but received {cube.shape}."
        )

    if ground_truth.ndim != 2:
        raise ValueError(
            "Ground truth must be two-dimensional, "
            f"but received {ground_truth.shape}."
        )

    if cube.shape[:2] != ground_truth.shape:
        raise ValueError(
            f"Cube spatial shape {cube.shape[:2]} does not "
            f"match ground-truth shape {ground_truth.shape}."
        )

    if not np.any(ground_truth > 0):
        raise ValueError(
            "Ground truth contains no labeled pixels."
        )

    return cube, ground_truth


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create spatially separated training and "
            "evaluation masks for an HSI dataset."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help=(
            "MAT file containing input plus either GT, "
            "or the original TR and TE masks."
        ),
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    parser.add_argument(
        "--block-size",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--patch-size",
        type=int,
        default=7,
    )

    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.80,
    )

    parser.add_argument(
        "--eval-ratio",
        type=float,
        default=0.20,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--attempts",
        type=int,
        default=5000,
    )

    parser.add_argument(
        "--min-train-per-class",
        type=int,
        default=5,
        help=(
            "Minimum retained samples of every class "
            "in the training mask."
        ),
    )

    parser.add_argument(
        "--min-eval-per-class",
        type=int,
        default=5,
        help=(
            "Minimum retained samples of every class "
            "in the evaluation mask."
        ),
    )

    args = parser.parse_args()

    ratios = np.array(
        [
            args.train_ratio,
            args.eval_ratio,
        ],
        dtype=np.float64,
    )

    if np.any(ratios <= 0.0):
        raise ValueError(
            "Both train-ratio and eval-ratio must be positive."
        )

    if not np.isclose(ratios.sum(), 1.0):
        raise ValueError(
            "Train and evaluation ratios must sum to 1."
        )

    if args.patch_size <= 0 or args.patch_size % 2 == 0:
        raise ValueError(
            "patch-size must be a positive odd integer."
        )

    if args.block_size <= args.patch_size:
        raise ValueError(
            "block-size must be larger than patch-size."
        )

    if args.attempts <= 0:
        raise ValueError(
            "attempts must be positive."
        )

    if args.min_train_per_class < 0:
        raise ValueError(
            "min-train-per-class cannot be negative."
        )

    if args.min_eval_per_class < 0:
        raise ValueError(
            "min-eval-per-class cannot be negative."
        )

    cube, ground_truth = load_cube_and_ground_truth(
        args.input
    )

    height, width = ground_truth.shape
    num_classes = int(ground_truth.max())
    patch_radius = args.patch_size // 2

    expected_class_ids = np.arange(
        1,
        num_classes + 1,
        dtype=np.int64,
    )

    actual_class_ids = np.unique(
        ground_truth[ground_truth > 0]
    )

    if not np.array_equal(
        actual_class_ids,
        expected_class_ids,
    ):
        raise ValueError(
            "Class IDs must be contiguous from 1 through "
            f"{num_classes}. Found: {actual_class_ids.tolist()}"
        )

    # blocks: list[tuple[int, int, int, int]] = []

    # for row_start in range(
    #     0,
    #     height,
    #     args.block_size,
    # ):
    #     row_end = min(
    #         row_start + args.block_size,
    #         height,
    #     )

    #     for column_start in range(
    #         0,
    #         width,
    #         args.block_size,
    #     ):
    #         column_end = min(
    #             column_start + args.block_size,
    #             width,
    #         )

    #         block_labels = ground_truth[
    #             row_start:row_end,
    #             column_start:column_end,
    #         ]

    #         # Blocks containing no labeled pixels do not affect
    #         # the classification split.
    #         if np.any(block_labels > 0):
    #             blocks.append(
    #                 (
    #                     row_start,
    #                     row_end,
    #                     column_start,
    #                     column_end,
    #                 )
    #             )

    # if not blocks:
    #     raise RuntimeError(
    #         "No blocks containing labeled pixels were found."
    #     )

    print("Cube shape:", cube.shape)
    print("Ground-truth shape:", ground_truth.shape)
    print("Labeled pixels:", int((ground_truth > 0).sum()))
    # print("Labeled blocks:", len(blocks))
    print("Classes:", num_classes)
    print("Patch size:", args.patch_size)
    print("Patch radius:", patch_radius)
    print(
        "Required cross-split center distance:",
        args.patch_size,
    )

    # A center is retained only when the entire patch centered
    # there remains inside the same assigned region.
    erosion_structure = np.ones(
        (
            2 * patch_radius + 1,
            2 * patch_radius + 1,
        ),
        dtype=bool,
    )

    minimum_counts = np.array(
        [
            args.min_train_per_class,
            args.min_eval_per_class,
        ],
        dtype=np.int64,
    )[:, None]

    best_result = None
    best_score = float("inf")

    # Diagnostic information retained even when no fully valid
    # candidate is found.
    best_failed_counts = None
    best_failed_minimum = -1
    valid_candidates = 0

    for attempt in range(args.attempts):
        rng = np.random.default_rng(
            args.seed + attempt
        )

        row_offset = int(
            rng.integers(0, args.block_size)
        )

        column_offset = int(
            rng.integers(0, args.block_size)
        )

        blocks = []

        for row_start in range(
            -row_offset,
            height,
            args.block_size,
        ):
            row_end = min(
                row_start + args.block_size,
                height,
            )

            clipped_row_start = max(
                row_start,
                0,
            )

            if clipped_row_start >= row_end:
                continue

            for column_start in range(
                -column_offset,
                width,
                args.block_size,
            ):
                column_end = min(
                    column_start + args.block_size,
                    width,
                )

                clipped_column_start = max(
                    column_start,
                    0,
                )

                if clipped_column_start >= column_end:
                    continue

                block_labels = ground_truth[
                    clipped_row_start:row_end,
                    clipped_column_start:column_end,
                ]

                if np.any(block_labels > 0):
                    blocks.append(
                        (
                            clipped_row_start,
                            row_end,
                            clipped_column_start,
                            column_end,
                        )
                    )

        # 0 = training, 1 = evaluation
        assignment = rng.choice(
            2,
            size=len(blocks),
            p=ratios,
        )

        regions = [
            np.zeros(
                (height, width),
                dtype=bool,
            )
            for _ in range(2)
        ]

        for split_id, bounds in zip(
            assignment,
            blocks,
        ):
            (
                row_start,
                row_end,
                column_start,
                column_end,
            ) = bounds

            regions[split_id][
                row_start:row_end,
                column_start:column_end,
            ] = True

        # Eroding both regions by the patch radius ensures that
        # patches centered in different splits share no pixels.
        #
        # For patch size 5:
        #     minimum cross-split center distance = 5
        #
        # For patch size 7:
        #     minimum cross-split center distance = 7
        safe_regions = [
            binary_erosion(
                region,
                structure=erosion_structure,
                border_value=1,
            )
            for region in regions
        ]

        split_masks = [
            np.where(
                safe_region,
                ground_truth,
                0,
            ).astype(np.int64)
            for safe_region in safe_regions
        ]

        counts = np.stack(
            [
                class_counts(
                    mask,
                    num_classes,
                )
                for mask in split_masks
            ],
            axis=0,
        )

        weakest_support = int(
            np.min(
                counts - minimum_counts
            )
        )

        if weakest_support > best_failed_minimum:
            best_failed_minimum = weakest_support
            best_failed_counts = counts.copy()

        # Reject candidates that do not meet class-support
        # requirements in either training or evaluation.
        if np.any(counts < minimum_counts):
            continue

        valid_candidates += 1

        totals = counts.sum(axis=1)
        total_retained = int(totals.sum())

        if total_retained == 0:
            continue

        actual_ratios = (
            totals / total_retained
        )

        ratio_error = float(
            np.abs(
                actual_ratios - ratios
            ).sum()
        )

        # Favor similar train/evaluation ratios within each class.
        per_class_totals = counts.sum(axis=0)

        per_class_fractions = (
            counts
            / np.maximum(
                per_class_totals[None, :],
                1,
            )
        )

        class_error = float(
            np.abs(
                per_class_fractions
                - ratios[:, None]
            ).mean()
        )

        # Mildly favor retaining more labeled samples after
        # boundary erosion.
        original_labeled = int(
            (ground_truth > 0).sum()
        )

        retention_fraction = (
            total_retained / original_labeled
        )

        retention_penalty = (
            1.0 - retention_fraction
        )

        score = (
            ratio_error
            + class_error
            + 0.10 * retention_penalty
        )

        if score < best_score:
            best_score = score
            best_result = (
                split_masks,
                counts,
                actual_ratios,
                retention_fraction,
                attempt,
            )

    if best_result is None:
        print(
            "\nNo valid split was found."
        )

        if best_failed_counts is not None:
            print(
                "\nClass counts from the closest candidate:"
            )
            print(
                f"{'Class':>5} "
                f"{'Train':>8} "
                f"{'Eval':>8} "
                f"{'Required TR':>12} "
                f"{'Required EV':>12}"
            )

            for class_id in range(
                1,
                num_classes + 1,
            ):
                print(
                    f"{class_id:5d} "
                    f"{best_failed_counts[0, class_id - 1]:8d} "
                    f"{best_failed_counts[1, class_id - 1]:8d} "
                    f"{args.min_train_per_class:12d} "
                    f"{args.min_eval_per_class:12d}"
                )

        raise RuntimeError(
            "Could not find a two-way spatial split satisfying "
            "the per-class requirements. Try a smaller block "
            "size, smaller minimum class counts, a different "
            "patch size, or more attempts."
        )

    (
        (train_mask, evaluation_mask),
        counts,
        actual_ratios,
        retention_fraction,
        selected_attempt,
    ) = best_result

    print("\nSelected attempt:", selected_attempt)
    print("Valid candidates found:", valid_candidates)
    print("Optimization score:", best_score)
    print("Requested ratios:", ratios)
    print("Actual retained ratios:", actual_ratios)
    print(
        "Retained labeled pixels:",
        f"{100.0 * retention_fraction:.2f}%",
    )

    split_names = (
        "Training",
        "Evaluation",
    )

    for split_name, mask, split_counts in zip(
        split_names,
        (train_mask, evaluation_mask),
        counts,
    ):
        print(
            f"\n{split_name}: "
            f"{int((mask > 0).sum())} centers"
        )

        for class_id, count in enumerate(
            split_counts,
            start=1,
        ):
            print(
                f"  Class {class_id:2d}: "
                f"{count:6d}"
            )

    exact_overlap = (
        (train_mask > 0)
        & (evaluation_mask > 0)
    )

    if exact_overlap.any():
        raise RuntimeError(
            "Internal error: generated TR and TE masks "
            "have exact center overlap."
        )

    output = Path(args.output)
    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    savemat(
        output,
        {
            "input": cube,
            "TR": train_mask,
            # TE is the evaluation mask in this two-way protocol.
            "TE": evaluation_mask,
            "GT": ground_truth,
        },
        do_compression=True,
    )

    print("\nSaved:", output)
    print(
        "Output keys: input, GT, TR, TE "
        "(TE is the evaluation split)."
    )


if __name__ == "__main__":
    main()
#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.io import loadmat, savemat
from scipy.ndimage import binary_erosion

"""
python make_pavia_block_split.py \
  --input ../data/splits/Houston_stratified_80_20_seed0.mat \
  --output ../data/Houston_spatial_blocks_seed0.mat \
  --block-size 20 \
  --patch-size 7 \
  --train-ratio 0.70 \
  --val-ratio 0.15 \
  --test-ratio 0.15 \
  --min-per-class 5 \
  --attempts 500 \
  --seed 0

python make_pavia_block_split.py \
  --input ../data/Houston2013_full.mat \
  --output ../data/Houston2013_blocks30_seed0.mat \
  --block-size 30 \
  --patch-size 7 \
  --train-ratio 0.70 \
  --val-ratio 0.15 \
  --test-ratio 0.15 \
  --min-train-per-class 5 \
  --min-val-per-class 5 \
  --min-test-per-class 5 \
  --attempts 10000 \
  --seed 0

python make_pavia_block_split.py \
  --input ../data/Houston.mat \
  --output ../data/Houston2013_lesstrain_blocks30_seed0.mat \
  --block-size 40 \
  --patch-size 7 \
  --train-ratio 0.15 \
  --val-ratio 0.20 \
  --test-ratio 0.65 \
  --min-train-per-class 5 \
  --min-val-per-class 5 \
  --min-test-per-class 5 \
  --attempts 20000 \
  --seed 0

python make_pavia_block_split.py \
  --input ../data/IndianPine.mat \
  --output ../data/IndianPines_blocks10_seed0.mat \
  --block-size 10 \
  --patch-size 7 \
  --train-ratio 0.70 \
  --val-ratio 0.15 \
  --test-ratio 0.15 \
  --min-per-class 5 \
  --attempts 50000 \
  --seed 0

python datautils/make_pavia_block_split.py \
  --input data/IndianPine.mat \
  --output data/IndianPines_blocks10_seed0.mat \
  --block-size 6 \
  --patch-size 5 \
  --train-ratio 0.70 \
  --val-ratio 0.15 \
  --test-ratio 0.15 \
  --min-train-per-class 3 \
  --min-val-per-class 0 \
  --min-test-per-class 3 \
  --attempts 100000 \
  --seed 0
"""


def class_counts(mask: np.ndarray, num_classes: int) -> np.ndarray:
    return np.array(
        [
            int((mask == class_id).sum())
            for class_id in range(1, num_classes + 1)
        ],
        dtype=np.int64,
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True,
        help="Packaged MAT file containing input, TR, and TE.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--block-size", type=int, default=20)
    parser.add_argument("--patch-size", type=int, default=7)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--attempts", type=int, default=500)
    # parser.add_argument(
    #     "--min-per-class",
    #     type=int,
    #     default=5,
    #     help="Minimum retained examples of every class in every split.",
    # )
    parser.add_argument(
        "--min-train-per-class",
        type=int,
        default=5,
        help="Minimum retained samples of each class in TR.",
    )

    parser.add_argument(
        "--min-val-per-class",
        type=int,
        default=0,
        help=(
            "Minimum retained samples of each class in VA. "
            "Use 0 for spatially concentrated rare classes."
        ),
    )

    parser.add_argument(
        "--min-test-per-class",
        type=int,
        default=5,
        help="Minimum retained samples of each class in TE.",
    )

    args = parser.parse_args()

    ratios = np.array(
        [
            args.train_ratio,
            args.val_ratio,
            args.test_ratio,
        ],
        dtype=np.float64,
    )

    if not np.isclose(ratios.sum(), 1.0):
        raise ValueError("Train, validation, and test ratios must sum to 1.")

    if args.patch_size <= 0 or args.patch_size % 2 == 0:
        raise ValueError("patch-size must be a positive odd integer.")

    if args.block_size <= args.patch_size:
        raise ValueError(
            "block-size should be substantially larger than patch-size."
        )

    mat = loadmat(args.input)

    for key in ("input", "TR", "TE"):
        if key not in mat:
            raise KeyError(f"Missing required key {key!r}.")

    cube = np.asarray(mat["input"])
    original_train = np.asarray(mat["TR"]).astype(np.int64)
    original_test = np.asarray(mat["TE"]).astype(np.int64)

    if ((original_train > 0) & (original_test > 0)).any():
        raise ValueError("Original TR and TE masks overlap.")

    # Recover the complete ground-truth map.
    ground_truth = np.maximum(
        original_train,
        original_test,
    )

    height, width = ground_truth.shape
    num_classes = int(ground_truth.max())
    patch_radius = args.patch_size // 2

    blocks: list[tuple[int, int, int, int]] = []

    for row_start in range(0, height, args.block_size):
        row_end = min(row_start + args.block_size, height)

        for column_start in range(0, width, args.block_size):
            column_end = min(
                column_start + args.block_size,
                width,
            )

            block_labels = ground_truth[
                row_start:row_end,
                column_start:column_end,
            ]

            # Ignore blocks containing no labeled pixels.
            if np.any(block_labels > 0):
                blocks.append(
                    (
                        row_start,
                        row_end,
                        column_start,
                        column_end,
                    )
                )

    print("Cube shape:", cube.shape)
    print("Labeled blocks:", len(blocks))
    print("Classes:", num_classes)
    print("Patch radius:", patch_radius)

    # A center is retained only when its entire patch stays inside
    # its assigned split region.
    erosion_structure = np.ones(
        (
            2 * patch_radius + 1,
            2 * patch_radius + 1,
        ),
        dtype=bool,
    )

    best_result = None
    best_score = float("inf")

    for attempt in range(args.attempts):
        rng = np.random.default_rng(args.seed + attempt)

        assignment = rng.choice(
            3,
            size=len(blocks),
            p=ratios,
        )

        regions = [
            np.zeros((height, width), dtype=bool)
            for _ in range(3)
        ]

        for split_id, bounds in zip(assignment, blocks):
            row_start, row_end, column_start, column_end = bounds

            regions[split_id][
                row_start:row_end,
                column_start:column_end,
            ] = True

        # Erode each region by the patch radius.
        #
        # Adjacent regions are therefore separated by:
        # 3 removed pixels + boundary + 3 removed pixels,
        # producing a minimum center distance of 7.
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
                class_counts(mask, num_classes)
                for mask in split_masks
            ],
            axis=0,
        )

        # Reject candidates that lose a class from any split.
        minimum_counts = np.array(
            [
                args.min_train_per_class,
                args.min_val_per_class,
                args.min_test_per_class,
            ],
            dtype=np.int64,
        )[:, None]

        if np.any(counts < minimum_counts):
            continue

        totals = counts.sum(axis=1)
        actual_ratios = totals / totals.sum()

        ratio_error = np.abs(
            actual_ratios - ratios
        ).sum()

        # Also favor approximately class-balanced split fractions.
        per_class_totals = counts.sum(axis=0)
        per_class_fractions = (
            counts
            / np.maximum(per_class_totals[None, :], 1)
        )

        class_error = np.abs(
            per_class_fractions - ratios[:, None]
        ).mean()

        score = ratio_error + class_error

        if score < best_score:
            best_score = score
            best_result = (
                split_masks,
                counts,
                actual_ratios,
                attempt,
            )

    if best_result is None:
        raise RuntimeError(
            "Could not find a split with every class represented. "
            "Try a smaller --block-size, a smaller "
            "--min-per-class, or more --attempts."
        )

    (
        (train_mask, validation_mask, test_mask),
        counts,
        actual_ratios,
        selected_attempt,
    ) = best_result

    print("\nSelected attempt:", selected_attempt)
    print("Actual retained ratios:", actual_ratios)

    split_names = ("Training", "Validation", "Testing")

    for split_name, mask, split_counts in zip(
        split_names,
        (train_mask, validation_mask, test_mask),
        counts,
    ):
        print(f"\n{split_name}: {int((mask > 0).sum())} centers")

        for class_id, count in enumerate(
            split_counts,
            start=1,
        ):
            print(f"  Class {class_id:2d}: {count:6d}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    savemat(
        output,
        {
            "input": cube,
            "TR": train_mask,
            "VA": validation_mask,
            "TE": test_mask,
            "GT": ground_truth,
        },
        do_compression=True,
    )

    print("\nSaved:", output)


if __name__ == "__main__":
    main()
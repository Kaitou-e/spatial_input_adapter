#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.io import loadmat, savemat
from scipy.ndimage import binary_dilation, distance_transform_cdt


def dilate_square(mask: np.ndarray, radius: int) -> np.ndarray:
    """Expand a boolean mask by a square Chebyshev radius."""

    if radius < 0:
        raise ValueError("radius must be nonnegative.")

    if radius == 0:
        return mask.copy()

    structure = np.ones(
        (2 * radius + 1, 2 * radius + 1),
        dtype=bool,
    )

    return binary_dilation(mask, structure=structure)


def print_test_fraction(
    name: str,
    selected: np.ndarray,
    test_mask: np.ndarray,
) -> None:
    count = int(selected.sum())
    total = int(test_mask.sum())
    percentage = 100.0 * count / total if total else 0.0

    print(
        f"{name:<55} "
        f"{count:>6}/{total:<6} "
        f"({percentage:7.2f}%)"
    )


def print_class_counts(
    name: str,
    label_mask: np.ndarray,
) -> None:
    print(f"\n{name}")

    classes = sorted(
        int(value)
        for value in np.unique(label_mask)
        if value > 0
    )

    for class_id in classes:
        count = int((label_mask == class_id).sum())
        print(f"  Class {class_id:2d}: {count:5d}")


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        required=True,
        help="IndianPine.mat containing input, TR, and TE.",
    )

    parser.add_argument(
        "--patch-size",
        type=int,
        default=7,
    )

    parser.add_argument(
        "--output-prefix",
        default=None,
        help=(
            "Optional prefix for safe-evaluation and "
            "spatially-disjoint MAT files."
        ),
    )

    args = parser.parse_args()

    if args.patch_size <= 0 or args.patch_size % 2 == 0:
        raise ValueError("patch-size must be a positive odd number.")

    mat = loadmat(args.data)

    for key in ("input", "TR", "TE"):
        if key not in mat:
            raise KeyError(f"Missing required MAT key: {key}")

    train_labels = mat["TR"].astype(np.int64)
    test_labels = mat["TE"].astype(np.int64)

    if train_labels.shape != test_labels.shape:
        raise ValueError("TR and TE shapes differ.")

    train = train_labels > 0
    test = test_labels > 0

    patch_radius = args.patch_size // 2

    # For p=7, this is 6.
    patch_overlap_radius = 2 * patch_radius

    print(f"Patch size: {args.patch_size}")
    print(f"Patch radius: {patch_radius}")
    print(
        "Minimum center distance for non-overlapping patches: "
        f"{patch_overlap_radius + 1}"
    )

    print(f"\nTraining centers: {int(train.sum())}")
    print(f"Testing centers:  {int(test.sum())}")

    # 1. Same center in both masks.
    exact_center_overlap = train & test

    # 2. A labeled train center occurs inside the test patch.
    train_centers_within_test_patch = test & dilate_square(
        train,
        patch_radius,
    )

    # 3. Test patch shares at least one raw pixel with any train patch.
    overlapping_test_patches = test & dilate_square(
        train,
        patch_overlap_radius,
    )

    # 4. Test centers whose patches have no pixel overlap with train patches.
    safe_test = test & ~dilate_square(
        train,
        patch_overlap_radius,
    )

    print("\nSpatial-overlap diagnostics")

    print_test_fraction(
        "Exact center overlap",
        exact_center_overlap,
        test,
    )

    print_test_fraction(
        "Test patches containing at least one train center",
        train_centers_within_test_patch,
        test,
    )

    print_test_fraction(
        "Test patches sharing any raw pixel with a train patch",
        overlapping_test_patches,
        test,
    )

    print_test_fraction(
        "Test patches with no train-patch pixel overlap",
        safe_test,
        test,
    )

    # Distance from every location to its nearest train center.
    #
    # distance_transform_cdt measures distance from nonzero entries
    # to the nearest zero. Therefore ~train has zeros at train centers.
    distance_to_train = distance_transform_cdt(
        ~train,
        metric="chessboard",
    )

    test_distances = distance_to_train[test]

    print("\nNearest-training-center distance for test centers")

    if len(test_distances):
        print(f"  Minimum: {int(test_distances.min())}")
        print(f"  Median:  {float(np.median(test_distances)):.2f}")
        print(f"  Mean:    {float(test_distances.mean()):.2f}")
        print(f"  Maximum: {int(test_distances.max())}")

        for threshold in (
            patch_radius,
            patch_overlap_radius,
            10,
            20,
        ):
            count = int((test_distances <= threshold).sum())
            percentage = 100.0 * count / len(test_distances)

            print(
                f"  Distance <= {threshold:2d}: "
                f"{count:5d} ({percentage:7.2f}%)"
            )

    # Safe evaluation subset: keep training unchanged, but evaluate only
    # test centers whose patches do not overlap train patches.
    safe_test_labels = np.where(
        safe_test,
        test_labels,
        0,
    ).astype(np.int64)

    # Proper disjoint retraining version: retain the original test set,
    # but remove every training center whose patch would overlap a test
    # patch.
    train_too_close_to_test = train & dilate_square(
        test,
        patch_overlap_radius,
    )

    disjoint_train_labels = train_labels.copy()
    disjoint_train_labels[train_too_close_to_test] = 0

    print_class_counts(
        "Original training counts",
        train_labels,
    )

    print_class_counts(
        "Training counts after enforcing non-overlap",
        disjoint_train_labels,
    )

    print_class_counts(
        "Safe test subset counts",
        safe_test_labels,
    )

    if args.output_prefix is not None:
        prefix = Path(args.output_prefix)

        base_payload = {
            key: value
            for key, value in mat.items()
            if not key.startswith("__")
        }

        # Diagnostic only: same training data, reduced safe test subset.
        safe_eval_payload = dict(base_payload)
        safe_eval_payload["TE"] = safe_test_labels

        safe_eval_path = prefix.with_name(
            prefix.name + "_safe_eval.mat"
        )

        savemat(safe_eval_path, safe_eval_payload)

        # Properly retrain with overlapping training centers removed.
        disjoint_payload = dict(base_payload)
        disjoint_payload["TR"] = disjoint_train_labels
        disjoint_payload["TE"] = test_labels

        disjoint_path = prefix.with_name(
            prefix.name + "_disjoint.mat"
        )

        savemat(disjoint_path, disjoint_payload)

        print(f"\nSaved diagnostic test file: {safe_eval_path}")
        print(f"Saved disjoint retraining file: {disjoint_path}")


if __name__ == "__main__":
    main()
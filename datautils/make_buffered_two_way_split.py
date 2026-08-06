#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.io import loadmat, savemat
from scipy.ndimage import binary_dilation

"""
python datautils/make_buffered_two_way_split.py \
  --input data/IndianPine.mat \
  --output data/IndianPines_patch5_buffered_seed0.mat \
  --patch-size 5 \
  --eval-ratio 0.20 \
  --min-train-per-class 1 \
  --min-eval-per-class 1 \
  --seed 0
"""

def load_data(path: str):
    mat = loadmat(path)

    if "input" not in mat:
        raise KeyError("Missing key 'input'.")

    cube = np.asarray(mat["input"])

    if "GT" in mat:
        gt = np.asarray(mat["GT"]).astype(np.int64)
    elif "TR" in mat and "TE" in mat:
        tr = np.asarray(mat["TR"]).astype(np.int64)
        te = np.asarray(mat["TE"]).astype(np.int64)

        if ((tr > 0) & (te > 0)).any():
            raise ValueError("Original TR and TE overlap.")

        gt = np.maximum(tr, te)
    else:
        raise KeyError(
            "File requires GT or both TR and TE."
        )

    return cube, gt


def farthest_pair(coords: np.ndarray):
    best_distance = -1
    best_first = None
    best_second = None

    for start in range(0, len(coords), 256):
        first = coords[start:start + 256]

        distances = np.max(
            np.abs(
                first[:, None, :]
                - coords[None, :, :]
            ),
            axis=2,
        )

        flat_index = int(np.argmax(distances))
        local_i, second_i = np.unravel_index(
            flat_index,
            distances.shape,
        )

        distance = int(
            distances[local_i, second_i]
        )

        if distance > best_distance:
            best_distance = distance
            best_first = coords[
                start + local_i
            ].copy()
            best_second = coords[
                second_i
            ].copy()

    return best_distance, best_first, best_second


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--patch-size", type=int, default=5)
    parser.add_argument("--eval-ratio", type=float, default=0.20)
    parser.add_argument("--min-train-per-class", type=int, default=1)
    parser.add_argument("--min-eval-per-class", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    if args.patch_size <= 0 or args.patch_size % 2 == 0:
        raise ValueError(
            "patch-size must be a positive odd number."
        )

    if not 0.0 < args.eval_ratio < 1.0:
        raise ValueError(
            "eval-ratio must be between zero and one."
        )

    cube, gt = load_data(args.input)

    height, width = gt.shape
    num_classes = int(gt.max())
    conflict_radius = args.patch_size - 1

    rng = np.random.default_rng(args.seed)

    train_mask = np.zeros_like(gt)
    eval_mask = np.zeros_like(gt)

    # First reserve one maximally separated train/evaluation
    # anchor pair for every class.
    train_anchors = {}
    eval_anchors = {}

    for class_id in range(1, num_classes + 1):
        coords = np.argwhere(gt == class_id)

        distance, train_anchor, eval_anchor = farthest_pair(
            coords
        )

        if distance < args.patch_size:
            raise RuntimeError(
                f"Class {class_id} cannot be split strictly "
                f"for patch size {args.patch_size}. "
                f"Maximum center distance is {distance}."
            )

        train_anchors[class_id] = train_anchor
        eval_anchors[class_id] = eval_anchor

        train_mask[
            train_anchor[0],
            train_anchor[1],
        ] = class_id

        eval_mask[
            eval_anchor[0],
            eval_anchor[1],
        ] = class_id

    structure = np.ones(
        (
            2 * conflict_radius + 1,
            2 * conflict_radius + 1,
        ),
        dtype=bool,
    )

    # Evaluation centers forbid training centers within
    # distance patch_size - 1.
    forbidden_train = binary_dilation(
        eval_mask > 0,
        structure=structure,
    )

    # Add every non-forbidden labeled center to training initially.
    available_train = (
        (gt > 0)
        & ~forbidden_train
        & ~(eval_mask > 0)
    )

    train_mask[available_train] = gt[available_train]

    # Confirm that every class retained enough training support.
    for class_id in range(1, num_classes + 1):
        train_count = int(
            (train_mask == class_id).sum()
        )

        if train_count < args.min_train_per_class:
            raise RuntimeError(
                f"Class {class_id} has only {train_count} "
                "strictly separated training samples."
            )

    # Grow evaluation class by class. A candidate can be added
    # only when its exclusion buffer leaves enough training data.
    for class_id in range(1, num_classes + 1):
        class_coords = np.argwhere(gt == class_id)
        rng.shuffle(class_coords)

        target_eval = max(
            args.min_eval_per_class,
            int(round(
                len(class_coords)
                * args.eval_ratio
            )),
        )

        for row, column in class_coords:
            if int((eval_mask == class_id).sum()) >= target_eval:
                break

            if eval_mask[row, column] > 0:
                continue

            proposed_eval = eval_mask.copy()
            proposed_eval[row, column] = class_id

            proposed_forbidden = binary_dilation(
                proposed_eval > 0,
                structure=structure,
            )

            proposed_train = np.where(
                (gt > 0)
                & ~proposed_forbidden
                & ~(proposed_eval > 0),
                gt,
                0,
            )

            # Preserve explicitly chosen train anchors.
            anchors_preserved = True

            for anchor_class, anchor in train_anchors.items():
                if proposed_train[
                    anchor[0],
                    anchor[1],
                ] != anchor_class:
                    anchors_preserved = False
                    break

            if not anchors_preserved:
                continue

            class_train_count = int(
                (
                    proposed_train
                    == class_id
                ).sum()
            )

            if (
                class_train_count
                < args.min_train_per_class
            ):
                continue

            eval_mask = proposed_eval
            train_mask = proposed_train

    print("Cube:", cube.shape)
    print("Patch size:", args.patch_size)
    print(
        "Required center separation:",
        args.patch_size,
    )

    for class_id in range(1, num_classes + 1):
        print(
            f"Class {class_id:2d}: "
            f"TR={int((train_mask == class_id).sum()):5d}, "
            f"TE={int((eval_mask == class_id).sum()):5d}"
        )

    # Final overlap assertion.
    train_conflicts = (
        (train_mask > 0)
        & binary_dilation(
            eval_mask > 0,
            structure=structure,
        )
    )

    if train_conflicts.any():
        raise RuntimeError(
            "Generated split contains patch overlap."
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
            "GT": gt,
            "TR": train_mask,
            "TE": eval_mask,
        },
        do_compression=True,
    )

    print("Saved:", output)


if __name__ == "__main__":
    main()
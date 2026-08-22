#!/usr/bin/env python3

from __future__ import annotations

import argparse
from itertools import permutations
from pathlib import Path

import h5py
import numpy as np
from scipy.io import loadmat, savemat

"""
python datautils/package_separate_hsi_gt.py \
  --cube-file data/Houston18/Houston_data.mat \
  --cube-key Houston \
  --gt-file data/Houston18/Houston_gt.mat \
  --gt-key Houston_GT \
  --output data/Houston2018_comb_full.mat


python datautils/package_separate_hsi_gt.py \
  --cube-file data/WHU_Hi_HongHu.mat \
  --cube-key WHU_Hi_HongHu \
  --gt-file data/WHU_Hi_HongHu_gt.mat \
  --gt-key WHU_Hi_HongHu_gt \
  --output data/WHU_Hi_HongHu_full.mat
"""

def read_mat_array(
    path: str,
    key: str,
) -> np.ndarray:
    """
    Read a numeric array from either:
      - a normal MATLAB MAT file;
      - a MATLAB v7.3/HDF5 file.
    """

    try:
        mat = loadmat(path)

        if key not in mat:
            available = {
                name: np.asarray(value).shape
                for name, value in mat.items()
                if not name.startswith("__")
            }

            raise KeyError(
                f"{path} does not contain key {key!r}. "
                f"Available arrays: {available}"
            )

        return np.asarray(mat[key])

    except NotImplementedError:
        with h5py.File(path, "r") as file:
            if key not in file:
                available: dict[str, tuple[int, ...]] = {}

                def collect(name, obj):
                    if isinstance(obj, h5py.Dataset):
                        available[name] = obj.shape

                file.visititems(collect)

                raise KeyError(
                    f"{path} does not contain key {key!r}. "
                    f"Available datasets: {available}"
                )

            dataset = file[key]

            if not isinstance(dataset, h5py.Dataset):
                raise TypeError(
                    f"{key!r} is a group, not a numeric dataset."
                )

            if dataset.dtype.kind not in "biufc":
                raise TypeError(
                    f"{key!r} has unsupported dtype "
                    f"{dataset.dtype}."
                )

            array = np.asarray(dataset)

            # h5py normally exposes MATLAB v7.3 arrays
            # with their dimensions reversed.
            if array.ndim >= 2:
                array = array.transpose(
                    tuple(
                        range(
                            array.ndim - 1,
                            -1,
                            -1,
                        )
                    )
                )

            return array


def orient_cube(
    cube: np.ndarray,
    ground_truth_shape: tuple[int, int],
) -> np.ndarray:
    """
    Find the cube orientation whose first two axes match GT.
    """

    if cube.ndim != 3:
        raise ValueError(
            f"Expected a 3-D cube, received {cube.shape}."
        )

    candidates: list[np.ndarray] = []

    for axes in permutations(range(3)):
        candidate = cube.transpose(axes)

        if candidate.shape[:2] == ground_truth_shape:
            candidates.append(candidate)

    if not candidates:
        raise ValueError(
            f"No orientation of cube shape {cube.shape} "
            f"matches GT shape {ground_truth_shape}."
        )

    # Prefer a candidate with the smallest final dimension,
    # since the spectral dimension is normally smaller than
    # either spatial dimension.
    candidates.sort(
        key=lambda candidate: candidate.shape[-1]
    )

    return candidates[0]


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--cube-file",
        required=True,
    )

    parser.add_argument(
        "--cube-key",
        required=True,
    )

    parser.add_argument(
        "--gt-file",
        required=True,
    )

    parser.add_argument(
        "--gt-key",
        required=True,
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    args = parser.parse_args()

    cube = read_mat_array(
        args.cube_file,
        args.cube_key,
    ).astype(np.float32)

    ground_truth = read_mat_array(
        args.gt_file,
        args.gt_key,
    ).astype(np.int64)

    # Remove singleton dimensions such as [1, H, W].
    ground_truth = np.squeeze(ground_truth)

    if ground_truth.ndim != 2:
        raise ValueError(
            "Ground truth must be two-dimensional after "
            f"squeezing, received {ground_truth.shape}."
        )

    cube = orient_cube(
        cube,
        ground_truth.shape,
    )

    if cube.shape[:2] != ground_truth.shape:
        raise RuntimeError(
            f"Final cube shape {cube.shape} does not match "
            f"GT shape {ground_truth.shape}."
        )

    class_ids = sorted(
        int(value)
        for value in np.unique(ground_truth)
        if value > 0
    )

    if not class_ids:
        raise ValueError(
            "Ground truth contains no positive class labels."
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
            "GT": ground_truth,
        },
        do_compression=True,
    )

    print("Saved:", output)
    print("Cube shape:", cube.shape)
    print("GT shape:", ground_truth.shape)
    print("Labeled pixels:", int((ground_truth > 0).sum()))
    print("Class IDs:", class_ids)
    print("Class count:", len(class_ids))

    print("\nPer-class counts")

    for class_id in class_ids:
        print(
            f"  Class {class_id:2d}: "
            f"{int((ground_truth == class_id).sum()):6d}"
        )


if __name__ == "__main__":
    main()
#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import rasterio
from scipy.io import savemat

"""
python datautils/package_tif_hsi_gt.py \
  --hsi data/WashingtonDC/dc.tif \
  --gt data/WashingtonDC/GT.tif \
  --output data/WashingtonDC_full.mat
"""

def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--hsi",
        required=True,
        help="Multiband hyperspectral GeoTIFF.",
    )

    parser.add_argument(
        "--gt",
        required=True,
        help="Single-band ground-truth GeoTIFF.",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output MATLAB file.",
    )

    args = parser.parse_args()

    # ---------------------------------------------------------
    # Load hyperspectral cube
    # ---------------------------------------------------------

    with rasterio.open(args.hsi) as src:
        print("HSI")
        print("  Raster shape:",
              (src.height, src.width, src.count))
        print("  CRS:", src.crs)
        print("  transform:", src.transform)

        # rasterio returns:
        #
        #   [bands, height, width]
        #
        hsi = src.read()

        hsi_crs = src.crs
        hsi_transform = src.transform
        hsi_height = src.height
        hsi_width = src.width

    # Convert to the format expected by your HyperSL code:
    #
    #   [height, width, bands]
    #
    cube = hsi.transpose(1, 2, 0).astype(np.float32)

    # ---------------------------------------------------------
    # Load ground truth
    # ---------------------------------------------------------

    with rasterio.open(args.gt) as src:
        if src.count != 1:
            raise ValueError(
                f"Expected GT to have one band, "
                f"but {args.gt} has {src.count}."
            )

        gt = src.read(1).astype(np.int64)

        print("\nGT")
        print("  Shape:", gt.shape)
        print("  CRS:", src.crs)
        print("  transform:", src.transform)

        gt_crs = src.crs
        gt_transform = src.transform

    # ---------------------------------------------------------
    # Verify registration
    # ---------------------------------------------------------

    if gt.shape != (hsi_height, hsi_width):
        raise ValueError(
            "HSI and GT dimensions do not match:\n"
            f"  HSI: {(hsi_height, hsi_width)}\n"
            f"  GT:  {gt.shape}"
        )

    if (
        hsi_crs is not None
        and gt_crs is not None
        and hsi_crs != gt_crs
    ):
        raise ValueError(
            f"HSI CRS {hsi_crs} does not match "
            f"GT CRS {gt_crs}."
        )

    if hsi_transform != gt_transform:
        print(
            "\nWARNING: HSI and GT transforms differ."
        )
        print("HSI:", hsi_transform)
        print("GT: ", gt_transform)
        print(
            "Verify that the ground truth is correctly "
            "registered before training."
        )

    # ---------------------------------------------------------
    # Inspect labels
    # ---------------------------------------------------------

    class_ids = sorted(
        int(value)
        for value in np.unique(gt)
        if value > 0
    )

    print("\nFinal cube:", cube.shape)
    print("Final GT:", gt.shape)
    print("Bands:", cube.shape[-1])
    print(
        "Labeled pixels:",
        int((gt > 0).sum()),
    )
    print("Class IDs:", class_ids)
    print("Class count:", len(class_ids))

    print("\nPer-class counts")

    for class_id in class_ids:
        print(
            f"  Class {class_id:2d}: "
            f"{int((gt == class_id).sum()):7d}"
        )

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

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
        },
        do_compression=True,
    )

    print("\nSaved:", output)


if __name__ == "__main__":
    main()
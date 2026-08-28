#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import rasterio
from scipy.io import savemat

"""
source .venv_preprocess/bin/activate

python datautils/package_washington_dc_project.py \
  --hsi data/WashingtonDC/dc.tif \
  --project data/WashingtonDC/dctest.project \
  --output data/WashingtonDC_full.mat
"""

def parse_project(project_path: str):
    text = Path(project_path).read_text(errors="replace")
    lines = text.splitlines()

    p2_line = next(
        (line for line in lines if line.startswith("P2\t")),
        None,
    )
    if p2_line is None:
        raise ValueError("No P2 record found in project file.")

    p2 = p2_line.split("\t")
    height = int(p2[1])
    width = int(p2[2])
    bands = int(p2[3])

    class_names = []
    for line in lines:
        if line.startswith("C1\t"):
            parts = line.split("\t")
            class_names.append(parts[2])

    if not class_names:
        raise ValueError("No C1 class records found.")

    gt = np.zeros((height, width), dtype=np.uint8)

    i = 0
    while i < len(lines):
        if not lines[i].startswith("F1\t"):
            i += 1
            continue

        parts = lines[i].split("\t")

        geometry_type = int(parts[3])
        class_id = int(parts[4])
        n_vertices = int(parts[5])
        expected_pixels = int(parts[7])

        # dctest.project uses two-corner rectangular fields.
        if geometry_type != 2 or n_vertices != 2:
            raise ValueError(
                f"Unsupported field geometry in {parts[2]}: "
                f"type={geometry_type}, vertices={n_vertices}"
            )

        if i + 2 >= len(lines):
            raise ValueError(f"Missing V1 records after {parts[2]}.")

        v1 = [int(x) for x in lines[i + 1].split("\t")[1:3]]
        v2 = [int(x) for x in lines[i + 2].split("\t")[1:3]]

        r1, c1 = v1
        r2, c2 = v2

        rlo, rhi = sorted((r1, r2))
        clo, chi = sorted((c1, c2))

        if not (1 <= rlo <= rhi <= height):
            raise ValueError(f"Row coordinates outside image in {parts[2]}.")
        if not (1 <= clo <= chi <= width):
            raise ValueError(f"Column coordinates outside image in {parts[2]}.")

        # MultiSpec project coordinates are 1-based and inclusive.
        rs = slice(rlo - 1, rhi)
        cs = slice(clo - 1, chi)

        area = (rhi - rlo + 1) * (chi - clo + 1)
        if area != expected_pixels:
            raise ValueError(
                f"{parts[2]} says {expected_pixels} pixels, "
                f"but coordinates define {area}."
            )

        if np.any(gt[rs, cs] != 0):
            raise ValueError(f"Overlapping fields detected in {parts[2]}.")

        gt[rs, cs] = class_id
        i += 3

    return gt, class_names, bands


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Package Washington DC Mall dc.tif with the "
            "7-class labels stored in dctest.project."
        )
    )
    parser.add_argument("--hsi", required=True, help="dc.tif")
    parser.add_argument("--project", required=True, help="dctest.project")
    parser.add_argument("--output", required=True, help="Output .mat file")
    args = parser.parse_args()

    gt, class_names, expected_bands = parse_project(args.project)

    with rasterio.open(args.hsi) as src:
        if src.height != gt.shape[0] or src.width != gt.shape[1]:
            raise ValueError(
                "HSI/project spatial mismatch: "
                f"HSI={(src.height, src.width)}, GT={gt.shape}"
            )

        if src.count != expected_bands:
            raise ValueError(
                "HSI/project band mismatch: "
                f"HSI={src.count}, project={expected_bands}"
            )

        cube = src.read().transpose(1, 2, 0).astype(np.float32)

    print("Cube:", cube.shape)
    print("GT:", gt.shape)
    print("Classes:", len(class_names))

    for class_id, name in enumerate(class_names, start=1):
        print(
            f"  {class_id}: {name:<8} "
            f"{int((gt == class_id).sum())} pixels"
        )

    print("Total labeled:", int((gt > 0).sum()))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    savemat(
        output,
        {
            "input": cube,
            "GT": gt,
            "class_names": np.array(class_names, dtype=object),
        },
        do_compression=True,
    )

    print("Saved:", output)


if __name__ == "__main__":
    main()

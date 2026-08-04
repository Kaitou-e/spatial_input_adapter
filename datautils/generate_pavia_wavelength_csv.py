# export DATA_PATH="$PWD/data/PaviaU_stratified_10_90_seed0.mat"
# export WAVELENGTHS_PATH="$PWD/data/pavia_u_wavelengths_approx.csv"

# python - <<'PY'
import os

import numpy as np
import pandas as pd
from scipy.io import loadmat

data_path = "/home/lxdcis/hypersl_training/data/Pavia_spatial_blocks_seed0.mat"
output_path = "/home/lxdcis/hypersl_training/data/pavia_wavelengths_approx.csv"

mat = loadmat(data_path)

if "input" not in mat:
    raise KeyError(
        f"{data_path} does not contain an 'input' array."
    )

cube = mat["input"]

if cube.ndim != 3:
    raise ValueError(
        f"Expected [height, width, bands], got {cube.shape}."
    )

num_bands = cube.shape[-1]

wavelengths = np.linspace(
    430.0,
    860.0,
    num_bands,
    dtype=np.float32,
)

pd.DataFrame(
    {
        "band": np.arange(1, num_bands + 1),
        "wavelength_nm": wavelengths,
    }
).to_csv(output_path, index=False)

print("Cube shape:", cube.shape)
print("Bands:", num_bands)
print("Saved:", output_path)
print(
    "WARNING: These wavelengths are approximate, "
    "not exact ROSIS calibration values."
)
# PY
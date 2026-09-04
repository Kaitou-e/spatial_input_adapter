# Spatial Input Adapter for Hyperspectral Image Classification

Official implementation of **Beyond One Pixel: Parameter-Efficient Spatial Context for Hyperspectral Image Classification**.

The **Spatial Input Adapter (SIA)** is a lightweight parameter-efficient method for incorporating local spatial context into a pretrained spectral foundation model. SIA learns a weighted combination of neighboring hyperspectral pixels before passing the resulting spectrum through a frozen HyperSL encoder.

For a `p × p` spatial patch, SIA introduces only `p²` additional trainable parameters while leaving the pretrained HyperSL backbone frozen.

## Method

Standard HyperSL linear probing classifies a target pixel using only its spectral signature. SIA instead operates on a local spatial patch centered on the target pixel.

Given center spectrum $X_{\mathrm{center}}$ and neighboring spectra $X_{i,j}$, SIA computes

$$
X_{\mathrm{context}}
=
\sum_{(i,j)\neq\mathrm{center}}
w_{i,j} X_{i,j}
$$

and produces the adapted spectrum

$$
X_{\mathrm{mixed}}
=
(1-\alpha)X_{\mathrm{center}}
+
\alpha X_{\mathrm{context}}.
$$

The spatial weights are learned using a softmax over the non-center positions, while $\alpha$ is a learned scalar mixing coefficient.

The resulting spectrum is passed through the frozen HyperSL-Small encoder followed by a trainable linear classifier.

## Results

We evaluate SIA against HyperSL linear probing on six hyperspectral image classification benchmarks using spatially separated data splits.

| Dataset | Method | AA | OA | Kappa |
|---|---|---:|---:|---:|
| Botswana | Linear Probe | 0.8637 | 0.8852 | 0.8726 |
|  | **SIA** | **0.9268** | **0.9617** | **0.9574** |
| Chikusei | Linear Probe | 0.9643 | 0.9644 | 0.9592 |
|  | **SIA** | **0.9930** | **0.9865** | **0.9845** |
| Indian Pines | Linear Probe | 0.7704 | 0.8209 | 0.7901 |
|  | **SIA** | **0.8866** | **0.9007** | **0.8838** |
| Pavia Center | Linear Probe | 0.8868 | 0.9064 | 0.8689 |
|  | **SIA** | **0.9793** | **0.9830** | **0.9762** |
| Salinas | Linear Probe | 0.9477 | 0.8883 | 0.8746 |
|  | **SIA** | **0.9667** | **0.9303** | **0.9215** |
| WHU-Hi HongHu | Linear Probe | 0.6494 | 0.8327 | 0.7830 |
|  | **SIA** | **0.8348** | **0.9240** | **0.9022** |

Across the six datasets, SIA improves AA, OA, and Cohen's Kappa by an average of **8.42**, **6.47**, and **7.95 percentage points**, respectively.

## Installation

Clone the repository:

```bash
git clone https://github.com/Kaitou-e/spatial_input_adapter.git
cd spatial_input_adapter
```

Create a Python environment and install the required packages:

```bash
python -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

### HyperSL

This project uses the pretrained **HyperSL-Small** spectral foundation model.

HyperSL and the pretrained HyperSL-Small checkpoint are not distributed with this repository. Obtain and install HyperSL separately following the official HyperSL instructions, and download the pretrained HyperSL-Small checkpoint before running the experiments.

## Datasets

The datasets used in the paper are:

- Botswana
- Chikusei
- Indian Pines
- Pavia Center
- Salinas
- WHU-Hi HongHu

The hyperspectral datasets are **not distributed with this repository**. Please obtain them from their original providers.

Some datasets may not currently have permanent public download links, so automatic dataset downloading is not provided.

Each experiment requires:

- Hyperspectral image data
- Ground-truth labels
- Corresponding wavelength CSV
- Spatial split
- HyperSL-Small pretrained checkpoint

Dataset and split preparation utilities are available under:

```text
datautils/
```

These include scripts for constructing spatially separated splits and checking for spatial leakage between partitions.

## Running Experiments

Two main scripts are provided:

```text
run_linear_probe.sh
run_sia.sh
```

Before running either script, edit the configuration variables in the script to point to your local files.

At minimum, specify paths for:

```text
DATA_PATH
WAVELENGTHS_PATH
CHECKPOINT_PATH
OUTPUT_PATH
```

Additional dataset-specific variables may also need to be changed.

### Linear Probe

Run the HyperSL linear-probing baseline with:

```bash
bash run_linear_probe.sh
```

Linear probing uses only the center-pixel spectrum and trains a linear classification head while keeping the HyperSL encoder frozen.

### Spatial Input Adapter

Run SIA with:

```bash
bash run_sia.sh
```

SIA trains the Spatial Input Adapter and linear classifier while keeping the pretrained HyperSL encoder frozen.

## Spatial Splits

To reduce spatial leakage, the experiments use spatially separated partitions rather than randomly splitting individual labeled pixels.

Contiguous regions are assigned to separate partitions, and patch boundaries are constrained so that samples from different partitions do not share underlying image pixels.

Utilities for generating and validating these splits are located in:

```text
datautils/
```

Indian Pines uses a two-way spatial split because its small and spatially concentrated classes make a three-way spatial split impractical while retaining all 16 classes.

## Repository Structure

```text
.
├── run_linear_probe.sh        # HyperSL linear probing
├── run_sia.sh                 # Spatial Input Adapter training
├── hypersl_linear_probe.py
├── requirements.txt
│
├── datautils/                 # Dataset packaging, spatial split, and visualization utilities
├── engine/                    # Model/training components
├── hypersl/                   # HyperSL-related code
├── notebooks/                 # Experimental notebooks
└── scripts/                   # Additional experiment scripts
```

## Training Setup

Experiments in the paper use:

- HyperSL-Small pretrained encoder
- AdamW optimizer
- Weight decay: `1e-4`
- Linear-probe learning rate: `1e-4`
- SIA learning rate: `1e-3`
- SIA classifier learning rate: `3e-4`
- Initial spatial mixing coefficient: `α = 0.7`

The pretrained HyperSL encoder remains frozen during both linear probing and SIA training.

## Citation

If you use this code, please cite our paper.

Please also cite the original HyperSL paper and the corresponding publications for any hyperspectral datasets used in your experiments.

## Acknowledgments

This work builds on the HyperSL spectral foundation model.

This work was supported by the National Science Foundation CAREER Award No. 2542166.
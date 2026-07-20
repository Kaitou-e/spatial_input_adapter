from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.io import loadmat
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import minmax_scale

from mae_viz.hsi import choose_sample_indices, format_label

from .checkpoints import HyperSLBundle, display_path


CLASS_MARKERS = ["o", "s", "^", "D", "P", "X", "v", "<"]


def default_wavelength_range(dataset: str) -> tuple[float, float]:
    dataset_lower = dataset.lower()
    if dataset_lower == "pavia":
        return 430.0, 860.0
    if dataset_lower == "houston":
        return 380.0, 1050.0
    return 400.0, 2500.0


def load_wavelengths(path: str, dataset: str, num_bands: int, wave_min=None, wave_max=None) -> tuple[np.ndarray, bool]:
    if not path:
        lo, hi = default_wavelength_range(dataset)
        if wave_min is not None:
            lo = float(wave_min)
        if wave_max is not None:
            hi = float(wave_max)
        return np.linspace(lo, hi, num_bands, dtype=np.float32), False

    path_obj = Path(path)
    if path_obj.suffix == ".npy":
        wavelengths = np.load(path_obj)
    else:
        delimiter = "," if path_obj.suffix == ".csv" else None
        wavelengths = np.loadtxt(path_obj, delimiter=delimiter)
        if wavelengths.ndim > 1:
            wavelengths = wavelengths[:, -1]
    wavelengths = np.asarray(wavelengths, dtype=np.float32).reshape(-1)
    if wavelengths.shape[0] != num_bands:
        raise ValueError(f"Expected {num_bands} wavelengths, got {wavelengths.shape[0]} from {path}.")
    return wavelengths, True


def load_hsi_data(dataset: str, data_path: str | Path):
    path = Path(data_path)
    mat = loadmat(path)
    missing = {"input", "TR", "TE"} - set(mat.keys())
    if missing:
        raise KeyError(f"{path} is missing required keys: {sorted(missing)}")
    cube = mat["input"].astype(np.float32)
    flat = cube.reshape(-1, cube.shape[-1])
    flat = minmax_scale(flat)
    cube = flat.reshape(cube.shape).astype(np.float32)
    train_mask = mat["TR"].astype(np.int64)
    test_mask = mat["TE"].astype(np.int64)
    label_map = train_mask.copy()
    label_map[test_mask > 0] = test_mask[test_mask > 0]
    height, width, bands = cube.shape
    grid_y, grid_x = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    all_points = np.stack([grid_y.reshape(-1), grid_x.reshape(-1)], axis=1).astype(int)
    flat_labels = label_map.reshape(-1)
    labeled_indices = np.flatnonzero(flat_labels > 0)
    return {
        "dataset": dataset,
        "path": path,
        "cube": cube,
        "shape": cube.shape,
        "height": height,
        "width": width,
        "bands": bands,
        "label_map": label_map,
        "flat_labels": flat_labels,
        "labeled_indices": labeled_indices,
        "all_points": all_points,
    }


def spectra_for_points(cube: np.ndarray, points: np.ndarray) -> np.ndarray:
    return cube[points[:, 0], points[:, 1], :][:, None, :].astype(np.float32)


def wave_batch(wavelengths: np.ndarray, batch_size: int) -> torch.Tensor:
    return torch.from_numpy(np.repeat(wavelengths[None, :], batch_size, axis=0)).float()


def encode_spectra(
    model: torch.nn.Module,
    spectra_np: np.ndarray,
    wavelengths: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    tensor = torch.from_numpy(spectra_np).float()
    outputs = []
    with torch.no_grad():
        for start in range(0, tensor.shape[0], batch_size):
            batch = tensor[start : start + batch_size].to(device)
            wave = wave_batch(wavelengths, batch.shape[0]).to(device)
            z, *_ = model.encoder_forward(batch, wave, 0.0)
            outputs.append(z[:, 0, :].cpu().numpy())
    return np.concatenate(outputs, axis=0)


def reconstruct_spectra(
    model: torch.nn.Module,
    spectra_np: np.ndarray,
    wavelengths: np.ndarray,
    mask_ratio: float,
    batch_size: int,
    device: torch.device,
    seed: int,
) -> tuple[np.ndarray, float]:
    tensor = torch.from_numpy(spectra_np).float()
    outputs = []
    losses = []
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    with torch.no_grad():
        for start in range(0, tensor.shape[0], batch_size):
            batch = tensor[start : start + batch_size].to(device)
            wave = wave_batch(wavelengths, batch.shape[0]).to(device)
            _, recon = model(batch, wave, mask_ratio)
            outputs.append(recon.cpu().numpy())
            losses.append(float(((recon - batch) ** 2).mean().item()))
    return np.concatenate(outputs, axis=0), float(np.mean(losses))


def scatter_embeddings_by_label(ax, coordinates, labels, dataset, display_labels, point_size=24, alpha=0.78):
    handles = []
    colors = plt.get_cmap("tab20", display_labels.shape[0])
    for class_index, label in enumerate(display_labels):
        class_mask = labels == label
        if not np.any(class_mask):
            continue
        handles.append(
            ax.scatter(
                coordinates[class_mask, 0],
                coordinates[class_mask, 1],
                s=point_size,
                alpha=alpha,
                linewidths=0,
                color=colors(class_index),
                marker=CLASS_MARKERS[class_index % len(CLASS_MARKERS)],
                label=format_label(dataset, int(label)),
            )
        )
    return handles


def save_embedding_plots(
    embeddings,
    labels,
    dataset,
    model_name,
    output_dir: Path,
    run_tsne,
    tsne_sample_size,
    tsne_min_points_per_class,
    tsne_perplexity,
    tsne_pre_pca_dim,
    seed,
    rng,
) -> dict:
    display_labels = np.asarray(np.unique(labels), dtype=int)
    outputs = {}
    emb_pca = PCA(n_components=2, random_state=seed).fit_transform(embeddings)

    if run_tsne:
        take = min(tsne_sample_size, embeddings.shape[0])
        sel = choose_sample_indices(labels, take, rng, min_points_per_class=tsne_min_points_per_class)
        tsne_embeddings = embeddings[sel]
        tsne_labels = labels[sel]
        pca_dim = min(tsne_pre_pca_dim, tsne_embeddings.shape[1], tsne_embeddings.shape[0] - 1)
        if 1 <= pca_dim < tsne_embeddings.shape[1]:
            tsne_input = PCA(n_components=pca_dim, random_state=seed).fit_transform(tsne_embeddings)
        else:
            tsne_input = tsne_embeddings
        perplexity = max(2, min(tsne_perplexity, tsne_input.shape[0] - 1))
        emb_tsne = TSNE(
            n_components=2,
            init="pca",
            learning_rate="auto",
            perplexity=perplexity,
            random_state=seed,
        ).fit_transform(tsne_input)
        fig, axes = plt.subplots(1, 2, figsize=(19, 7), squeeze=False)
        axes = axes.ravel()
        handles = scatter_embeddings_by_label(axes[0], emb_pca, labels, dataset, display_labels)
        axes[0].set_title(f"{model_name}\nHyperSL Embeddings (PCA)")
        axes[0].set_xlabel("PC 1")
        axes[0].set_ylabel("PC 2")
        axes[0].grid(alpha=0.2)
        scatter_embeddings_by_label(axes[1], emb_tsne, tsne_labels, dataset, display_labels, alpha=0.85)
        axes[1].set_title(f"{model_name}\nHyperSL Embeddings (t-SNE)")
        axes[1].set_xlabel("t-SNE 1")
        axes[1].set_ylabel("t-SNE 2")
        axes[1].grid(alpha=0.2)
        if handles:
            fig.legend(handles=handles, loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)
        fig.tight_layout(rect=[0, 0, 0.88, 1])
        output = output_dir / "hypersl_embeddings_pca_tsne.png"
        fig.savefig(output, dpi=180, bbox_inches="tight")
        plt.close(fig)
        outputs["embeddings_pca_tsne"] = str(output)
    else:
        fig, ax = plt.subplots(figsize=(10, 7))
        handles = scatter_embeddings_by_label(ax, emb_pca, labels, dataset, display_labels)
        ax.set_title(f"{model_name}\nHyperSL Embeddings (PCA)")
        ax.set_xlabel("PC 1")
        ax.set_ylabel("PC 2")
        ax.grid(alpha=0.2)
        if handles:
            ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
        fig.tight_layout()
        output = output_dir / "hypersl_embeddings_pca.png"
        fig.savefig(output, dpi=180, bbox_inches="tight")
        plt.close(fig)
        outputs["embeddings_pca"] = str(output)
    return outputs


def save_reconstruction_examples(bundle, data, wavelengths, output_dir, sample_size, batch_size, device, seed, rng):
    take = min(sample_size, len(data["labeled_indices"]))
    chosen = np.sort(rng.choice(data["labeled_indices"], size=take, replace=False))
    points = data["all_points"][chosen]
    labels = data["flat_labels"][chosen]
    spectra = spectra_for_points(data["cube"], points)
    recon, masked_loss = reconstruct_spectra(
        bundle.model,
        spectra,
        wavelengths,
        mask_ratio=float(bundle.mask_ratio),
        batch_size=batch_size,
        device=device,
        seed=seed,
    )
    ncols = 2
    nrows = int(np.ceil(take / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows), squeeze=False)
    axes = axes.ravel()
    for ax, point, label, original, predicted in zip(axes, points, labels, spectra, recon):
        ax.plot(wavelengths, original.squeeze(0), label="Original", linewidth=2)
        ax.plot(wavelengths, predicted.squeeze(0), label=f"Recon (mask={bundle.mask_ratio:.2f})", linewidth=1.4)
        ax.set_title(f"Pixel ({point[0]}, {point[1]}) | {format_label(data['dataset'], int(label))}")
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Normalized signal")
        ax.legend(loc="best", fontsize=8)
    for ax in axes[take:]:
        ax.axis("off")
    fig.suptitle(f"{bundle.name} | mean reconstruction MSE={masked_loss:.6f}", y=1.02)
    fig.tight_layout()
    output = output_dir / "hypersl_reconstruction_examples.png"
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return {"reconstruction_examples": str(output), "masked_reconstruction_mse": masked_loss}


def save_scene_error(bundle, data, wavelengths, output_dir, batch_size, device):
    spectra = spectra_for_points(data["cube"], data["all_points"])
    recon, _ = reconstruct_spectra(
        bundle.model,
        spectra,
        wavelengths,
        mask_ratio=0.0,
        batch_size=batch_size,
        device=device,
        seed=0,
    )
    mse_map = ((recon - spectra) ** 2).mean(axis=(1, 2)).reshape(data["shape"][:2])
    mse_norm = minmax_scale(mse_map.reshape(-1)).reshape(mse_map.shape)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    im = axes[0].imshow(mse_norm, cmap="magma", vmin=0.0, vmax=1.0)
    axes[0].set_title("HyperSL Reconstruction MSE\n(min-max normalized)")
    axes[0].axis("off")
    plt.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)
    axes[1].hist(mse_norm.reshape(-1), bins=100, range=(0, 1), color="steelblue", alpha=0.9)
    axes[1].set_xlim(0, 1)
    axes[1].set_title("Normalized Reconstruction Error Distribution")
    axes[1].set_xlabel("Min-max normalized MSE")
    axes[1].set_ylabel("Pixel count")
    fig.tight_layout()
    output = output_dir / "hypersl_reconstruction_mse_map.png"
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    np.save(output_dir / "hypersl_reconstruction_mse_map.npy", mse_norm)
    return {
        "reconstruction_mse_map": str(output),
        "mean_mse": float(mse_map.mean()),
        "median_mse": float(np.median(mse_map)),
        "p95_mse": float(np.quantile(mse_map, 0.95)),
    }


def generate_report(args, bundle: HyperSLBundle, repo_root: Path, device: torch.device) -> dict:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    data = load_hsi_data(args.dataset, args.data_path)
    wavelengths, has_real_wavelengths = load_wavelengths(
        args.wavelengths_path,
        args.dataset,
        data["bands"],
        wave_min=args.wave_min,
        wave_max=args.wave_max,
    )

    summary_rows = [
        {
            "name": bundle.name,
            "checkpoint": display_path(bundle.checkpoint_path, repo_root),
            "embedding_dim": bundle.config["embedding_dim"],
            "encoder_depth": bundle.config["encoder_depth"],
            "decoder_depth": bundle.config["decoder_depth"],
            "num_heads": bundle.config["num_heads"],
            "mask_ratio": bundle.mask_ratio,
            "has_real_wavelengths": has_real_wavelengths,
        }
    ]
    pd.DataFrame(summary_rows).to_csv(output_dir / "hypersl_model_summary.csv", index=False)

    pool = data["labeled_indices"]
    take = min(args.embed_sample_size, len(pool))
    pool_labels = data["flat_labels"][pool]
    sel = choose_sample_indices(
        pool_labels,
        sample_size=take,
        rng=rng,
        min_points_per_class=args.embed_min_points_per_class,
    )
    embed_idx = pool[sel]
    embed_points = data["all_points"][embed_idx]
    embed_labels = data["flat_labels"][embed_idx]
    embed_spectra = spectra_for_points(data["cube"], embed_points)
    embeddings = encode_spectra(bundle.model, embed_spectra, wavelengths, args.embed_batch_size, device)
    outputs = save_embedding_plots(
        embeddings=embeddings,
        labels=embed_labels,
        dataset=args.dataset,
        model_name=bundle.name,
        output_dir=output_dir,
        run_tsne=args.run_tsne,
        tsne_sample_size=args.tsne_sample_size,
        tsne_min_points_per_class=args.tsne_min_points_per_class,
        tsne_perplexity=args.tsne_perplexity,
        tsne_pre_pca_dim=args.tsne_pre_pca_dim,
        seed=args.seed,
        rng=rng,
    )
    outputs.update(
        save_reconstruction_examples(
            bundle,
            data,
            wavelengths,
            output_dir,
            args.recon_sample_size,
            args.recon_batch_size,
            device,
            args.seed,
            rng,
        )
    )
    if args.skip_scene:
        scene_skipped = True
    else:
        outputs.update(save_scene_error(bundle, data, wavelengths, output_dir, args.scene_batch_size, device))
        scene_skipped = False

    report = {
        "dataset": args.dataset,
        "data_path": str(Path(args.data_path)),
        "data_shape": list(data["shape"]),
        "labeled_pixels": int(len(data["labeled_indices"])),
        "embedding_sample_size": int(take),
        "device": str(device),
        "model": summary_rows[0],
        "outputs": outputs,
        "scene_skipped": scene_skipped,
    }
    summary_json = output_dir / "hypersl_summary.json"
    summary_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    report["summary_json"] = str(summary_json)
    return report

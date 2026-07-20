from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .checkpoints import find_repo_root, load_bundle


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("HyperSL visualization report")
    parser.add_argument("--dataset", required=True, help="Dataset name, e.g. Indian, Pavia, Houston.")
    parser.add_argument("--data-path", required=True, help="MAT file with input, TR, and TE.")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint path, filename, or unique substring.")
    parser.add_argument("--output-dir", required=True, help="Directory where report figures are saved.")
    parser.add_argument("--wavelengths-path", default="", help="Optional .npy/.txt/.csv wavelength file.")
    parser.add_argument("--wave-min", type=float, default=None, help="Placeholder wavelength start.")
    parser.add_argument("--wave-max", type=float, default=None, help="Placeholder wavelength end.")
    parser.add_argument("--name", default="", help="Display name for plots.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or torch device string.")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--embedding-dim", type=int, default=None)
    parser.add_argument("--encoder-depth", type=int, default=None)
    parser.add_argument("--decoder-depth", type=int, default=None)
    parser.add_argument("--num-heads", type=int, default=None)
    parser.add_argument("--mask-ratio", type=float, default=None)

    parser.add_argument("--embed-sample-size", type=int, default=3000)
    parser.add_argument("--embed-min-points-per-class", type=int, default=32)
    parser.add_argument("--embed-batch-size", type=int, default=512)
    parser.add_argument("--recon-sample-size", type=int, default=6)
    parser.add_argument("--recon-batch-size", type=int, default=256)
    parser.add_argument("--scene-batch-size", type=int, default=256)
    parser.add_argument("--skip-scene", action="store_true")
    parser.add_argument("--run-tsne", action="store_true")
    parser.add_argument("--tsne-sample-size", type=int, default=3000)
    parser.add_argument("--tsne-min-points-per-class", type=int, default=32)
    parser.add_argument("--tsne-perplexity", type=int, default=30)
    parser.add_argument("--tsne-pre-pca-dim", type=int, default=50)
    return parser


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def main() -> None:
    args = build_parser().parse_args()
    repo_root = find_repo_root()
    device = resolve_device(args.device)
    bundle = load_bundle(
        checkpoint=args.checkpoint,
        repo_root=repo_root,
        device=device,
        name=args.name or None,
        embedding_dim=args.embedding_dim,
        encoder_depth=args.encoder_depth,
        decoder_depth=args.decoder_depth,
        num_heads=args.num_heads,
        mask_ratio=args.mask_ratio,
    )

    from .report import generate_report

    report = generate_report(args, bundle, repo_root=repo_root, device=device)
    print(f"HyperSL visualization report saved to {report['summary_json']}")


if __name__ == "__main__":
    main()


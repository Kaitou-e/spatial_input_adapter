from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

import torch

from hypersl.engine.model import SpectralSharedEncoder


@dataclass
class HyperSLBundle:
    name: str
    checkpoint_path: Path
    model: SpectralSharedEncoder
    config: dict[str, Any]
    mask_ratio: float


def display_path(path: Path, repo_root: Path) -> str:
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)


def find_repo_root() -> Path:
    cwd = Path.cwd().resolve()
    for path in [cwd, *cwd.parents]:
        if (path / "hypersl").is_dir() and (path / "data").is_dir():
            return path
    return cwd


def find_checkpoints(repo_root: Path) -> list[Path]:
    roots = [
        repo_root / "hypersl" / "modelarchive",
        repo_root / "hypersl" / "runs",
        repo_root / "checkpoints",
    ]
    paths: list[Path] = []
    for root in roots:
        if root.exists():
            paths.extend(sorted(root.glob("**/*.pt")))
    return paths


def resolve_checkpoint_path(match: str | Path, repo_root: Path) -> Path:
    raw = Path(match).expanduser()
    if raw.is_file():
        return raw.resolve()
    repo_path = (repo_root / raw).resolve()
    if repo_path.is_file():
        return repo_path

    match_lower = str(match).lower()
    candidates = find_checkpoints(repo_root)
    exact = [
        path
        for path in candidates
        if path.name.lower() == match_lower or display_path(path, repo_root).lower() == match_lower
    ]
    if len(exact) == 1:
        return exact[0]
    if len(exact) > 1:
        raise ValueError(f"Ambiguous checkpoint match {match!r}: {[display_path(p, repo_root) for p in exact]}")

    contains = [
        path
        for path in candidates
        if match_lower in path.name.lower() or match_lower in display_path(path, repo_root).lower()
    ]
    if len(contains) == 1:
        return contains[0]
    if not contains:
        raise FileNotFoundError(f"Could not resolve checkpoint: {match}")
    raise ValueError(f"Ambiguous checkpoint match {match!r}: {[display_path(p, repo_root) for p in contains]}")


def _clean_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    cleaned = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[len("module.") :]
        if key.startswith("spectral_encoder."):
            key = key[len("spectral_encoder.") :]
        cleaned[key] = value
    return cleaned


def unpack_checkpoint(raw: Any) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    if isinstance(raw, dict) and "model" in raw and isinstance(raw["model"], dict):
        return _clean_state_dict(raw["model"]), dict(raw.get("args", {}))
    if isinstance(raw, dict) and all(torch.is_tensor(value) for value in raw.values()):
        return _clean_state_dict(raw), {}
    raise TypeError("Expected a HyperSL checkpoint dict with a 'model' state_dict or a raw state_dict.")


def _filename_hints(path: Path) -> dict[str, Any]:
    name = path.name.lower()
    hints: dict[str, Any] = {}
    patterns = {
        "embedding_dim": r"embed(\d+)",
        "encoder_depth": r"enc(\d+)",
        "decoder_depth": r"dec(\d+)",
        "num_heads": r"heads(\d+)",
        "mask_ratio": r"mask(\d+)",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, name)
        if match:
            value = int(match.group(1))
            hints[key] = value / 100.0 if key == "mask_ratio" else value
    return hints


def infer_config(
    state_dict: dict[str, torch.Tensor],
    checkpoint_args: dict[str, Any],
    checkpoint_path: Path,
    overrides: dict[str, Any],
) -> dict[str, Any]:
    hints = _filename_hints(checkpoint_path)
    config: dict[str, Any] = {}

    if "global_token" in state_dict:
        config["embedding_dim"] = int(state_dict["global_token"].shape[-1])
    if "encoder_blocks.0.norm1.weight" in state_dict:
        config["embedding_dim"] = int(state_dict["encoder_blocks.0.norm1.weight"].shape[0])
    if "encoder_blocks.0.attn.qkv.weight" in state_dict:
        qkv_rows = int(state_dict["encoder_blocks.0.attn.qkv.weight"].shape[0])
        # Prefer explicit values for heads; qkv alone cannot uniquely identify num_heads.
        config.setdefault("num_heads", hints.get("num_heads"))

    config["encoder_depth"] = len(
        {
            int(key.split(".")[1])
            for key in state_dict
            if key.startswith("encoder_blocks.") and key.split(".")[1].isdigit()
        }
    ) + 1
    config["decoder_depth"] = len(
        {
            int(key.split(".")[1])
            for key in state_dict
            if key.startswith("decoder_blocks.") and key.split(".")[1].isdigit()
        }
    ) + 1

    for key in ("embedding_dim", "encoder_depth", "decoder_depth", "num_heads", "mask_ratio"):
        if key in checkpoint_args and checkpoint_args[key] is not None:
            config[key] = checkpoint_args[key]
        if key in hints and (config.get(key) is None):
            config[key] = hints[key]
        if key in overrides and overrides[key] is not None:
            config[key] = overrides[key]

    config.setdefault("embedding_dim", 128)
    config.setdefault("encoder_depth", 8)
    config.setdefault("decoder_depth", 8)
    config.setdefault("num_heads", 8)
    config.setdefault("mask_ratio", 0.75)
    config["max_band"] = int(state_dict.get("pe", torch.empty(1, 500, 1)).shape[1])
    return config


def load_bundle(
    checkpoint: str | Path,
    repo_root: Path,
    device: torch.device,
    name: str | None = None,
    embedding_dim: int | None = None,
    encoder_depth: int | None = None,
    decoder_depth: int | None = None,
    num_heads: int | None = None,
    mask_ratio: float | None = None,
) -> HyperSLBundle:
    checkpoint_path = resolve_checkpoint_path(checkpoint, repo_root)
    raw = torch.load(checkpoint_path, map_location="cpu")
    state_dict, checkpoint_args = unpack_checkpoint(raw)
    config = infer_config(
        state_dict,
        checkpoint_args,
        checkpoint_path,
        {
            "embedding_dim": embedding_dim,
            "encoder_depth": encoder_depth,
            "decoder_depth": decoder_depth,
            "num_heads": num_heads,
            "mask_ratio": mask_ratio,
        },
    )

    model = SpectralSharedEncoder(
        embedding_dim=int(config["embedding_dim"]),
        max_band=int(config["max_band"]),
        encoder_depth=int(config["encoder_depth"]),
        decoder_depth=int(config["decoder_depth"]),
        num_heads=int(config["num_heads"]),
    )
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device).eval()

    return HyperSLBundle(
        name=name or checkpoint_path.stem,
        checkpoint_path=checkpoint_path,
        model=model,
        config=config,
        mask_ratio=float(config["mask_ratio"]),
    )


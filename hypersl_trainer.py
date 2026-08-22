import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.io import loadmat
from sklearn.preprocessing import minmax_scale
from torch.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, Dataset
from engine.loss import MSE_SAM_loss
from engine.model import SpectralSharedEncoder
import wandb


WANDB_KEY = "" # was an old key, remember to make new

wandb.login(key = WANDB_KEY)

class NullSummaryWriter:
    def add_scalar(self, *args, **kwargs):
        pass

    def close(self):
        pass


class IndianPinesMAEDataset(Dataset):
    def __init__(
        self,
        mat_path,
        wavelengths,
        split="all",
        normalize=True,
    ):
        mat = loadmat(mat_path)
        if "input" not in mat:
            raise KeyError(f"{mat_path} does not contain an 'input' array.")

        cube = mat["input"].astype(np.float32)
        train_mask = mat.get("TR")
        test_mask = mat.get("TE")

        height, width, bands = cube.shape
        spectra = cube.reshape(-1, bands)
        if normalize:
            spectra = minmax_scale(spectra)
        spectra = spectra.astype(np.float32)

        if split == "all":
            selected = np.ones((height * width,), dtype=bool)
        elif split == "train":
            if train_mask is None:
                raise KeyError(f"{mat_path} does not contain a 'TR' array.")
            selected = train_mask.reshape(-1) > 0
        elif split == "test":
            if test_mask is None:
                raise KeyError(f"{mat_path} does not contain a 'TE' array.")
            selected = test_mask.reshape(-1) > 0
        elif split == "labeled":
            if train_mask is None or test_mask is None:
                raise KeyError(f"{mat_path} must contain both 'TR' and 'TE' for split='labeled'.")
            selected = (train_mask.reshape(-1) > 0) | (test_mask.reshape(-1) > 0)
        else:
            raise ValueError("split must be one of {'all', 'train', 'test', 'labeled'}.")

        self.spectra = spectra[selected]
        self.wavelengths = np.asarray(wavelengths, dtype=np.float32)

        if self.wavelengths.ndim != 1:
            raise ValueError("wavelengths must be a 1D vector.")
        if len(self.wavelengths) != bands:
            raise ValueError(f"Expected {bands} wavelengths, got {len(self.wavelengths)}.")

    def __len__(self):
        return len(self.spectra)

    def __getitem__(self, idx):
        spectrum = self.spectra[idx][np.newaxis, :]
        return spectrum.astype(np.float32), self.wavelengths


def load_wavelengths(path, num_bands, wave_min=400.0, wave_max=2500.0):
    if path is None:
        return np.linspace(wave_min, wave_max, num_bands, dtype=np.float32), False

    ext = Path(path).suffix.lower()
    if ext == ".npy":
        wavelengths = np.load(path).astype(np.float32)
    elif ext in {".txt", ".csv"}:
        if ext == ".csv":
            wavelengths = pd.read_csv(path).iloc[:, -1].to_numpy(dtype=np.float32)
        else:
            wavelengths = np.loadtxt(path, dtype=np.float32)
    else:
        raise ValueError("wavelengths file must be .npy, .txt, or .csv")

    wavelengths = np.asarray(wavelengths, dtype=np.float32).reshape(-1)
    if len(wavelengths) != num_bands:
        raise ValueError(f"Expected {num_bands} wavelengths, got {len(wavelengths)} from {path}.")
    return wavelengths, True


def build_argparser():
    parser = argparse.ArgumentParser(description="MAE-style Indian Pines pretraining with SpectralSharedEncoder.")
    parser.add_argument("--data-path", default="data/IndianPine.mat")
    parser.add_argument("--wavelengths-path", default=None)
    parser.add_argument("--split", default="all", choices=["all", "train", "test", "labeled"])
    parser.add_argument("--output-dir", default="modelarchive/indian_pines_mae")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-5)
    parser.add_argument("--mask-ratio", type=float, default=0.75)
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--encoder-depth", type=int, default=8)
    parser.add_argument("--decoder-depth", type=int, default=8)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-steps-per-epoch", type=int, default=None)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--disable-tensorboard", action="store_true")
    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", "--wandb_project", dest="wandb_project", default="HyperSL")
    parser.add_argument("--wandb-entity", "--wandb_entity", dest="wandb_entity", default="lxdcis-rochester-institute-of-technology")
    parser.add_argument("--wandb-run-name", "--wandb_run_name", dest="wandb_run_name", default="")
    parser.add_argument("--wandb-mode", "--wandb_mode", dest="wandb_mode", choices=["online", "offline", "disabled"], default="online")
    return parser


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def save_checkpoint(path, epoch, step, model, optimizer, scheduler, args):
    state = {
        "epoch": epoch,
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "args": vars(args),
    }
    torch.save(state, path)


def log_line(message=""):
    print(message, flush=True)


def log_banner(title):
    rule = "=" * 88
    log_line(rule)
    log_line(title)
    log_line(rule)


def format_duration(seconds):
    total_seconds = int(round(seconds))
    minutes, seconds = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {seconds:02d}s"
    if minutes:
        return f"{minutes}m {seconds:02d}s"
    return f"{seconds}s"


def default_wandb_run_name(args):
    split_tag = args.split.replace("/", "_")
    return (
        f"hypersl_pretrain_{split_tag}_"
        f"embed{args.embedding_dim}_enc{args.encoder_depth}_dec{args.decoder_depth}_"
        f"heads{args.num_heads}_mask{int(args.mask_ratio * 100)}_seed{args.seed}"
    )


def start_wandb_run(args, extra_config):
    if not args.wandb:
        return None

    try:
        import wandb
    except ImportError as exc:
        raise ImportError("wandb is not installed. Install it with: pip install wandb") from exc

    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity or None,
        name=args.wandb_run_name or default_wandb_run_name(args),
        mode=args.wandb_mode,
        config={**vars(args), **extra_config},
    )
    wandb.define_metric("epoch")
    for metric_name in (
        "mae_loss",
        "learning_rate",
        "epoch_time_seconds",
        "batches",
        "step",
    ):
        wandb.define_metric(metric_name, step_metric="epoch")
    return run


def main():
    args = build_argparser().parse_args()
    set_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    summary_writer_cls = None
    if not args.disable_tensorboard:
        try:
            from torch.utils.tensorboard import SummaryWriter as TorchSummaryWriter
            summary_writer_cls = TorchSummaryWriter
        except Exception as exc:
            log_line(f"TensorBoard unavailable ({exc}). Falling back to console-only logging.")

    if summary_writer_cls is None:
        writer = NullSummaryWriter()
        tensorboard_status = "off"
    else:
        writer = summary_writer_cls(log_dir=os.path.join(args.output_dir, "runs"))
        tensorboard_status = "on"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.disable_amp

    mat = loadmat(args.data_path)
    if "input" not in mat:
        raise KeyError(f"{args.data_path} does not contain an 'input' array.")
    num_bands = mat["input"].shape[-1]
    wavelengths, has_real_waves = load_wavelengths(args.wavelengths_path, num_bands)

    dataset = IndianPinesMAEDataset(
        mat_path=args.data_path,
        wavelengths=wavelengths,
        split=args.split,
    )

    wandb_run = start_wandb_run(
        args,
        {
            "device": str(device),
            "amp": use_amp,
            "tensorboard": tensorboard_status,
            "num_bands": int(num_bands),
            "spectra": int(len(dataset)),
            "has_real_wavelengths": has_real_waves,
        },
    )

    log_banner("HyperSL MAE Pretraining")
    log_line(f"device={device} | amp={use_amp} | tensorboard={tensorboard_status} | output={args.output_dir}")
    log_line(f"data={args.data_path} | split={args.split} | spectra={len(dataset)} | bands={num_bands}")
    log_line(f"embed={args.embedding_dim} | enc={args.encoder_depth} | dec={args.decoder_depth} | "f"heads={args.num_heads} | mask={args.mask_ratio:.2f}")
    log_line(f"batch_size={args.batch_size} | lr={args.lr:.2e} | weight_decay={args.weight_decay:.2e} | "f"workers={args.num_workers}")
    if has_real_waves:
        log_line(f"wavelengths={args.wavelengths_path}")
    else:
        log_line(f"wavelengths=placeholder linspace [{wavelengths[0]:.1f}, {wavelengths[-1]:.1f}]")
    log_line("-" * 88)

    dataloader = DataLoader(dataset,batch_size=args.batch_size,shuffle=True,num_workers=args.num_workers,pin_memory=torch.cuda.is_available())

    model = SpectralSharedEncoder(embedding_dim=args.embedding_dim,
                                  num_heads=args.num_heads,
                                  decoder_depth=args.decoder_depth,
                                  encoder_depth=args.encoder_depth,
    )
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.to(device)

    optimizer = torch.optim.AdamW(params=model.parameters(),
                                  lr=args.lr,
                                  betas=(0.9, 0.95),
                                  weight_decay=args.weight_decay,
                                  eps=1e-8,
    )
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    scaler = GradScaler(device.type, enabled=use_amp)

    start_epoch = 0
    step = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = checkpoint.get("epoch", 0) + 1
        step = checkpoint.get("step", 0)
        log_line(f"resume={args.resume} | next_epoch={start_epoch + 1} | step={step}")

    total_start_time = time.time()
    total_batches = len(dataloader)
    checkpoint_path = None
    final_epoch_loss = None
    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_losses = []
        epoch_start_time = time.time()
        epoch_batches = (
            min(total_batches, args.max_steps_per_epoch)
            if args.max_steps_per_epoch is not None
            else total_batches
        )

        for batch_idx, (spectra, wave) in enumerate(dataloader, start=1):
            step += 1
            spectra = spectra.to(device, non_blocking=True)
            wave = wave.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                _, recon = model(spectra, wave, args.mask_ratio)
                loss = MSE_SAM_loss(recon, spectra)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), max_norm=args.grad_clip, norm_type=2)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            loss_value = loss.item()
            epoch_losses.append(loss_value)

            writer.add_scalar("loss/train_step", loss_value, step)
            writer.add_scalar("lr", optimizer.param_groups[0]["lr"], step)

            if args.max_steps_per_epoch is not None and len(epoch_losses) >= args.max_steps_per_epoch:
                break

        mean_epoch_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        final_epoch_loss = mean_epoch_loss
        writer.add_scalar("loss/train_epoch", mean_epoch_loss, epoch)
        epoch_duration = time.time() - epoch_start_time
        log_line(
            f"[Epoch {epoch + 1:03d}/{args.epochs:03d}] "
            f"loss={mean_epoch_loss:.6f} | batches={epoch_batches} | "
            f"lr={optimizer.param_groups[0]['lr']:.2e} | "
            f"time={format_duration(epoch_duration)}"
        )
        if wandb_run is not None:
            wandb_run.log(
                {
                    "epoch": epoch + 1,
                    "mae_loss": mean_epoch_loss,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "epoch_time_seconds": epoch_duration,
                    "batches": epoch_batches,
                    "step": step,
                }
            )

        if epoch + 1 == args.epochs:
            # Save a single checkpoint after the full training run completes.
            checkpoint_path = os.path.join(
                args.output_dir,
                f"embed{args.embedding_dim}_enc{args.encoder_depth}_dec{args.decoder_depth}_heads{args.num_heads}_mask{int(args.mask_ratio*100)}_epoch{epoch + 1}.pt"
            )
            save_checkpoint(checkpoint_path, epoch, step, model, optimizer, scheduler, args)
            log_line(f"[Checkpoint] saved to {checkpoint_path}")

    writer.close()
    total_duration = time.time() - total_start_time
    log_banner("Training Complete")
    log_line(
        f"epochs_run={max(args.epochs - start_epoch, 0)} | total_steps={step} | "
        f"total_time={format_duration(total_duration)}")
    if checkpoint_path is not None:
        log_line(f"checkpoint={checkpoint_path}")
    if wandb_run is not None:
        wandb_run.summary["total_steps"] = int(step)
        wandb_run.summary["total_time_seconds"] = float(total_duration)
        wandb_run.summary["checkpoint"] = checkpoint_path or ""
        if final_epoch_loss is not None:
            wandb_run.summary["final_mae_loss"] = float(final_epoch_loss)
        wandb_run.finish()


if __name__ == "__main__":
    main()

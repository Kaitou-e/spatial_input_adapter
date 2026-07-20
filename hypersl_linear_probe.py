import argparse
import math
import random
import time
import warnings
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
from scipy.io import loadmat
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import minmax_scale
from torch.utils.data import DataLoader, Dataset
from torch.amp import GradScaler, autocast

from engine.classification import ClassificationModel

warnings.filterwarnings("ignore")

INDIAN_PINES_REMOVED_BANDS = list(range(103, 108)) + list(range(149, 163)) + [119]


class HyperDataset(Dataset):
    def __init__(self, data, label, wave):
        self.data = data.astype(np.float32)
        self.label = label.astype(np.int64)
        self.wave = wave.astype(np.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.wave, self.label[idx]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalize_hsi(hsi):
    height, width, bands = hsi.shape
    hsi = hsi.reshape(-1, bands)
    hsi = minmax_scale(hsi)
    return hsi.reshape(height, width, bands).astype(np.float32)


def infer_placeholder_wavelengths(num_bands):
    return np.linspace(400.0, 2500.0, num_bands, dtype=np.float32)


def load_wavelengths(wavelengths_path, num_bands, remove_bands=None):
    if wavelengths_path is None:
        return infer_placeholder_wavelengths(num_bands), False

    waves = pd.read_csv(wavelengths_path).iloc[:, -1].to_numpy(dtype=np.float32)
    if remove_bands and len(waves) == num_bands + len(remove_bands):
        waves = np.delete(waves, remove_bands)

    if len(waves) != num_bands:
        raise ValueError(
            f"Wavelength count mismatch: expected {num_bands}, got {len(waves)} from {wavelengths_path}."
        )

    return waves.astype(np.float32), True


def read_indian_pines(data_path, wavelengths_path=None):
    mat = loadmat(data_path)
    required_keys = {"input", "TR", "TE"}
    missing = required_keys.difference(mat)
    if missing:
        raise KeyError(f"{data_path} is missing required keys: {sorted(missing)}")

    hsi = normalize_hsi(mat["input"].astype(np.float32))
    train_mask = mat["TR"].astype(np.int64)
    test_mask = mat["TE"].astype(np.int64)
    waves, has_real_waves = load_wavelengths(
        wavelengths_path, hsi.shape[-1], remove_bands=INDIAN_PINES_REMOVED_BANDS
    )
    return hsi, train_mask, test_mask, waves, has_real_waves


def combine_label_masks(train_mask, test_mask):
    overlap = (train_mask > 0) & (test_mask > 0)
    if np.any(overlap):
        raise ValueError(f"TR and TE overlap on {int(overlap.sum())} pixels; expected disjoint masks.")

    labels = train_mask.copy()
    labels[test_mask > 0] = test_mask[test_mask > 0]
    return labels.astype(np.int64)


def make_stratified_split(label_mask, train_ratio, split_seed):
    if not 0.0 < train_ratio < 1.0:
        raise ValueError(f"train_ratio must be in (0, 1), got {train_ratio}.")

    train_split = np.zeros_like(label_mask, dtype=np.int64)
    test_split = np.zeros_like(label_mask, dtype=np.int64)
    rng = np.random.default_rng(split_seed)

    for class_id in sorted(int(value) for value in np.unique(label_mask) if value > 0):
        coords = np.argwhere(label_mask == class_id)
        rng.shuffle(coords)

        train_count = int(math.ceil(len(coords) * train_ratio))
        if len(coords) > 1:
            train_count = min(max(train_count, 1), len(coords) - 1)
        else:
            train_count = len(coords)

        train_coords = coords[:train_count]
        test_coords = coords[train_count:]

        train_split[train_coords[:, 0], train_coords[:, 1]] = class_id
        if len(test_coords):
            test_split[test_coords[:, 0], test_coords[:, 1]] = class_id

    return train_split, test_split


def extract_patches(data, label_mask, patch_size):
    assert patch_size % 2 == 1, "The window size must be odd."
    radius = patch_size // 2
    padded = np.pad(data, ((radius, radius), (radius, radius), (0, 0)), mode="reflect")
    coords = np.argwhere(label_mask > 0)
    patches = np.asarray(
        [padded[x : x + patch_size, y : y + patch_size, :] for x, y in coords],
        dtype=np.float32,
    )
    labels = label_mask[coords[:, 0], coords[:, 1]].astype(np.int64) - 1
    return patches, labels, coords


def load_pretrained_encoder(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint

    cleaned = OrderedDict()
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[7:]
        if key.startswith("spectral_encoder."):
            key = key[len("spectral_encoder."):]
        cleaned[key] = value

    incompatible = model.spectral_encoder.load_state_dict(cleaned, strict=False)
    return incompatible.missing_keys, incompatible.unexpected_keys


def run_epoch(model, loader, optimizer, criterion, device, max_batches=None):
    model.train()
    losses = []
    correct = 0
    total = 0
    grad_accum_steps = max(1, loader.grad_accum_steps)
    use_amp = loader.use_amp

    optimizer.zero_grad(set_to_none=True)

    for step, (x, w, y) in enumerate(loader, start=1):
        x = x.to(device)
        w = w.to(device)
        y = y.to(device)

        with autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(x, w)
            loss = criterion(logits, y)

        scaled_loss = loss / grad_accum_steps
        loader.scaler.scale(scaled_loss).backward()

        should_step = step % grad_accum_steps == 0
        if should_step:
            loader.scaler.step(optimizer)
            loader.scaler.update()
            optimizer.zero_grad(set_to_none=True)

        losses.append(loss.item())
        correct += (logits.argmax(1) == y).sum().item()
        total += y.numel()

        if max_batches is not None and step >= max_batches:
            break

    if total and step % grad_accum_steps != 0:
        loader.scaler.step(optimizer)
        loader.scaler.update()
        optimizer.zero_grad(set_to_none=True)

    mean_loss = float(np.mean(losses)) if losses else 0.0
    accuracy = correct / total if total else 0.0
    return mean_loss, accuracy


def evaluate(model, loader, device, class_num, max_batches=None):
    model.eval()
    preds = []
    labels = []
    use_amp = loader.use_amp

    with torch.no_grad():
        for step, (x, w, y) in enumerate(loader, start=1):
            x = x.to(device)
            w = w.to(device)
            with autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(x, w)
            preds.append(logits.argmax(1).cpu().numpy())
            labels.append(y.numpy())

            if max_batches is not None and step >= max_batches:
                break

    preds = np.concatenate(preds) if preds else np.empty((0,), dtype=np.int64)
    labels = np.concatenate(labels) if labels else np.empty((0,), dtype=np.int64)

    cm = confusion_matrix(labels, preds, labels=np.arange(class_num))
    oa = float((preds == labels).mean()) if len(labels) else 0.0

    per_class_total = cm.sum(axis=1)
    per_class_acc = np.divide(
        np.diag(cm),
        per_class_total,
        out=np.zeros_like(per_class_total, dtype=np.float64),
        where=per_class_total != 0,
    )
    aa = float(per_class_acc[per_class_total > 0].mean()) if np.any(per_class_total > 0) else 0.0

    total = cm.sum()
    pe = float((cm.sum(axis=0) * cm.sum(axis=1)).sum() / (total ** 2)) if total else 0.0
    kappa = (oa - pe) / (1 - pe) if total and (1 - pe) != 0 else 0.0

    return {
        "oa": oa,
        "aa": aa,
        "kappa": float(kappa),
        "confusion_matrix": cm,
    }


def default_wandb_run_name(args):
    probe_type = "linear_probe" if args.linear_probe else "finetune"
    split_tag = f"ratio{args.train_ratio:g}" if args.train_ratio is not None else "packaged_split"
    return (
        f"hypersl_{probe_type}_{args.model_size}_"
        f"p{args.patch_size}_{split_tag}_seed{args.seed}"
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
        "train_loss",
        "train_accuracy",
        "test_oa",
        "test_aa",
        "test_kappa",
        "learning_rate",
        "epoch_time_seconds",
    ):
        wandb.define_metric(metric_name, step_metric="epoch")
    return run


def build_argparser():
    parser = argparse.ArgumentParser(description="Indian Pines classification with HyperSL.")
    parser.add_argument("--data-path", default="data/IndianPine.mat")
    parser.add_argument("--wavelengths-path", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--model-size", default="small", choices=["small", "base", "large", "huge"])
    parser.add_argument("--embedding-dim", type=int, default=None)
    parser.add_argument("--encoder-depth", type=int, default=None)
    parser.add_argument("--decoder-depth", type=int, default=None)
    parser.add_argument("--num-heads", type=int, default=None)
    parser.add_argument("--patch-size", type=int, default=15)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--test-batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-ratio", type=float, default=None)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument("--use-checkpointing", action="store_true")
    parser.add_argument("--linear-probe", action="store_true")
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-test-batches", type=int, default=None)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", "--wandb_project", dest="wandb_project", default="HyperSL")
    parser.add_argument("--wandb-entity", "--wandb_entity", dest="wandb_entity", default="dk6012")
    parser.add_argument("--wandb-run-name", "--wandb_run_name", dest="wandb_run_name", default="")
    parser.add_argument("--wandb-mode", "--wandb_mode", dest="wandb_mode", choices=["online", "offline", "disabled"], default="online")
    return parser


def main():
    args = build_argparser().parse_args()
    set_seed(args.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.disable_amp
    print(f"Using device: {device}")
    print(
        f"Memory settings: batch_size={args.batch_size}, test_batch_size={args.test_batch_size}, "
        f"grad_accum_steps={args.grad_accum_steps}, amp={use_amp}, checkpointing={args.use_checkpointing}"
    )

    data, train_mask, test_mask, wavelengths, has_real_waves = read_indian_pines(args.data_path, args.wavelengths_path)
    if has_real_waves:
        print(f"Loaded {len(wavelengths)} wavelengths from {args.wavelengths_path}.")
    else:
        print("No wavelength file provided. Using placeholder wavelengths; supply a real wavelength CSV for best results.")

    if args.train_ratio is not None:
        full_label_mask = combine_label_masks(train_mask, test_mask)
        train_mask, test_mask = make_stratified_split(full_label_mask, args.train_ratio, args.split_seed)
        print(
            f"Using stratified runtime split from combined labeled mask: "
            f"train_ratio={args.train_ratio:.4f}, split_seed={args.split_seed}"
        )
    else:
        print("Using packaged TR/TE split from the dataset file.")

    x_train, y_train, _ = extract_patches(data, train_mask, args.patch_size)
    x_test, y_test, _ = extract_patches(data, test_mask, args.patch_size)
    class_num = int(max(train_mask.max(), test_mask.max()))
    total_labeled = len(y_train) + len(y_test)
    train_fraction = (len(y_train) / total_labeled) if total_labeled else 0.0
    test_fraction = (len(y_test) / total_labeled) if total_labeled else 0.0
    print(
        f"Train patches: {x_train.shape}, Test patches: {x_test.shape}, Classes: {class_num}, "
        f"Split: train={len(y_train)} ({train_fraction:.2%}) test={len(y_test)} ({test_fraction:.2%})"
    )
    train_dataset = HyperDataset(x_train, y_train, wavelengths)
    test_dataset = HyperDataset(x_test, y_test, wavelengths)

    train_loader = DataLoader(train_dataset,batch_size=args.batch_size,shuffle=True,num_workers=args.num_workers,pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(test_dataset,batch_size=args.test_batch_size,shuffle=False,num_workers=args.num_workers,pin_memory=torch.cuda.is_available())

    model = ClassificationModel(class_num=class_num,
                                model_size=args.model_size,
                                embedding_dim=args.embedding_dim,
                                encoder_depth=args.encoder_depth,
                                decoder_depth=args.decoder_depth,
                                num_heads=args.num_heads,
                                use_checkpointing=args.use_checkpointing,
                                linear_probe=args.linear_probe,).to(device)

    if args.checkpoint:
        missing_keys, unexpected_keys = load_pretrained_encoder(model, args.checkpoint)
        print(
            f"Loaded encoder weights from {args.checkpoint}. "
            f"Missing keys: {len(missing_keys)}, Unexpected keys: {len(unexpected_keys)}"
        )

    if args.linear_probe:
        model.freeze_encoder()
        print("Running in linear-probe mode: encoder frozen, training linear classifier only.")
    else:
        print("Running in fine-tuning mode: encoder and classifier are trainable.")

    trainable_params = [param for param in model.parameters() if param.requires_grad]
    total_params = sum(param.numel() for param in model.parameters())
    trainable_count = sum(param.numel() for param in trainable_params)
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_count}")

    optimizer = torch.optim.AdamW(trainable_params,lr=args.lr,weight_decay=args.weight_decay)
    
    criterion = torch.nn.CrossEntropyLoss()
    scaler = GradScaler(device.type, enabled=use_amp)

    train_loader.grad_accum_steps = args.grad_accum_steps
    train_loader.use_amp = use_amp
    train_loader.scaler = scaler
    test_loader.use_amp = use_amp

    wandb_run = start_wandb_run(
        args,
        {
            "device": str(device),
            "amp": use_amp,
            "has_real_wavelengths": has_real_waves,
            "num_bands": int(data.shape[-1]),
            "class_num": int(class_num),
            "train_samples": int(len(y_train)),
            "test_samples": int(len(y_test)),
            "train_fraction": float(train_fraction),
            "test_fraction": float(test_fraction),
            "total_parameters": int(total_params),
            "trainable_parameters": int(trainable_count),
        },
    )
    best_metrics = None
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        train_loss, train_acc = run_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            max_batches=args.max_train_batches,
        )
        print(f"Epoch {epoch}: train_loss={train_loss:.6f} train_acc={train_acc:.4f}")
        epoch_time = time.time() - epoch_start_time
        if wandb_run is not None:
            wandb_run.log(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_accuracy": train_acc,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "epoch_time_seconds": epoch_time,
                }
            )

        should_eval = epoch % args.eval_every == 0 or epoch == args.epochs
        if should_eval:
            metrics = evaluate(
                model,
                test_loader,
                device,
                class_num,
                max_batches=args.max_test_batches,
            )
            print(
                f"Epoch {epoch}: test_oa={metrics['oa']:.4f} "
                f"test_aa={metrics['aa']:.4f} test_kappa={metrics['kappa']:.4f}"
            )
            if best_metrics is None or metrics["oa"] >= best_metrics["oa"]:
                best_metrics = metrics
                best_epoch = epoch
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "epoch": epoch,
                        "test_oa": metrics["oa"],
                        "test_aa": metrics["aa"],
                        "test_kappa": metrics["kappa"],
                    }
                )

    if wandb_run is not None:
        if best_metrics is not None:
            wandb_run.summary["best_epoch"] = int(best_epoch)
            wandb_run.summary["best_test_oa"] = float(best_metrics["oa"])
            wandb_run.summary["best_test_aa"] = float(best_metrics["aa"])
            wandb_run.summary["best_test_kappa"] = float(best_metrics["kappa"])
            wandb_run.summary["confusion_matrix"] = best_metrics["confusion_matrix"].tolist()
        wandb_run.finish()


if __name__ == "__main__":
    main()

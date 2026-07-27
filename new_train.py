import os
from pathlib import Path
import time

import matplotlib.pyplot as plt
# import numpy as np
import numpy as np
import torch
import wandb
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter
from datautils.mutihsi import MultisourceHSI, MultisourceHSIDataset
from engine.model import SpectralSharedEncoder
from engine.loss import MSE_SAM_loss
from torch.nn.utils import clip_grad_norm_

WANDB_KEY = "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"
REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = REPO_ROOT / 'hypersl_training' / 'data' / 'MultiSourceHSI_test.hdf5'
# DATASET_PATH = '/home/lxdcis/hypersl_training/data/MultiSourceHSI_test.hdf5'
MODELARCHIVE_DIR = REPO_ROOT / 'hypersl' / 'modelarchive' / 'indian_local'
MODELARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

def scheduler_lr(optimizer, scheduler):
    lr_history = []

    """optimizer的更新在scheduler更新的前面"""
    for epoch in range(2):
        for i in range(32000):
            optimizer.step()  # 更新参数
            lr_history.append(optimizer.param_groups[0]['lr'])
            scheduler.step()  # 调整学习率
    return lr_history

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

def start_wandb_run(wandb_project, wandb_mode, extra_config, wandb_entity=None, wandb_run_name=''):
    try:
        import wandb
    except ImportError as exc:
        raise ImportError("wandb is not installed. Install it with: pip install wandb") from exc

    run = wandb.init(
        project=wandb_project,
        entity=wandb_entity or None,
        name=wandb_run_name,
        mode=wandb_mode,
        config={**extra_config},
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

if __name__ == '__main__':
    EPOCH = 3
    start_epoch = 0
    lr = 5e-4
    batch_size = 256
    mask_ratio = 0.95
    resume = False

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}. Check that the file exists in the repository data folder.")

    MHSIs = MultisourceHSI(dest=str(DATASET_PATH))
    EnMap_dataset = MultisourceHSIDataset(MHSIs, source='ENMap')
    DESIS_dataset = MultisourceHSIDataset(MHSIs, source='DESIS')
    EnMap_dataloader = DataLoader(EnMap_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=8)
    DESIS_dataloader = DataLoader(DESIS_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=8)

    model = SpectralSharedEncoder(embedding_dim=256,
                                  num_heads=8,
                                  decoder_depth=4,
                                  encoder_depth=8,
                                  )

    optimizer = torch.optim.AdamW(
        params=model.parameters(),
        lr=lr,
        betas=(0.9, 0.95),
        weight_decay=5e-5,
        eps=1e-8,
    )

    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, )

    # writer = SummaryWriter()

    wandb_run = start_wandb_run(
        'hyperSL-local-aware-test',
        'online',
        {
            "device": str(device),
            "total epochs": EPOCH,
        },
        'lxdcis-rochester-institute-of-technology',
        'hypersl-local-aware-pretrain'
    )

    if resume:
        checkpoint_path = MODELARCHIVE_DIR / 'latest_checkpoint.pt'
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location=device)
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            print(f'Loaded checkpoint from {checkpoint_path}')
        else:
            print(f'No checkpoint found at {checkpoint_path}; starting training from scratch.')
            resume = False

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.to(device)
    length = min(len(EnMap_dataloader), len(DESIS_dataloader))
    step = 0
    total_start_time = time.time()
    final_epoch_loss = None
    for epoch in range(start_epoch, EPOCH):
        model.train()
        epoch_losses = []
        epoch_start_time = time.time()
        print('epoch', epoch, ':')
        for (enmap_s, enmap_w), (desis_s, desis_w) in zip(EnMap_dataloader, DESIS_dataloader):
            step += 1
            _, output1 = model(enmap_s.cuda(), enmap_w.cuda(), mask_ratio)
            _, output2 = model(desis_s.cuda(), desis_w.cuda(), mask_ratio)
            loss1 = MSE_SAM_loss(output1, enmap_s.cuda())
            loss2 = MSE_SAM_loss(output2, desis_s.cuda())
            loss = loss1 + loss2

            loss_value = loss.item()
            epoch_losses.append(loss_value)

            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=5.0, norm_type=2)
            optimizer.step()
            scheduler.step()

            print(f'Step:{step}', ': total_loss:', loss.item(), 'loss1:', loss1.item(), 'loss2:', loss2.item(),
                  'lr:', scheduler.get_last_lr())

            # writer.add_scalar('Total_loss/train', loss.item(), step)
            # writer.add_scalar('Loss1/train', loss1.item(), step)
            # writer.add_scalar('Loss2/train', loss2.item(), step)

            if step % 5000 == 0:
                plt.plot(output1.detach().cpu().numpy().squeeze()[10, :], label='en_r')
                plt.plot(enmap_s.detach().cpu().numpy().squeeze()[10, :], label=f'en_o{loss1.item()}')
                plt.plot(output2.detach().cpu().numpy().squeeze()[10, :], label='de_r')
                plt.plot(desis_s.detach().cpu().numpy().squeeze()[10, :], label=f'de_o{loss2.item()}')
                plt.legend()
                plt.ylim((0, 1.0))
                plt.show()

        # with torch.no_grad():
        #     pass
            # test

        mean_epoch_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        epoch_duration = time.time() - epoch_start_time

        log_line(
            f"[Epoch {epoch + 1:03d}/{EPOCH:03d}] "
            # f"loss={mean_epoch_loss:.6f} | batches={epoch_batches} | "
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
                    # "batches": epoch_batches,
                    "step": step,
                }
            )
        
        # save model
        if epoch % 10 == 0:
            # save checkpoint
            to_save = {
                'epoch':epoch,
                'model':model.state_dict(),
                'optimizer':optimizer.state_dict(),
                'scheduler':scheduler.state_dict(),
            }
            path_checkpoint = MODELARCHIVE_DIR / f'{epoch}_checkpoint.pt'
            torch.save(to_save, path_checkpoint)
            torch.save(to_save, MODELARCHIVE_DIR / 'latest_checkpoint.pt')

        # if epoch + 1 == EPOCH:
        #     # Save a single checkpoint after the full training run completes.
        #     checkpoint_path = os.path.join(
        #         '/home/lxdcis/hypersl_training/hypersl/modelarchive/',
        #         f"_epoch{epoch + 1}.pt"
        #     )
        #     save_checkpoint(checkpoint_path, epoch, step, model, optimizer, scheduler, args)
        #     log_line(f"[Checkpoint] saved to {checkpoint_path}")

    total_duration = time.time() - total_start_time
    log_banner("Training Complete")
    if wandb_run is not None:
        wandb_run.summary["total_steps"] = int(step)
        wandb_run.summary["total_time_seconds"] = float(total_duration)
        wandb_run.summary["checkpoint"] = checkpoint_path or ""
        if final_epoch_loss is not None:
            wandb_run.summary["final_mae_loss"] = float(final_epoch_loss)
        wandb_run.finish()

    a = 0
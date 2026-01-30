import torch
from torch.utils.data import DataLoader

import time
import argparse
from dataclasses import dataclass
from pathlib import Path


from models.ddnaf import *
from models.utils.ema import *
from models.utils.imagedataset import *
from models.utils.training import *
from models.utils.learning import *
from models.utils.loss import *
from models.utils.metrics import *

@dataclass
class TrainConfig:
    noisy_train: str
    gt_train: str
    noisy_val: str
    gt_val: str
    out_dir: str = "./runs/ddnafnet_sidd"
    width: int = 32
    patch: int = 256
    batch: int = 8
    workers: int = 4
    epochs: int = 200
    lr: float = 2e-4
    weight_decay: float = 1e-4
    warmup_steps: int = 2000
    ema_decay: float = 0.999
    grad_clip: float = 1.0
    log_every: int = 50
    val_every: int = 1
    save_every: int = 1
    amp: bool = True
    seed: int = 123

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--noisy_train", required=True)
    p.add_argument("--gt_train", required=True)
    p.add_argument("--noisy_val", required=True)
    p.add_argument("--gt_val", required=True)
    p.add_argument("--out_dir", required=True)

    p.add_argument("--width", type=int, default=32)
    p.add_argument("--patch", type=int, default=256)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--warmup_steps", type=int, default=1500)
    p.add_argument("--ema_decay", type=float, default=0.999)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--log_every", type=int, default=5)
    p.add_argument("--val_every", type=int, default=50)
    p.add_argument("--save_every", type=int, default=25)
    p.add_argument("--no_amp", action="store_true")
    p.add_argument("--seed", type=int, default=123)

    p.add_argument("--resume", type=str, default="")

    args = p.parse_args()

    cfg = TrainConfig(
        noisy_train=args.noisy_train,
        gt_train=args.gt_train,
        noisy_val=args.noisy_val,
        gt_val=args.gt_val,
        out_dir=args.out_dir,
        width=args.width,
        patch=args.patch,
        batch=args.batch,
        workers=args.workers,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        ema_decay=args.ema_decay,
        grad_clip=args.grad_clip,
        log_every=args.log_every,
        val_every=args.val_every,
        save_every=args.save_every,
        amp=not args.no_amp,
        seed=args.seed
    )

    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Datasets / Loaders
    train_ds = PairedImageFolder(cfg.noisy_train, cfg.gt_train, patch_size=cfg.patch, train=True)
    val_ds   = PairedImageFolder(cfg.noisy_val, cfg.gt_val, patch_size=cfg.patch, train=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch,
        shuffle=True,
        num_workers=cfg.workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        persistent_workers=False,
    )

    # Model (D-NAFNet)
    model = DNAFNet(img_channel=3, width=cfg.width).to(device)

    # Loss / Optimizer
    criterion = CharbonnierLoss(eps=1e-3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # AMP + EMA
    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.amp and device.type == "cuda"))
    ema = EMA(model, decay=cfg.ema_decay)

    total_steps = cfg.epochs * len(train_loader)
    start_epoch = 1
    global_step = 0
    if args.resume:
        resume_path = Path(args.resume)
        global_step, last_epoch = load_ckpt(resume_path, model, optimizer, ema, device)
        start_epoch = last_epoch + 1
        print(f"Resumed from {resume_path} | last_epoch={last_epoch} | global_step={global_step}")

    best_psnr = -1.0

    print(f"Train pairs: {len(train_ds)} | Val pairs: {len(val_ds)}")
    print(f"Device: {device} | AMP: {scaler.is_enabled()} | Steps/epoch: {len(train_loader)} | Total steps: {total_steps}")

    for epoch in range(start_epoch, cfg.epochs + 1):
        model.train()
        t0 = time.time()

        for it, (noisy, gt) in enumerate(train_loader, start=1):
            noisy = noisy.to(device, non_blocking=True)
            gt = gt.to(device, non_blocking=True)

            # LR schedule
            lr_now = lr_cosine_with_warmup(global_step, total_steps, cfg.warmup_steps, cfg.lr)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_now

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                pred = model(noisy)
                loss = criterion(pred, gt)

            scaler.scale(loss).backward()

            # grad clip (after unscale)
            scaler.unscale_(optimizer)
            if cfg.grad_clip and cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            ema.update(model)
            global_step += 1

            if global_step % cfg.log_every == 0:
                dt = time.time() - t0
                ips = (cfg.log_every * cfg.batch) / max(1e-9, dt)
                print(
                    f"epoch {epoch:03d}/{cfg.epochs} | step {global_step:07d}/{total_steps} "
                    f"| loss {loss.item():.5f} | lr {lr_now:.2e} | {ips:.1f} img/s"
                )
                t0 = time.time()

        # Validation (EMA weights)
        if (epoch % cfg.val_every) == 0:
            if device.type == "cuda":
                torch.cuda.empty_cache()

            backup = apply_ema_weights(model, ema.shadow)
            val_psnr = run_validation(model, val_loader, device)
            restore_weights(model, backup)

            print(f"[val] epoch {epoch:03d}: PSNR(EMA) = {val_psnr:.4f} dB")

            if val_psnr > best_psnr:
                best_psnr = val_psnr
                save_ckpt(out_dir / "best.pt", model, optimizer, global_step, epoch, ema)
                print(f"  ↳ saved best checkpoint (PSNR={best_psnr:.4f})")

        # Periodic save
        if (epoch % cfg.save_every) == 0:
            save_ckpt(out_dir / f"epoch_{epoch:03d}.pt", model, optimizer, global_step, epoch, ema)

    print(f"Done. Best PSNR(EMA) = {best_psnr:.4f} dB")
    print(f"Checkpoints in: {out_dir}")


if __name__ == "__main__":
    main()
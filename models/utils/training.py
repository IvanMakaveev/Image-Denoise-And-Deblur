import torch
import torch.nn as nn

from pathlib import Path
from models.utils.ema import EMA
from models.utils.metrics import psnr

def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_ckpt(path: Path, model: nn.Module, opt: torch.optim.Optimizer, step: int, epoch: int, ema: EMA | None):
    payload = {
        "model": model.state_dict(),
        "opt": opt.state_dict(),
        "step": step,
        "epoch": epoch,
    }
    if ema is not None:
        payload["ema"] = ema.shadow
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, str(path))
    
def load_ckpt(path: Path, model: nn.Module, opt: torch.optim.Optimizer, ema: EMA | None, device):
    ckpt = torch.load(str(path), map_location=device)
    model.load_state_dict(ckpt["model"], strict=True)
    opt.load_state_dict(ckpt["opt"])
    if ema is not None and ckpt.get("ema") is not None:
        ema.shadow = ckpt["ema"]
    step = int(ckpt.get("step", 0))
    epoch = int(ckpt.get("epoch", 0))
    return step, epoch

@torch.no_grad()
def run_validation(model, loader, device):
    model.eval()
    total_psnr = 0.0
    n = 0
    use_amp = (device.type == "cuda")
    for noisy, gt in loader:
        noisy = noisy.to(device, non_blocking=True)
        gt = gt.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            pred = model(noisy).clamp(0.0, 1.0)
        total_psnr += psnr(pred, gt)
        n += 1

        del noisy, gt, pred
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return total_psnr / max(1, n)
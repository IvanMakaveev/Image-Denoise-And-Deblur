import torch
import torch.nn as nn

class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module):
        msd = model.state_dict()
        for k, v in msd.items():
            self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def copy_to(self, model: nn.Module):
        model.load_state_dict(self.shadow, strict=True)

@torch.no_grad()
def apply_ema_weights(model: nn.Module, ema_shadow: dict):
    """Swap EMA weights into model. Returns a backup of current weights."""
    backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(ema_shadow, strict=True)
    return backup

@torch.no_grad()
def restore_weights(model: nn.Module, backup: dict):
    model.load_state_dict(backup, strict=True)
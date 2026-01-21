import torch
import torch.nn.functional as F

def mean_filter(img: torch.Tensor, k: int) -> torch.Tensor:
    pad = k // 2
    weight = torch.ones(3, 1, k, k, device=img.device) / (k * k)
    x = F.pad(img, (pad, pad, pad, pad), mode="reflect")
    return F.conv2d(x, weight, groups=3)

def geometric_mean_filter(img: torch.Tensor, k: int, eps: float = 1e-8) -> torch.Tensor:
    assert k % 2 == 1, "Kernel size must be odd"

    x = img.clamp(eps, 1.0)
    log_x = torch.log(x)
    log_mean = mean_filter(log_x, k)
    return torch.exp(log_mean)

def gaussian_filter(img: torch.Tensor, k: int, sigma: float) -> torch.Tensor:
    pad = k // 2
    coords = torch.arange(k, device=img.device) - pad
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()

    g1 = g.view(1, 1, 1, k)
    g2 = g.view(1, 1, k, 1)

    x = F.pad(img, (pad, pad, pad, pad), mode="reflect")
    x = F.conv2d(x, g1.expand(3, 1, 1, k), groups=3)
    x = F.conv2d(x, g2.expand(3, 1, k, 1), groups=3)
    return x

def median_filter(img: torch.Tensor, k: int) -> torch.Tensor:
    pad = k // 2
    x = F.pad(img, (pad, pad, pad, pad), mode="reflect")

    patches = x.unfold(2, k, 1).unfold(3, k, 1)
    patches = patches.contiguous().view(1, 3, img.shape[2], img.shape[3], -1)
    return patches.median(dim=-1).values

def mean_kernel(k: int, device="cpu", dtype=torch.float32) -> torch.Tensor:
    return torch.full((k, k), 1.0 / (k * k), device=device, dtype=dtype)

def gaussian_kernel(k: int, sigma: float, device="cpu", dtype=torch.float32) -> torch.Tensor:
    assert k % 2 == 1
    r = k // 2
    coords = torch.arange(-r, r + 1, device=device, dtype=dtype)
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    K = torch.outer(g, g)
    K = K / K.sum()
    return K
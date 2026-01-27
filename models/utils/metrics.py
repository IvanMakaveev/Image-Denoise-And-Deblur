import torch
import torch.nn.functional as F

@torch.no_grad()
def psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> float:
    mse = torch.mean((pred - target) ** 2).clamp_min(1e-12)
    return float(10.0 * torch.log10((max_val * max_val) / mse))


def _gaussian_kernel_2d(window_size: int, sigma: float, device, dtype):
    # 1D Gaussian
    coords = torch.arange(window_size, device=device, dtype=dtype) - (window_size - 1) / 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()

    # 2D separable Gaussian
    kernel_2d = torch.outer(g, g)
    kernel_2d = kernel_2d / kernel_2d.sum()
    return kernel_2d

@torch.no_grad()
def ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
    window_size: int = 11,
    sigma: float = 1.5,
    k1: float = 0.01,
    k2: float = 0.03,
    reduction: str = "mean",  # "mean" | "none"
) -> torch.Tensor:
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: pred {pred.shape} vs target {target.shape}")
    if pred.ndim != 4:
        raise ValueError(f"Expected NCHW (4D) tensors, got ndim={pred.ndim}")

    if window_size % 2 == 0:
        raise ValueError("window_size must be odd")

    device, dtype = pred.device, pred.dtype
    n, c, h, w = pred.shape

    kernel = _gaussian_kernel_2d(window_size, sigma, device, dtype)
    # conv2d weight shape: (out_channels, in_channels/groups, kH, kW)
    weight = kernel.view(1, 1, window_size, window_size).repeat(c, 1, 1, 1)

    padding = window_size // 2

    # Local means
    mu_x = F.conv2d(pred, weight, padding=padding, groups=c)
    mu_y = F.conv2d(target, weight, padding=padding, groups=c)

    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    # Local variances and covariance
    sigma_x2 = F.conv2d(pred * pred, weight, padding=padding, groups=c) - mu_x2
    sigma_y2 = F.conv2d(target * target, weight, padding=padding, groups=c) - mu_y2
    sigma_xy = F.conv2d(pred * target, weight, padding=padding, groups=c) - mu_xy

    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2

    ssim_map = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / (
        (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
    )

    # Average over C,H,W -> per-image SSIM
    per_image = ssim_map.mean(dim=(1, 2, 3))

    if reduction == "none":
        return per_image
    if reduction == "mean":
        return per_image.mean()
    raise ValueError("reduction must be 'mean' or 'none'")
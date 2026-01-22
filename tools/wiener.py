import torch
import torch.nn.functional as F
from typing import Callable, Optional, Union

@torch.no_grad()
def wiener_denoise_local_torch(
    img01: torch.Tensor,
    k: int = 7,
    noise_var: Optional[float] = None,
    eps: float = 1e-8,
    estimate_noise_var_fn: Optional[Callable[[torch.Tensor], float]] = None,
) -> torch.Tensor:
    assert k % 2 == 1, "k must be odd"
    if not torch.is_tensor(img01):
        raise TypeError("img01 must be a torch.Tensor")

    x_in = img01.detach()
    orig_device = x_in.device
    orig_dtype = x_in.dtype

    if x_in.ndim == 2:
        x = x_in[None, None, ...]
        layout = "HW"
    elif x_in.ndim == 3:
        x = x_in[None, ...]
        layout = "CHW"
    elif x_in.ndim == 4:
        x = x_in
        layout = "NCHW"
    else:
        raise ValueError("Expected (H,W), (C,H,W), or (N,C,H,W)")

    x = x.to(torch.float32).clamp(0.0, 1.0)
    N, C, H, W = x.shape
    if C not in (1, 3):
        raise ValueError("Expected C in {1,3}")

    def box_blur_nchw(z: torch.Tensor, k: int) -> torch.Tensor:
        pad = k // 2
        zpad = F.pad(z, (pad, pad, pad, pad), mode="reflect")
        return F.avg_pool2d(zpad, kernel_size=k, stride=1)

    def gaussian_blur3x3_depthwise(z: torch.Tensor) -> torch.Tensor:
        kernel = torch.tensor(
            [[1.0, 2.0, 1.0],
             [2.0, 4.0, 2.0],
             [1.0, 2.0, 1.0]],
            device=z.device,
            dtype=z.dtype,
        ) / 16.0
        kernel = kernel.view(1, 1, 3, 3)
        weight = kernel.repeat(z.shape[1], 1, 1, 1)
        zpad = F.pad(z, (1, 1, 1, 1), mode="reflect")
        return F.conv2d(zpad, weight, groups=z.shape[1])

    def estimate_noise_var_channel(ch_hw: torch.Tensor) -> float:
        if estimate_noise_var_fn is not None:
            return float(estimate_noise_var_fn(ch_hw))

        ch = ch_hw[None, None, ...]
        blur = gaussian_blur3x3_depthwise(ch)[0, 0]
        resid = ch_hw - blur
        r = resid.reshape(-1)
        med = r.median()
        mad = (r - med).abs().median()
        sigma = 1.4826 * mad
        return float((sigma * sigma).item())

    mu = box_blur_nchw(x, k)
    mu2 = box_blur_nchw(x * x, k)
    var = (mu2 - mu * mu).clamp_min(0.0)

    if noise_var is not None:
        nv = torch.full((N, C, 1, 1), float(noise_var), device=x.device, dtype=x.dtype)
    else:
        nv_vals = torch.empty((N, C), device=x.device, dtype=x.dtype)
        for n in range(N):
            for c in range(C):
                nv_vals[n, c] = estimate_noise_var_channel(x[n, c])
        nv = nv_vals[..., None, None]

    gain = (var - nv) / (var + eps)
    gain = gain.clamp(0.0, 1.0)

    out = mu + gain * (x - mu)
    out = out.clamp(0.0, 1.0)

    out = out.to(device=orig_device, dtype=orig_dtype)
    if layout == "HW":
        return out[0, 0]
    if layout == "CHW":
        return out[0]
    return out

@torch.no_grad()
def estimate_awgn_var_from_local_variance_torch(
    img: torch.Tensor,
    k: int = 7,
    percentile: float = 10.0,
    eps: float = 1e-8,
) -> float:
    if not torch.is_tensor(img):
        raise TypeError("img must be a torch.Tensor")

    x = img.detach()

    if x.ndim == 2:
        x = x[None, None, ...]
    elif x.ndim == 3:
        x = x[None, ...]
    elif x.ndim != 4:
        raise ValueError("Expected (H,W), (C,H,W), or (N,C,H,W)")

    x = x.to(dtype=torch.float32)

    N, C, H, W = x.shape

    if C == 3:
        r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
        x = 0.299 * r + 0.587 * g + 0.114 * b
    elif C == 1:
        pass
    else:
        raise ValueError("Expected C in {1,3}")

    pad = k // 2
    if pad > 0:
        x_pad = F.pad(x, (pad, pad, pad, pad), mode="reflect")
    else:
        x_pad = x

    mu = F.avg_pool2d(x_pad, kernel_size=k, stride=1)
    mu2 = F.avg_pool2d(x_pad * x_pad, kernel_size=k, stride=1)
    var = (mu2 - mu * mu).clamp_min(0.0)

    v = var.reshape(-1)
    q = float(percentile) / 100.0
    nv = torch.quantile(v, q).item()

    return max(float(nv), float(eps))


@torch.no_grad()
def wiener_denoise_frequency_torch(
    img: torch.Tensor,
    noise_var: Optional[float] = None,
    noise_var_estimator: Optional[Callable[[torch.Tensor], float]] = None,
    smooth_psd_sigma: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    if not torch.is_tensor(img):
        raise TypeError("img must be a torch.Tensor")

    x_in = img.detach()
    orig_device = x_in.device
    orig_dtype = x_in.dtype

    if x_in.ndim == 2:
        x = x_in[None, None, ...]
        layout = "HW"
    elif x_in.ndim == 3:
        x = x_in[None, ...]
        layout = "CHW"
    elif x_in.ndim == 4:
        x = x_in
        layout = "NCHW"
    else:
        raise ValueError("Expected (H,W), (C,H,W), or (N,C,H,W)")

    x = x.to(dtype=torch.float32)

    N, C, H, W = x.shape
    if C not in (1, 3):
        raise ValueError("Expected C in {1,3}")

    if noise_var is None:
        if noise_var_estimator is None:
            raise ValueError("Provide noise_var or noise_var_estimator")
        noise_var = float(noise_var_estimator(x_in))
    noise_var = float(noise_var)

    def _gaussian_kernel1d(sigma: float, device, dtype):
        r = max(1, int(3.0 * sigma + 0.5))
        t = torch.arange(-r, r + 1, device=device, dtype=dtype)
        k = torch.exp(-(t * t) / (2.0 * sigma * sigma))
        k = k / k.sum()
        return k, r

    def _gaussian_blur_2d(img_nchw: torch.Tensor, sigma: float) -> torch.Tensor:
        if sigma <= 0:
            return img_nchw
        k1, r = _gaussian_kernel1d(sigma, img_nchw.device, img_nchw.dtype)
        Cc = img_nchw.shape[1]
        kh = k1.view(1, 1, 1, -1).repeat(Cc, 1, 1, 1)
        kv = k1.view(1, 1, -1, 1).repeat(Cc, 1, 1, 1)
        y = F.pad(img_nchw, (r, r, 0, 0), mode="reflect")
        y = F.conv2d(y, kh, groups=Cc)
        y = F.pad(y, (0, 0, r, r), mode="reflect")
        y = F.conv2d(y, kv, groups=Cc)
        return y

    mean = x.mean(dim=(-2, -1), keepdim=True)
    x0 = x - mean

    Y = torch.fft.fft2(x0, dim=(-2, -1))
    Sy = (Y.real * Y.real + Y.imag * Y.imag).to(torch.float32)

    if smooth_psd_sigma and smooth_psd_sigma > 0:
        Sy = _gaussian_blur_2d(Sy, smooth_psd_sigma)

    Sn = noise_var * (H * W)

    Sx = (Sy - Sn).clamp_min(0.0)

    Hw = Sx / (Sx + Sn + eps)

    Xhat0 = torch.fft.ifft2(Hw.to(Y.dtype) * Y, dim=(-2, -1)).real.to(torch.float32)

    out = Xhat0 + mean
    out = out.to(device=orig_device, dtype=orig_dtype)

    if layout == "HW":
        return out[0, 0]
    if layout == "CHW":
        return out[0]
    return out

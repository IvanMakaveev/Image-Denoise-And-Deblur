import numpy as np
import torch
from bm3d import bm3d

def bm3d_denoise(img01: np.ndarray, sigma: float, colorspace: str = "rgb") -> np.ndarray:
    x = np.asarray(img01, dtype=np.float32)
    x = np.clip(x, 0.0, 1.0)

    if x.ndim == 2:
        out = bm3d(x, sigma_psd=sigma)
        return np.clip(out.astype(np.float32), 0.0, 1.0)

    if x.ndim != 3 or x.shape[2] != 3:
        raise ValueError("Expected grayscale (H,W) or RGB (H,W,3).")

    if colorspace == "rgb":
        out = np.empty_like(x)
        for c in range(3):
            out[..., c] = bm3d(x[..., c], sigma_psd=sigma).astype(np.float32)
        return np.clip(out, 0.0, 1.0)

    elif colorspace == "y":
        R, G, B = x[..., 0], x[..., 1], x[..., 2]
        Y  = 0.299 * R + 0.587 * G + 0.114 * B
        Cb = -0.168736 * R - 0.331264 * G + 0.5 * B + 0.5
        Cr = 0.5 * R - 0.418688 * G - 0.081312 * B + 0.5

        Yd = bm3d(Y, sigma_psd=sigma).astype(np.float32)

        Cr0 = Cr - 0.5
        Cb0 = Cb - 0.5
        R2 = Yd + 1.402 * Cr0
        G2 = Yd - 0.344136 * Cb0 - 0.714136 * Cr0
        B2 = Yd + 1.772 * Cb0

        out = np.stack([R2, G2, B2], axis=-1)
        return np.clip(out.astype(np.float32), 0.0, 1.0)

    else:
        raise ValueError("Colorspace must be 'rgb' or 'y'")


def bm3d_denoise_torch(
    x: torch.Tensor,
    sigma: float,
    colorspace: str = "rgb",
    clamp: bool = True,
) -> torch.Tensor:
    orig_device = x.device
    orig_dtype = x.dtype
    orig_shape = x.shape

    # Work in float32 on CPU for bm3d
    x_cpu = x.detach()
    if x_cpu.is_cuda:
        x_cpu = x_cpu.cpu()
    x_cpu = x_cpu.to(torch.float32)

    def denoise_one(img: torch.Tensor) -> torch.Tensor:
        if img.ndim == 2:
            arr = img.numpy()
            out = bm3d_denoise(arr, sigma=sigma, colorspace=colorspace)
            return torch.from_numpy(out)

        if img.ndim != 3:
            raise ValueError("Expected (H,W) or (C,H,W) for a single image.")

        C, H, W = img.shape
        if C == 1:
            arr = img[0].numpy()
            out = bm3d_denoise(arr, sigma=sigma, colorspace=colorspace)
            out_t = torch.from_numpy(out)[None, ...] 
            return out_t

        if C == 3:
            arr = img.permute(1, 2, 0).numpy()
            out = bm3d_denoise(arr, sigma=sigma, colorspace=colorspace)
            out_t = torch.from_numpy(out).permute(2, 0, 1)
            return out_t

        raise ValueError(f"Expected C in {{1,3}}, got C={C}")

    if x_cpu.ndim == 4:
        N, C, H, W = x_cpu.shape
        outs = [denoise_one(x_cpu[i]) for i in range(N)]
        y_cpu = torch.stack(outs, dim=0)
    elif x_cpu.ndim in (2, 3):
        y_cpu = denoise_one(x_cpu)
    else:
        raise ValueError("Expected (H,W), (C,H,W), or (N,C,H,W).")

    if clamp:
        y_cpu = y_cpu.clamp(0.0, 1.0)

    y = y_cpu.to(device=orig_device, dtype=orig_dtype)

    if y.shape != orig_shape:
        raise RuntimeError(f"Output shape {y.shape} != input shape {orig_shape}")

    return y

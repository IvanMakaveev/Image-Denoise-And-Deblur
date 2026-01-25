import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple, Union

def pca_denoise_patch_group(group: np.ndarray, sigma: float, hard_thresh: float = 2.7, eps: float = 1e-8) -> np.ndarray:
    mean = group.mean(axis=0, keepdims=True)
    X = group - mean

    U, S, Vt = np.linalg.svd(X, full_matrices=False)

    coeff = U * S

    T = hard_thresh * sigma
    coeff_d = coeff * (np.abs(coeff) > T)

    Xd = coeff_d @ Vt
    return Xd + mean


def pca_patch_denoise(
    noisy: np.ndarray,
    sigma: float,
    patch: int = 7,
    search_window: int = 21,
    group_size: int = 32,
    stride: int = 3,
    hard_thresh: float = 2.7,
) -> np.ndarray:
    y = np.asarray(noisy, dtype=np.float32)
    assert y.ndim == 2, "pca_patch_denoise expects grayscale image (H,W)."

    H, W = y.shape
    p = patch
    pad = p // 2
    half_sw = search_window // 2

    yp = np.pad(y, ((pad, pad), (pad, pad)), mode="reflect")

    out = np.zeros_like(yp, dtype=np.float32)
    wgt = np.zeros_like(yp, dtype=np.float32)

    all_patches = np.empty((H * W, p * p), dtype=np.float32)
    idx = 0
    for i in range(H):
        for j in range(W):
            all_patches[idx] = yp[i:i+p, j:j+p].reshape(-1)
            idx += 1

    def patch_index(i, j):
        return i * W + j

    for i in range(0, H, stride):
        for j in range(0, W, stride):
            ref = all_patches[patch_index(i, j)]

            i0 = max(i - half_sw, 0)
            i1 = min(i + half_sw, H - 1)
            j0 = max(j - half_sw, 0)
            j1 = min(j + half_sw, W - 1)

            M = (i1 - i0 + 1) * (j1 - j0 + 1)
            candidates = np.empty((M, p * p), dtype=np.float32)
            m = 0
            for ii in range(i0, i1 + 1):
                base = ii * W
                for jj in range(j0, j1 + 1):
                    candidates[m] = all_patches[base + jj]
                    m += 1

            diff = candidates - ref[None, :]
            d2 = np.sum(diff * diff, axis=1)
            nn = np.argsort(d2)[:group_size]
            group = candidates[nn]

            group_d = pca_denoise_patch_group(group, sigma=sigma, hard_thresh=hard_thresh)

            patch_d = group_d[0].reshape(p, p)
            out[i:i+p, j:j+p] += patch_d
            wgt[i:i+p, j:j+p] += 1.0

    den = out / np.maximum(wgt, 1e-8)
    den = den[pad:pad+H, pad:pad+W]
    return den.astype(np.float32)

def estimate_sigma_from_residual(residual: np.ndarray, method: str = "mad") -> float:
    r = np.asarray(residual, dtype=np.float32).reshape(-1)

    if method == "std":
        return float(np.std(r))

    if method == "mad":
        med = float(np.median(r))
        mad = float(np.median(np.abs(r - med)))
        # For Gaussian: sigma ≈ 1.4826 * MAD
        return 1.4826 * mad

    raise ValueError("method must be 'std' or 'mad'")

def lpg_pca_denoise_two_stage(
    noisy: np.ndarray,
    sigma: float,
    patch: int = 7,
    search_window: int = 21,
    group_size: int = 32,
    stride_stage1: int = 3,
    stride_stage2: int = 2,
    hard_thresh_stage1: float = 2.8,
    hard_thresh_stage2: float = 2.6,
    noise_est_method: str = "mad",
    sigma2_floor_ratio: float = 0.3,
    sigma2_cap_ratio: float = 1.0,
) -> tuple[np.ndarray, float]:
    y = np.asarray(noisy, dtype=np.float32)
    assert y.ndim == 2, "Start with grayscale image (H,W)."

    den1 = pca_patch_denoise(
        y, sigma=sigma,
        patch=patch, search_window=search_window, group_size=group_size,
        stride=stride_stage1, hard_thresh=hard_thresh_stage1
    )

    residual = y - den1
    sigma2 = estimate_sigma_from_residual(residual, method=noise_est_method)

    sigma2 = float(np.clip(sigma2, sigma * sigma2_floor_ratio, sigma * sigma2_cap_ratio))

    den2 = pca_patch_denoise(
        den1, sigma=sigma2,
        patch=patch, search_window=search_window, group_size=group_size,
        stride=stride_stage2, hard_thresh=hard_thresh_stage2
    )

    return den2.astype(np.float32), sigma2

def rgb_to_ycbcr(img_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(img_rgb, dtype=np.float32)
    R, G, B = x[..., 0], x[..., 1], x[..., 2]
    Y  = 0.299 * R + 0.587 * G + 0.114 * B
    Cb = -0.168736 * R - 0.331264 * G + 0.5 * B + 0.5
    Cr =  0.5 * R - 0.418688 * G - 0.081312 * B + 0.5
    return Y, Cb, Cr

def ycbcr_to_rgb(Y: np.ndarray, Cb: np.ndarray, Cr: np.ndarray) -> np.ndarray:
    Y = np.asarray(Y, dtype=np.float32)
    Cb0 = np.asarray(Cb, dtype=np.float32) - 0.5
    Cr0 = np.asarray(Cr, dtype=np.float32) - 0.5
    R = Y + 1.402 * Cr0
    G = Y - 0.344136 * Cb0 - 0.714136 * Cr0
    B = Y + 1.772 * Cb0
    return np.clip(np.stack([R, G, B], axis=-1), 0.0, 1.0).astype(np.float32)
    
def lpg_pca_denoise_rgb_yonly(
    noisy_rgb: np.ndarray,
    sigma: float,
    patch: int = 7,
    search_window: int = 21,
    group_size: int = 32,
    **kwargs
) -> tuple[np.ndarray, float]:
    x = np.asarray(noisy_rgb, dtype=np.float32)
    assert x.ndim == 3 and x.shape[2] == 3, "Expected RGB (H,W,3)"

    Y, Cb, Cr = rgb_to_ycbcr(x)

    Yd, sigma2 = lpg_pca_denoise_two_stage(
        Y,
        sigma=sigma,
        patch=patch,
        search_window=search_window,
        group_size=group_size,
        **kwargs
    )

    out_rgb = ycbcr_to_rgb(Yd, Cb, Cr)
    return out_rgb, sigma2

def _to_numpy_image01(x: torch.Tensor):
    meta = {
        "device": x.device,
        "dtype": x.dtype,
        "shape": tuple(x.shape),
        "ndim": x.ndim,
    }

    x_cpu = x.detach()
    if x_cpu.is_cuda:
        x_cpu = x_cpu.cpu()
    x_cpu = x_cpu.to(torch.float32)

    if x_cpu.ndim == 2:
        # (H,W)
        np_img = x_cpu.numpy().astype(np.float32)
        return np_img, meta, "HW"

    if x_cpu.ndim == 3:
        # (C,H,W) or (H,W,C) ??? We assume torch format is CHW.
        C, H, W = x_cpu.shape
        if C == 1:
            np_img = x_cpu[0].numpy().astype(np.float32)  # (H,W)
            return np_img, meta, "CHW1"
        if C == 3:
            np_img = x_cpu.permute(1, 2, 0).numpy().astype(np.float32)  # (H,W,3)
            return np_img, meta, "CHW3"
        raise ValueError(f"Expected C in {{1,3}}, got {C}")

    raise ValueError("Expected image tensor shape (H,W) or (C,H,W).")


def _from_numpy_image01(np_img: np.ndarray, meta, layout: str) -> torch.Tensor:
    out = torch.from_numpy(np_img.astype(np.float32))

    if layout == "HW":
        y = out
    elif layout == "CHW1":
        y = out.unsqueeze(0)
    elif layout == "CHW3":
        y = out.permute(2, 0, 1)
    else:
        raise RuntimeError(f"Unknown layout {layout}")

    y = y.to(device=meta["device"], dtype=meta["dtype"])

    if tuple(y.shape) != meta["shape"]:
        raise RuntimeError(f"Output shape {tuple(y.shape)} != input shape {meta['shape']}")

    return y


def lpg_pca_denoise_torch_gray(
    noisy: torch.Tensor,
    sigma: float,
    **kwargs
):
    np_img, meta, layout = _to_numpy_image01(noisy)

    if np_img.ndim != 2:
        raise ValueError("lpg_pca_denoise_torch_gray expects grayscale input (H,W) or (1,H,W).")

    den_np, sigma2 = lpg_pca_denoise_two_stage(np_img, sigma=sigma, **kwargs)

    den_t = _from_numpy_image01(den_np, meta, layout)
    return den_t, sigma2


def lpg_pca_denoise_torch_rgb_yonly(
    noisy: torch.Tensor,
    sigma: float,
    clamp: bool = True,
    **kwargs
):
    np_img, meta, layout = _to_numpy_image01(noisy)
    if np_img.ndim != 3 or np_img.shape[2] != 3:
        raise ValueError("Expected RGB tensor shape (3,H,W).")

    den_np, sigma2 = lpg_pca_denoise_rgb_yonly(np_img, sigma=sigma, **kwargs)

    if clamp:
        den_np = np.clip(den_np, 0.0, 1.0)

    den_t = _from_numpy_image01(den_np, meta, layout)
    return den_t, sigma2


def lpg_pca_denoise_torch_rgb_per_channel(
    noisy: torch.Tensor,
    sigma: float,
    clamp: bool = True,
    **kwargs
):
    if noisy.ndim != 3 or noisy.shape[0] != 3:
        raise ValueError("Expected RGB tensor shape (3,H,W).")

    meta = {"device": noisy.device, "dtype": noisy.dtype, "shape": tuple(noisy.shape)}
    x_cpu = noisy.detach()
    if x_cpu.is_cuda:
        x_cpu = x_cpu.cpu()
    x_cpu = x_cpu.to(torch.float32)

    outs = []
    sigma2s = []
    for c in range(3):
        den_c_np, sigma2 = lpg_pca_denoise_two_stage(x_cpu[c].numpy(), sigma=sigma, **kwargs)
        outs.append(torch.from_numpy(den_c_np))
        sigma2s.append(float(sigma2))

    y = torch.stack(outs, dim=0)
    if clamp:
        y = y.clamp(0.0, 1.0)

    y = y.to(device=meta["device"], dtype=meta["dtype"])
    if tuple(y.shape) != meta["shape"]:
        raise RuntimeError("Shape mismatch after denoising.")
    return y, float(np.mean(sigma2s))


def estimate_sigma(noisy: np.ndarray) -> float:
    y = np.asarray(noisy, dtype=np.float32)
    assert y.ndim == 2

    y00 = y[0::2, 0::2]
    y01 = y[0::2, 1::2]
    y10 = y[1::2, 0::2]
    y11 = y[1::2, 1::2]
    hh = (y00 - y01 - y10 + y11) * 0.5

    r = hh.reshape(-1)
    med = float(np.median(r))
    mad = float(np.median(np.abs(r - med)))
    return 1.4826 * mad

@torch.no_grad()
def estimate_sigma_torch_haar_mad(
    x: torch.Tensor,
    per_channel: bool = False,
    reduce_batch: str = "median",
) -> Union[float, torch.Tensor]:
    if x.ndim == 2:
        x = x[None, None, ...]
    elif x.ndim == 3:
        x = x[None, ...]  # CHW
    elif x.ndim != 4:
        raise ValueError("Expected (H,W), (C,H,W), or (N,C,H,W)")

    N, C, H, W = x.shape
    if C not in (1, 3):
        raise ValueError("Expected C in {1,3}")

    if x.dtype.is_floating_point:
        y = x.float()
    else:
        y = x.float()

    H2 = (H // 2) * 2
    W2 = (W // 2) * 2
    y = y[..., :H2, :W2]

    y00 = y[..., 0::2, 0::2]
    y01 = y[..., 0::2, 1::2]
    y10 = y[..., 1::2, 0::2]
    y11 = y[..., 1::2, 1::2]

    hh = (y00 - y01 - y10 + y11) * 0.5

    r = hh.reshape(N, C, -1)
    med = r.median(dim=-1).values
    mad = (r - med[..., None]).abs().median(dim=-1).values

    sigma = 1.4826 * mad

    if not per_channel:
        sigma = sigma.median(dim=1).values

    if reduce_batch == "none":
        return sigma
    if reduce_batch == "mean":
        return float(sigma.mean().item())
    if reduce_batch == "median":
        return float(sigma.median().item())

    raise ValueError("reduce_batch must be 'median', 'mean', or 'none'")
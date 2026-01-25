import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple, Union

def _extract_patches_view_2d(img: np.ndarray, patch: int) -> np.ndarray:
    H, W = img.shape
    p = patch
    H_out, W_out = H - p + 1, W - p + 1
    s0, s1 = img.strides
    shape = (H_out, W_out, p, p)
    strides = (s0, s1, s0, s1)
    return np.lib.stride_tricks.as_strided(img, shape=shape, strides=strides)


def _precompute_window_offsets(search_window: int) -> np.ndarray:
    r = search_window // 2
    di, dj = np.mgrid[-r:r+1, -r:r+1]
    return np.stack([di.reshape(-1), dj.reshape(-1)], axis=1).astype(np.int32)


def pca_denoise_patch_group(group: np.ndarray, sigma: float, hard_thresh: float = 2.7) -> np.ndarray:
    mean = group.mean(axis=0, keepdims=True)
    X = group - mean

    # PCA via SVD
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    coeff = U * S  # (K, r)

    T = hard_thresh * sigma
    coeff_d = coeff * (np.abs(coeff) > T)

    Xd = coeff_d @ Vt
    return Xd + mean

def pca_patch_denoise_fast(
    noisy: np.ndarray,
    sigma: float,
    patch: int = 7,
    search_window: int = 21,
    group_size: int = 32,
    stride: int = 3,
    hard_thresh: float = 2.7,
) -> np.ndarray:
    y = np.asarray(noisy, dtype=np.float32)
    assert y.ndim == 2, "pca_patch_denoise_fast expects grayscale (H,W)."
    assert patch % 2 == 1, "patch must be odd"
    assert search_window % 2 == 1, "search_window must be odd"

    H, W = y.shape
    p = patch
    pad = p // 2

    yp = np.pad(y, ((pad, pad), (pad, pad)), mode="reflect")

    patches_hwpp = _extract_patches_view_2d(yp, p)
    all_patches = patches_hwpp.reshape(H * W, p * p)
    all_patches = np.asarray(all_patches, dtype=np.float32, order="C")

    offsets = _precompute_window_offsets(search_window)
    M = offsets.shape[0]

    out = np.zeros_like(yp, dtype=np.float32)
    wgt = np.zeros_like(yp, dtype=np.float32)

    ref_is = np.arange(0, H, stride, dtype=np.int32)
    ref_js = np.arange(0, W, stride, dtype=np.int32)

    def lin(i, j):
        return i * W + j

    for i in ref_is:
        for j in ref_js:
            ref = all_patches[lin(i, j)]
            ci = i + offsets[:, 0]
            cj = j + offsets[:, 1]
            ci = np.clip(ci, 0, H - 1)
            cj = np.clip(cj, 0, W - 1)

            cand_idx = ci * W + cj
            candidates = all_patches[cand_idx]

            diff = candidates - ref[None, :]
            d2 = np.einsum("md,md->m", diff, diff)

            K = min(group_size, M)
            nn = np.argpartition(d2, K - 1)[:K]
            nn = nn[np.argsort(d2[nn])]

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
        return 1.4826 * mad
    raise ValueError("method must be 'std' or 'mad'")


def lpg_pca_denoise_two_stage_fast(
    noisy: np.ndarray,
    sigma: float,
    patch: int = 7,
    search_window: int = 21,
    group_size: int = 32,
    stride_stage1: int = 4,
    stride_stage2: int = 2,
    hard_thresh_stage1: float = 2.8,
    hard_thresh_stage2: float = 2.6,
    noise_est_method: str = "mad",
    sigma2_floor_ratio: float = 0.3,
    sigma2_cap_ratio: float = 1.0,
) -> tuple[np.ndarray, float]:
    y = np.asarray(noisy, dtype=np.float32)
    assert y.ndim == 2, "Expected grayscale (H,W)."

    den1 = pca_patch_denoise_fast(
        y, sigma=sigma,
        patch=patch, search_window=search_window, group_size=group_size,
        stride=stride_stage1, hard_thresh=hard_thresh_stage1
    )

    residual = y - den1
    sigma2 = estimate_sigma_from_residual(residual, method=noise_est_method)
    sigma2 = float(np.clip(sigma2, sigma * sigma2_floor_ratio, sigma * sigma2_cap_ratio))

    den2 = pca_patch_denoise_fast(
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

def lpg_pca_denoise_rgb_yonly_fast(
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

    Yd, sigma2 = lpg_pca_denoise_two_stage_fast(
        Y,
        sigma=sigma,
        patch=patch,
        search_window=search_window,
        group_size=group_size,
        **kwargs
    )

    out_rgb = ycbcr_to_rgb(Yd, Cb, Cr)
    return out_rgb, sigma2

def _torch_to_numpy_image01(x: torch.Tensor):
    meta = {
        "device": x.device,
        "dtype": x.dtype,
        "shape": tuple(x.shape),
    }

    x_cpu = x.detach()
    if x_cpu.is_cuda:
        x_cpu = x_cpu.cpu()
    x_cpu = x_cpu.to(torch.float32)

    if x_cpu.ndim == 2:
        # (H,W)
        return x_cpu.numpy().astype(np.float32), meta, "HW"

    if x_cpu.ndim == 3:
        C, H, W = x_cpu.shape
        if C == 1:
            return x_cpu[0].numpy().astype(np.float32), meta, "CHW1"  # -> (H,W)
        if C == 3:
            return x_cpu.permute(1, 2, 0).numpy().astype(np.float32), meta, "CHW3"  # -> (H,W,3)
        raise ValueError(f"Expected C in {{1,3}}, got C={C}")

    raise ValueError("Expected (H,W), (1,H,W), or (3,H,W) torch tensor.")


def _numpy_to_torch_image01(np_img: np.ndarray, meta, layout: str) -> torch.Tensor:
    t = torch.from_numpy(np_img.astype(np.float32))

    if layout == "HW":
        y = t
    elif layout == "CHW1":
        y = t.unsqueeze(0)
    elif layout == "CHW3":
        y = t.permute(2, 0, 1)
    else:
        raise RuntimeError(f"Unknown layout '{layout}'")

    y = y.to(device=meta["device"], dtype=meta["dtype"])

    if tuple(y.shape) != meta["shape"]:
        raise RuntimeError(f"Output shape {tuple(y.shape)} != input shape {meta['shape']}")

    return y


def lpg_pca_denoise_torch_gray_fast(
    noisy: torch.Tensor,
    sigma: float,
    clamp: bool = True,
    **kwargs
):
    np_img, meta, layout = _torch_to_numpy_image01(noisy)

    if np_img.ndim != 2:
        raise ValueError("Expected grayscale input (H,W) or (1,H,W).")

    den_np, sigma2 = lpg_pca_denoise_two_stage_fast(np_img, sigma=sigma, **kwargs)

    if clamp:
        den_np = np.clip(den_np, 0.0, 1.0)

    den_t = _numpy_to_torch_image01(den_np, meta, layout)
    return den_t, float(sigma2)


def lpg_pca_denoise_torch_rgb_yonly_fast(
    noisy: torch.Tensor,
    sigma: float,
    clamp: bool = True,
    **kwargs
):
    np_img, meta, layout = _torch_to_numpy_image01(noisy)

    if np_img.ndim != 3 or np_img.shape[2] != 3:
        raise ValueError("Expected RGB input (3,H,W).")

    den_np, sigma2 = lpg_pca_denoise_rgb_yonly_fast(np_img, sigma=sigma, **kwargs)

    if clamp:
        den_np = np.clip(den_np, 0.0, 1.0)

    den_t = _numpy_to_torch_image01(den_np, meta, layout)
    return den_t, float(sigma2)

def lpg_pca_denoise_torch_batched_fast(
    noisy: torch.Tensor,
    sigma: float,
    y_only_rgb: bool = True,
    clamp: bool = True,
    **kwargs
):
    if noisy.ndim != 4:
        raise ValueError("Expected batched tensor (N,C,H,W).")

    N, C, H, W = noisy.shape
    outs = []
    sigmas2 = []

    for i in range(N):
        if C == 1:
            den_i, s2 = lpg_pca_denoise_torch_gray_fast(noisy[i], sigma=sigma, clamp=clamp, **kwargs)
        elif C == 3:
            if y_only_rgb:
                den_i, s2 = lpg_pca_denoise_torch_rgb_yonly_fast(noisy[i], sigma=sigma, clamp=clamp, **kwargs)
            else:
                chans = []
                s2s = []
                for c in range(3):
                    dc, sc = lpg_pca_denoise_torch_gray_fast(noisy[i, c], sigma=sigma, clamp=clamp, **kwargs)
                    chans.append(dc)
                    s2s.append(sc)
                den_i = torch.stack(chans, dim=0)
                s2 = float(np.mean(s2s))
        else:
            raise ValueError(f"Expected C in {{1,3}}, got C={C}")

        outs.append(den_i)
        sigmas2.append(s2)

    den = torch.stack(outs, dim=0)
    return den, sigmas2

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
import torch

def add_awgn(img: torch.Tensor, sigma: float, seed: int = 0) -> torch.Tensor:
    g = torch.Generator(device=img.device).manual_seed(seed)
    noise = torch.randn(
        img.shape,
        device=img.device,
        dtype=img.dtype,
        generator=g
    ) * sigma
    return (img + noise).clamp(0.0, 1.0)

def add_salt_pepper(
    img: torch.Tensor,
    amount: float = 0.02,
    salt_vs_pepper: float = 0.5,
    seed: int = 0
) -> torch.Tensor:
    g = torch.Generator(device=img.device).manual_seed(seed)
    out = img.clone()

    _, _, H, W = img.shape
    num = int(amount * H * W)

    idx = torch.randperm(H * W, generator=g, device=img.device)[:num]
    ns = int(num * salt_vs_pepper)

    flat = out.view(3, -1)

    flat[:, idx[:ns]] = 1.0  # salt
    flat[:, idx[ns:]] = 0.0  # pepper

    return out

def add_speckle_noise(img: torch.Tensor, sigma: float, seed: int = 0) -> torch.Tensor:
    g = torch.Generator(device=img.device).manual_seed(seed)
    noise = torch.randn(
        img.shape,
        device=img.device,
        dtype=img.dtype,
        generator=g
    ) * sigma
    return (img + img * noise).clamp(0.0, 1.0)

def add_poisson_noise(
    img: torch.Tensor,
    peak: float = 1.0,
    seed: int | None = None
) -> torch.Tensor:
    x = img.clamp(0.0, 1.0)
    lam = x * peak

    if seed is not None:
        torch.manual_seed(seed)

    y = torch.poisson(lam) / peak
    return y.clamp(0.0, 1.0)


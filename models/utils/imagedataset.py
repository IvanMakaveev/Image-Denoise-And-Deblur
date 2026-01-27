import torch
import torch.nn.functional as F

from pathlib import Path
from typing import List, Tuple

from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image, ImageReadMode
from torchvision.transforms import functional as TF

def list_images(root: Path) -> List[Path]:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    files = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            files.append(p)
    return sorted(files)


class PairedImageFolder(Dataset):
    """
    Expects:
      noisy_root/xxx/yyy.png
      gt_root/xxx/yyy.png
    Same relative path under both roots.
    """
    def __init__(
        self,
        noisy_root: str,
        gt_root: str,
        patch_size: int = 256,
        train: bool = True,
    ):
        super().__init__()
        self.noisy_root = Path(noisy_root)
        self.gt_root = Path(gt_root)
        self.patch = patch_size
        self.train = train

        self.noisy_files = list_images(self.noisy_root)
        if len(self.noisy_files) == 0:
            raise RuntimeError(f"No images found under noisy_root: {self.noisy_root}")

        pairs: List[Tuple[Path, Path]] = []
        for nf in self.noisy_files:
            rel = nf.relative_to(self.noisy_root)
            gf = self.gt_root / rel
            if gf.exists():
                pairs.append((nf, gf))
        if len(pairs) == 0:
            raise RuntimeError(
                "Found 0 paired files. Make sure gt_root mirrors noisy_root relative paths.\n"
                f"noisy_root={self.noisy_root}\n"
                f"gt_root={self.gt_root}"
            )
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def _read_rgb_float(self, path: Path) -> torch.Tensor:
        x = read_image(str(path), mode=ImageReadMode.RGB).float() / 255.0
        return x

    def _random_crop_pair(self, a: torch.Tensor, b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        _, h, w = a.shape
        ps = self.patch

        pad_h = max(0, ps - h)
        pad_w = max(0, ps - w)
        if pad_h > 0 or pad_w > 0:
            a = F.pad(a, (0, pad_w, 0, pad_h), mode="reflect")
            b = F.pad(b, (0, pad_w, 0, pad_h), mode="reflect")
            _, h, w = a.shape

        top = torch.randint(0, h - ps + 1, (1,)).item()
        left = torch.randint(0, w - ps + 1, (1,)).item()
        a = a[:, top:top + ps, left:left + ps]
        b = b[:, top:top + ps, left:left + ps]
        return a, b

    def _augment_pair(self, a: torch.Tensor, b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if torch.rand(()) < 0.5:
            a = TF.hflip(a); b = TF.hflip(b)
        if torch.rand(()) < 0.5:
            a = TF.vflip(a); b = TF.vflip(b)
        k = torch.randint(0, 4, (1,)).item()
        if k:
            a = torch.rot90(a, k, dims=[1, 2])
            b = torch.rot90(b, k, dims=[1, 2])
        return a, b

    def __getitem__(self, idx: int):
        nf, gf = self.pairs[idx]
        noisy = self._read_rgb_float(nf)
        gt = self._read_rgb_float(gf)

        if self.train:
            noisy, gt = self._random_crop_pair(noisy, gt)
            noisy, gt = self._augment_pair(noisy, gt)

        return noisy, gt
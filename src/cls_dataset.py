import math
from pathlib import Path
from typing import Optional, Tuple

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class CycloneH5Dataset(Dataset):
    """
    Fork-safe HDF5 dataset reading from data/processed/cls_h5_448/{train,val,test}.h5.
    Applies random 360° rotation + physical center jitter during training to simulate
    YOLOv26 localization noise, then crops 448x448 (±600 km) -> 384x384 (±500 km).
    """

    def __init__(
        self,
        h5_path: Path = Path("./../data/processed/cls_h5_448/train.h5"),
        out_size: int = 384,
        is_train: bool = True,
        max_jitter_km: float = 35.0,
    ):
        self.h5_path = Path(h5_path)
        self.out_size = out_size
        self.is_train = is_train
        self._hf: Optional[h5py.File] = None

        with h5py.File(self.h5_path, "r") as f:
            self.length = int(f["images"].shape[0])
            self.padded_size = int(f["images"].shape[1])
            self.padded_crop_km = float(f.attrs.get("padded_crop_km", 600.0))
            self.class_ids = f["class_id"][:].astype(np.int64)
            self.wind_kts = f["wind_kts"][:].astype(np.float32)

        km_per_px = (2.0 * self.padded_crop_km) / float(self.padded_size)
        max_border_px = (self.padded_size - self.out_size) // 2
        self.max_jitter_px = min(max_border_px, int(round(max_jitter_km / km_per_px)))

    def _ensure_open(self):
        if self._hf is None:
            self._hf = h5py.File(self.h5_path, "r", swmr=True, rdcc_nbytes=64 * 1024 * 1024)

    def get_sample_weights(self, power: float = 0.5) -> torch.DoubleTensor:
        """Compute per-sample weights (inverse frequency^power) for WeightedRandomSampler."""
        counts = np.bincount(self.class_ids, minlength=7).astype(np.float64)
        class_weights = 1.0 / np.maximum(counts, 1.0) ** power
        return torch.from_numpy(class_weights[self.class_ids])

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, float]:
        self._ensure_open()
        patch = self._hf["images"][idx].astype(np.float32)  # (448, 448), Tb_K * 100

        if self.is_train:
            angle = np.random.uniform(0.0, 360.0)
            rad = math.radians(angle)
            cos_a, sin_a = math.cos(rad), math.sin(rad)

            jx = np.random.randint(-self.max_jitter_px, self.max_jitter_px + 1)
            jy = np.random.randint(-self.max_jitter_px, self.max_jitter_px + 1)

            scale = self.out_size / float(self.padded_size)
            tx = (2.0 * jx) / float(self.padded_size)
            ty = (2.0 * jy) / float(self.padded_size)

            theta = torch.tensor(
                [[[scale * cos_a, -scale * sin_a, tx], [scale * sin_a, scale * cos_a, ty]]],
                dtype=torch.float32,
            )
            grid = F.affine_grid(theta, size=(1, 1, self.out_size, self.out_size), align_corners=False)
            crop = F.grid_sample(
                torch.from_numpy(patch)[None, None],
                grid,
                mode="bilinear",
                padding_mode="border",
                align_corners=False,
            ).squeeze(0).squeeze(0)
        else:
            start = (self.padded_size - self.out_size) // 2
            crop = torch.from_numpy(patch[start : start + self.out_size, start : start + self.out_size])

        return crop, int(self.class_ids[idx]), float(self.wind_kts[idx])


class GPUCycloneTensorBuilder(torch.nn.Module):
    """
    Converts a batch of (B, 384, 384) Tb*100 tensors into ImageNet-standardized
    (B, 3, 384, 384) float32 tensors on GPU for convnext_tiny.fb_in22k_ft_in1k_384:
      Ch 0: Cold-topped normalized Tb: clip((310 - Tb_K) / 130, 0, 1)
      Ch 1: Normalized physical radial distance [0, 1]
      Ch 2: Normalized physical spatial gradient |∇Tb| (K/km) in [0, 1]
    """

    def __init__(self, img_size: int = 384, crop_km: float = 500.0, imagenet_norm: bool = True):
        super().__init__()
        self.imagenet_norm = imagenet_norm
        km_per_px = (2.0 * crop_km) / float(img_size)

        coords = torch.arange(img_size, dtype=torch.float32) - (img_size - 1) / 2.0
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        radial = torch.clamp(torch.sqrt(xx ** 2 + yy ** 2) / (img_size / 2.0), 0.0, 1.0)
        self.register_buffer("radial_map", radial[None, None])

        sobel_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]) / (8.0 * km_per_px)
        sobel_y = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]) / (8.0 * km_per_px)
        self.register_buffer("sobel_x", sobel_x.view(1, 1, 3, 3))
        self.register_buffer("sobel_y", sobel_y.view(1, 1, 3, 3))

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    @torch.no_grad()
    def forward(self, tb_x100: torch.Tensor) -> torch.Tensor:
        tb_k = tb_x100.unsqueeze(1) * 0.01
        norm_tb = torch.clamp((310.0 - tb_k) / 130.0, 0.0, 1.0)

        tb_pad = F.pad(tb_k, (1, 1, 1, 1), mode="replicate")
        gx = F.conv2d(tb_pad, self.sobel_x)
        gy = F.conv2d(tb_pad, self.sobel_y)
        grad_k_km = torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)
        norm_grad = torch.clamp(grad_k_km / 5.0, 0.0, 1.0)

        radial_batch = self.radial_map.expand(tb_k.shape[0], -1, -1, -1)
        x = torch.cat([norm_tb, radial_batch, norm_grad], dim=1)

        if self.imagenet_norm:
            x = (x - self.mean) / self.std
        return x

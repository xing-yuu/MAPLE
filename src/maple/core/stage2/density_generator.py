from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


class Stage2NPZDataset(Dataset):
    """Dataset for samples produced by maple.training.prepare_stage2_training_data."""

    def __init__(
        self,
        manifest_csv: Union[str, Path],
        *,
        target_key: str = "rho_target_32",
        init_key: Optional[str] = None,
        require_target: bool = True,
        require_init: bool = False,
    ) -> None:
        self.manifest_csv = Path(manifest_csv)
        self.target_key = str(target_key)
        self.init_key = None if init_key in (None, "", "none", "None") else str(init_key)
        self.require_target = bool(require_target)
        self.require_init = bool(require_init)

        with self.manifest_csv.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))

        self.rows = []
        for row in rows:
            sample_path = Path(row["stage2_npz"])
            if not sample_path.is_absolute():
                sample_path = self.manifest_csv.parent / sample_path
            if self.require_target and str(row.get("target_available", "0")) != "1":
                continue
            self.rows.append({**row, "stage2_npz": str(sample_path)})

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self.rows[index]
        with np.load(row["stage2_npz"]) as data:
            condition = torch.from_numpy(data["condition"].astype(np.float32))
            item: Dict[str, Any] = {
                "condition": condition,
                "stage2_npz": row["stage2_npz"],
            }
            if self.init_key is not None:
                if self.init_key in data:
                    item["rho_init"] = torch.from_numpy(data[self.init_key].astype(np.float32))
                elif self.require_init:
                    raise KeyError(f"{self.init_key!r} is missing from {row['stage2_npz']}")
            if self.target_key in data:
                item["rho_target"] = torch.from_numpy(data[self.target_key].astype(np.float32))
        return item


class ConvBlock3d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        groups = min(8, out_channels)
        while out_channels % groups != 0:
            groups -= 1
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConditionEncoder(nn.Module):
    def __init__(self, cond_dim: int, latent_dim: int = 256, hidden_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, condition: torch.Tensor) -> torch.Tensor:
        return self.net(condition)


class NoiseDensityEncoder(nn.Module):
    """Legacy encoder for the old G(condition, rho_init) training path."""

    def __init__(self, in_channels: int = 1, latent_dim: int = 256, base_channels: int = 32) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            ConvBlock3d(in_channels, base_channels),
            nn.AvgPool3d(kernel_size=2),
            ConvBlock3d(base_channels, base_channels * 2),
            nn.AvgPool3d(kernel_size=2),
            ConvBlock3d(base_channels * 2, base_channels * 4),
            nn.AvgPool3d(kernel_size=2),
            ConvBlock3d(base_channels * 4, base_channels * 4),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
        )
        self.fc = nn.Linear(base_channels * 4, latent_dim)

    def forward(self, rho_init: torch.Tensor) -> torch.Tensor:
        x = self.conv(rho_init).flatten(1)
        return self.fc(x)


class Stage2DensityGenerator(nn.Module):
    """Conditional 3D density generator for the second-stage warm-start network.

    Input:
      condition: [w_stiffness, w_shear, w_thermal, vf, stiffness, shear, thermal]
      rho_init: optional legacy smoothed seed density at 32^3 when use_noise_encoder=True.

    Output:
      rho_net: (B, 1, 32, 32, 32), constrained to [0, 1].
    """

    def __init__(
        self,
        cond_dim: int = 7,
        out_res: int = 32,
        base_res: int = 4,
        latent_dim: int = 256,
        base_channels: int = 128,
        use_noise_encoder: bool = False,
    ) -> None:
        super().__init__()
        if out_res % base_res != 0:
            raise ValueError(f"out_res must be divisible by base_res, got {out_res} and {base_res}")
        upsample_levels = int(round(np.log2(out_res // base_res)))
        if base_res * (2 ** upsample_levels) != out_res:
            raise ValueError(f"out_res/base_res must be a power of two, got {out_res}/{base_res}")

        self.cond_dim = int(cond_dim)
        self.out_res = int(out_res)
        self.base_res = int(base_res)
        self.latent_dim = int(latent_dim)
        self.base_channels = int(base_channels)
        self.use_noise_encoder = bool(use_noise_encoder)

        self.condition_encoder = ConditionEncoder(cond_dim=cond_dim, latent_dim=latent_dim)
        if self.use_noise_encoder:
            self.noise_encoder = NoiseDensityEncoder(in_channels=1, latent_dim=latent_dim)
            fusion_dim = latent_dim * 2
        else:
            self.noise_encoder = None
            fusion_dim = latent_dim

        self.fc = nn.Sequential(
            nn.Linear(fusion_dim, base_channels * base_res * base_res * base_res),
            nn.SiLU(inplace=True),
        )

        blocks = []
        in_ch = base_channels
        for _ in range(upsample_levels):
            out_ch = max(16, in_ch // 2)
            blocks.append(nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False))
            blocks.append(ConvBlock3d(in_ch, out_ch))
            in_ch = out_ch
        self.decoder = nn.Sequential(*blocks)
        self.out = nn.Conv3d(in_ch, 1, kernel_size=1)

    def encode_latent(self, condition: torch.Tensor, rho_init: Optional[torch.Tensor] = None) -> torch.Tensor:
        cond_latent = self.condition_encoder(condition)
        if self.use_noise_encoder:
            if rho_init is None:
                raise ValueError("rho_init is required when use_noise_encoder=True")
            noise_latent = self.noise_encoder(rho_init)
            return torch.cat([cond_latent, noise_latent], dim=1)
        return cond_latent

    def decode_latent(self, latent: torch.Tensor) -> torch.Tensor:
        x = self.fc(latent)
        x = x.reshape(latent.shape[0], self.base_channels, self.base_res, self.base_res, self.base_res)
        x = self.decoder(x)
        return torch.sigmoid(self.out(x))

    def forward(self, condition: torch.Tensor, rho_init: Optional[torch.Tensor] = None) -> torch.Tensor:
        latent = self.encode_latent(condition, rho_init)
        return self.decode_latent(latent)


def density_smoothness_loss(rho: torch.Tensor) -> torch.Tensor:
    dx = rho[:, :, 1:, :, :] - rho[:, :, :-1, :, :]
    dy = rho[:, :, :, 1:, :] - rho[:, :, :, :-1, :]
    dz = rho[:, :, :, :, 1:] - rho[:, :, :, :, :-1]
    return dx.square().mean() + dy.square().mean() + dz.square().mean()


def density_generation_loss(
    rho_pred: torch.Tensor,
    rho_target: torch.Tensor,
    vf_target: torch.Tensor,
    *,
    alpha_vol: float = 1.0,
    beta_smooth: float = 1.0e-3,
) -> Dict[str, torch.Tensor]:
    rec = F.mse_loss(rho_pred, rho_target)
    vf = vf_target.reshape(-1).to(device=rho_pred.device, dtype=rho_pred.dtype)
    pred_vf = rho_pred.mean(dim=(1, 2, 3, 4))
    vol = (pred_vf - vf).abs().mean()
    smooth = density_smoothness_loss(rho_pred)
    total = rec + float(alpha_vol) * vol + float(beta_smooth) * smooth
    return {
        "loss": total,
        "loss_rec": rec,
        "loss_vol": vol,
        "loss_smooth": smooth,
    }

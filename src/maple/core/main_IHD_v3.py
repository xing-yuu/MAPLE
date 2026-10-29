"""
Inverse homogenization topology optimization entrypoint (CLI-only).

Changes vs previous main_IHD.py:
- No config files (OmegaConf removed).
- No folder/batch mode. Single run only.
- Two initialization modes:
    1) voxel: from a voxel .npy
    2) noise: random noise initialization with target volume fraction
- Exports a 3D mesh every N iterations (default: 10).
- Writes an iteration log (CSV + JSON summary): iteration, volume fraction, objective, etc.

Typical usage
-------------
# (A) init from voxel .npy (shape: (r,r,r) or flat r^3, values 0/1)
python -m maple.core.main_IHD_v3 --init voxel --voxel_npy ./input.npy --output_dir ./out

# (B) init from random noise
python -m maple.core.main_IHD_v3 --init noise --resolution 64 --volfrac 0.30 --seed 0 --output_dir ./out
"""
from __future__ import annotations

import os
import time
import json
import csv
import argparse
import random
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import torch
import torch.nn.functional as F
# marching cubes backend
try:
    import mcubes  # PyMCubes
    _HAS_MCUBES = True
except Exception:  # pragma: no cover
    mcubes = None
    _HAS_MCUBES = False
    from skimage import measure  # type: ignore

# project imports
from maple.core.utils._utils import assemble_F_fast, assemble_F_fast_scalar
from maple.core.utils.model import IHD
from maple.core.utils.io import save_mesh_from_rho
from maple.core.utils.objective import (
    objective_from_CH,
    objective_from_spec,
    objective_requires,
    parse_objective_spec,
    shear_objective,
    stiffness_objective,
    thermal_objective,
)
from maple.core.utils.printability import printability_penalty_overhang, cavity_penalty_closed_void, cavity_penalty_periodic_array
# -----------------------------
# Reproducibility
# -----------------------------
def setup_seed(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False




# -----------------------------
# Objective helpers
# -----------------------------

def _objective_needs_mech(obj: str) -> bool:
    need_ch, _ = objective_requires(obj)
    return bool(need_ch)


def _objective_needs_thermal(obj: str) -> bool:
    _, need_kappa = objective_requires(obj)
    return bool(need_kappa)



# -----------------------------
# Periodicity helpers
# -----------------------------
def _enforce_periodic_vol_(vol: torch.Tensor) -> torch.Tensor:
    """Legacy no-op kept for backward compatibility.

    Periodicity is already enforced by wrapped indexing in the FE assembly. Mutating
    the boundary slices here would change the user's geometry and effective volume
    fraction, so we intentionally leave the field unchanged.
    """
    return vol


def _parse_tiles_xyz(tiles_xyz: str | None, tiles: int) -> tuple[int, int, int]:
    """Parse tiling specification for array-cavity detection.

    tiles_xyz: string like "3,3,3" or "3 3 3"
    tiles: fallback single integer -> (tiles,tiles,tiles)
    """
    if tiles_xyz is None:
        t = int(tiles)
        return max(1, t), max(1, t), max(1, t)
    txt = str(tiles_xyz).replace(" ", ",")
    parts = [p for p in txt.split(",") if p != ""]
    if len(parts) != 3:
        raise ValueError("--cavity_tiles_xyz must have 3 integers, e.g. 3,3,3")
    tx, ty, tz = (max(1, int(parts[0])), max(1, int(parts[1])), max(1, int(parts[2])))
    return tx, ty, tz


def _resample_voxel_nearest(voxel: np.ndarray, out_res: int) -> np.ndarray:
    """Nearest-neighbor resample for boolean voxel grids."""
    voxel = voxel.astype(np.uint8)
    in_res = voxel.shape[0]
    if voxel.shape != (in_res, in_res, in_res):
        raise ValueError(f"voxel must be cubic, got {voxel.shape}")
    if in_res == out_res:
        return voxel.astype(bool)
    # map each output index to input index
    idx = (np.linspace(0, in_res - 1, out_res)).round().astype(int)
    out = voxel[np.ix_(idx, idx, idx)]
    return out.astype(bool)


# -----------------------------
# Unit-cell batch builder (periodic)
# -----------------------------
def _get_node_density(rho_grid: torch.Tensor) -> torch.Tensor:
    """
    rho_grid: (r,r,r) element-centered density.
    Returns node density features: (r,r,r,8) for each element center treated as "node" in this formulation.
    """
    r = rho_grid.shape[0]
    rho = rho_grid
    node_den = torch.empty((r, r, r, 8), device=rho.device, dtype=rho.dtype)

    # 8 corner offsets
    offs = torch.tensor(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]],
        dtype=torch.long,
        device=rho.device,
    )
    gx = torch.arange(r, device=rho.device)
    gy = torch.arange(r, device=rho.device)
    gz = torch.arange(r, device=rho.device)
    X, Y, Z = torch.meshgrid(gx, gy, gz, indexing="ij")
    base = torch.stack([X, Y, Z], dim=-1)  # (r,r,r,3)

    for i in range(8):
        coo = (base + offs[i] + r) % r
        node_den[..., i] = rho[coo[..., 0], coo[..., 1], coo[..., 2]]
    return node_den


def build_unitcell_batch(
    device: torch.device,
    res: int,
    rho_grid: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """
    Build a single periodic unit-cell batch dictionary.
    rho_grid: (res,res,res) float tensor on device
    """
    if rho_grid.shape != (res, res, res):
        raise ValueError(f"rho_grid must be {(res,res,res)}, got {tuple(rho_grid.shape)}")

    # full-grid element coordinates (res^3, 3)
    grid = torch.stack(
        torch.meshgrid(
            torch.arange(res, device=device),
            torch.arange(res, device=device),
            torch.arange(res, device=device),
            indexing="ij",
        ),
        dim=-1,
    ).reshape(-1, 3).long()

    # node ids (periodic => res^3 unique nodes)
    node_id_grid = torch.arange(res ** 3, device=device, dtype=torch.int64).reshape(res, res, res)

    # element->node connectivity (8)
    hex8 = torch.tensor(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]],
        dtype=torch.int64,
        device=device,
    )
    node_index = torch.empty((grid.shape[0], 8), dtype=torch.int64, device=device)
    for i in range(8):
        coo = (grid + hex8[i] + res) % res
        node_index[:, i] = node_id_grid[coo[:, 0], coo[:, 1], coo[:, 2]]

    # features (res^3, 8)
    feats = _get_node_density(rho_grid).reshape(-1, 8).contiguous()

    # batch indices (single sample => 0)
    b0 = torch.zeros((grid.shape[0], 1), dtype=torch.long, device=device)
    coord_bxyz = torch.cat([b0, grid], dim=1)       # (N,4)
    elem_bxyz = torch.cat([b0, grid], dim=1)        # (N,4) (same here)

    batch = {
        "coord": coord_bxyz,
        "feature": feats.to(dtype=torch.float32),
        "node_index": node_index,
        "elem_coords": elem_bxyz,
        "rho": rho_grid.reshape(-1, 1).to(dtype=torch.float32),
        "voxel_batch_id": torch.zeros((node_index.shape[0],), dtype=torch.long, device=device),
    }
    return batch


 # -----------------------------
# Symmetry projection
# -----------------------------
def parse_symmetry(symmetry: str) -> Tuple[str, List[int]]:
    """Parse symmetry option.

    Returns:
      kind: "none" | "mirror" | "cubic"
      axes: for mirror kind, a sorted list of axes to mirror about (0=x,1=y,2=z)

    Semantics:
      - mirror symmetry enforces invariance under reflections about center planes.
        Example: "xy" mirrors x and y, producing 1/4 symmetry in xy and leaving z free.
        "octant"/"xyz" mirrors x,y,z -> classic 1/8 symmetry (8 identical octants).
      - cubic symmetry ("cubic"/"iso") enforces full cube (octahedral) symmetry:
        all 48 axis permutations + reflections. This implies x/y/z directions are identical.
    """
    sym = (symmetry or "none").lower().strip()
    if sym in ("none", ""):
        return "none", []

    # aliases
    if sym in ("octant", "1/8", "one_eighth"):
        sym = "xyz"
    if sym in ("cubic", "iso", "isotropic"):
        return "cubic", [0, 1, 2]

    # mirror axes (x/y/z combos)
    axes: List[int] = []
    if "x" in sym:
        axes.append(0)
    if "y" in sym:
        axes.append(1)
    if "z" in sym:
        axes.append(2)
    axes = sorted(set(axes))
    if not axes:
        return "none", []
    return "mirror", axes


def symmetrize_vol_mirror(vol: torch.Tensor, axes: List[int]) -> torch.Tensor:
    """Mirror-symmetrize a cubic volume (r,r,r) about selected center planes."""
    out = vol
    for ax in axes:
        out = 0.5 * (out + torch.flip(out, dims=[ax]))
    return out


def symmetrize_vol_cubic(vol: torch.Tensor) -> torch.Tensor:
    """Project a cubic volume (r,r,r) to full cube (octahedral) symmetry.

    This enforces invariance under all 48 symmetries of the cube:
      - 6 permutations of axes
      - 8 combinations of reflections
    Result: x/y/z directions become identical (isotropic symmetry).
    """
    import itertools

    out = torch.zeros_like(vol)
    cnt = 0
    perms = list(itertools.permutations((0, 1, 2), 3))
    flips = list(itertools.product((0, 1), repeat=3))  # 0=no flip, 1=flip
    for p in perms:
        v = vol.permute(p)
        for fx, fy, fz in flips:
            vv = v
            dims = []
            if fx:
                dims.append(0)
            if fy:
                dims.append(1)
            if fz:
                dims.append(2)
            if dims:
                vv = torch.flip(vv, dims=dims)
            out = out + vv
            cnt += 1
    return out / float(cnt)


def vec_to_vol_by_elemcoords(vec: torch.Tensor, elem_coords: torch.Tensor, res: int, device: torch.device) -> torch.Tensor:
    """vec: (N,1), elem_coords: (N,4) with columns [b,x,y,z]. Return vol (res,res,res)."""
    vol = torch.zeros((res, res, res), device=device, dtype=vec.dtype)
    xyz = elem_coords[:, 1:4].long()
    vol[xyz[:, 0], xyz[:, 1], xyz[:, 2]] = vec.reshape(-1)
    return vol


def vol_to_vec_by_elemcoords(vol: torch.Tensor, elem_coords: torch.Tensor) -> torch.Tensor:
    xyz = elem_coords[:, 1:4].long()
    out = vol[xyz[:, 0], xyz[:, 1], xyz[:, 2]].reshape(-1, 1)
    return out


def symmetrize_vec_mirror(x_vec: torch.Tensor, elem_coords: torch.Tensor, res: int, axes: List[int]) -> torch.Tensor:
    if not axes:
        return x_vec.reshape(-1, 1)
    x_vec = x_vec.reshape(-1, 1)
    vol = vec_to_vol_by_elemcoords(x_vec, elem_coords, res, x_vec.device)
    vol_sym = symmetrize_vol_mirror(vol, axes)
    return vol_to_vec_by_elemcoords(vol_sym, elem_coords)


def symmetrize_vec_cubic(x_vec: torch.Tensor, elem_coords: torch.Tensor, res: int) -> torch.Tensor:
    x_vec = x_vec.reshape(-1, 1)
    vol = vec_to_vol_by_elemcoords(x_vec, elem_coords, res, x_vec.device)
    vol_sym = symmetrize_vol_cubic(vol)
    return vol_to_vec_by_elemcoords(vol_sym, elem_coords)


def symmetrize_vec(x_vec: torch.Tensor, elem_coords: torch.Tensor, res: int, kind: str, axes: List[int]) -> torch.Tensor:
    """Unified symmetry projection for design variables."""
    if kind == "none":
        return x_vec.reshape(-1, 1)
    if kind == "mirror":
        return symmetrize_vec_mirror(x_vec, elem_coords, res=res, axes=axes)
    if kind == "cubic":
        return symmetrize_vec_cubic(x_vec, elem_coords, res=res)
    raise ValueError(f"Unknown symmetry kind: {kind}")

# -----------------------------
# Filter + OC update
# -----------------------------
def avg_filter3d_x_periodic(x_vec: torch.Tensor, elem_coords: torch.Tensor, res: int, k: int = 3) -> torch.Tensor:
    """
    Periodic box filter for (N,1) vector laid out on (res,res,res).
    """
    if int(k) <= 0:
        raise ValueError(f"filter size must be positive, got {k}")
    if int(k) % 2 == 0:
        raise ValueError(f"filter size must be odd, got {k}")
    x_vec = x_vec.reshape(-1, 1)
    vol = vec_to_vol_by_elemcoords(x_vec, elem_coords, res, x_vec.device)  # (r,r,r)
    vol = vol[None, None]  # (1,1,r,r,r)
    pad = k // 2
    # circular pad to make it periodic
    vol_p = F.pad(vol, (pad, pad, pad, pad, pad, pad), mode="circular")
    ker = torch.ones((1, 1, k, k, k), device=vol_p.device, dtype=vol_p.dtype) / float(k ** 3)
    vol_f = F.conv3d(vol_p, ker)
    vol_f = vol_f[0, 0]
    return vol_to_vec_by_elemcoords(vol_f, elem_coords)


@torch.no_grad()
def oc_update(x: torch.Tensor, dJdx: torch.Tensor, volfrac: float, move: float = 0.1, xmin: float = 1e-3) -> torch.Tensor:
    """
    OC for maximization:
      x_new = clip( x * sqrt(dJdx / lambda) )
    with bisection on lambda to satisfy mean(x_new)=volfrac.
    """
    dJdx = torch.clamp(dJdx, min=1e-12)
    l1, l2 = 1e-12, 1e6
    for _ in range(60):
        lam = 0.5 * (l1 + l2)
        x_new = x * torch.sqrt(dJdx / lam)
        x_new = torch.clamp(x_new, x - move, x + move)
        x_new = torch.clamp(x_new, xmin, 1.0)
        if x_new.mean() > volfrac:
            l1 = lam
        else:
            l2 = lam
    return x_new


@torch.no_grad()
def make_random_x_init(
    res: int,
    volfrac: float,
    xmin: float,
    device: torch.device,
    iters: int = 5,
) -> torch.Tensor:
    """Random continuous x in [xmin,1] with mean(x)=volfrac. Returns (res^3,1)."""
    N = res ** 3
    x = torch.rand((N, 1), device=device, dtype=torch.float32)
    for _ in range(int(iters)):
        x = x - x.mean() + float(volfrac)
        x = torch.clamp(x, xmin, 1.0)
    x_vol = x.reshape(res, res, res)
    _enforce_periodic_vol_(x_vol)
    x = x_vol.reshape(N, 1)
    x = x - x.mean() + float(volfrac)
    x = torch.clamp(x, xmin, 1.0)
    return x



@torch.no_grad()
def evaluate_J(ihd_module: IHD, batch: Dict[str, torch.Tensor], rho: torch.Tensor, obj: str) -> float:
    """Evaluate (possibly multi-objective) J on a fixed rho."""
    batch["rho"] = rho
    n_nodes = batch["feature"].shape[0]

    need_CH, need_kappa = objective_requires(obj)

    CH = None
    kappa = None

    if need_kappa:
        # --- thermal conduction homogenization (3 load cases -> 3x3 kappa) ---
        F_rhs_t = assemble_F_fast_scalar(batch["node_index"], ihd_module.Fe_1d, n_nodes, rho)
        kappa = ihd_module.TO_homo_1d(batch, ihd_module.Ke_1d, F_rhs_t, ihd_module.X0_1d, batch_size=1)

    if need_CH:
        # --- mechanical homogenization (6 load cases -> 6x6 CH) ---
        F_rhs_m = assemble_F_fast(batch["node_index"], ihd_module.Fe_3d, n_nodes, rho)
        CH = ihd_module.TO_homo_3d(batch, ihd_module.Ke_3d, F_rhs_m, ihd_module.X0_3d, batch_size=1)

    J = objective_from_spec(obj=obj, CH=CH, kappa=kappa, normalize_by=None, return_parts=False)
    return float(J.item())


@torch.no_grad()
def evaluate_homogenized_terms(
    ihd_module: IHD,
    batch: Dict[str, torch.Tensor],
    rho: torch.Tensor,
    *,
    terms: Tuple[str, ...] = ("stiffness", "shear", "thermal"),
) -> Dict[str, float]:
    """Evaluate selected homogenized scalar metrics on a fixed rho field."""
    wanted = {str(t).lower() for t in terms}
    need_ch = any(t in wanted for t in ("stiffness", "shear"))
    need_kappa = "thermal" in wanted

    batch["rho"] = rho
    n_nodes = batch["feature"].shape[0]

    CH = None
    kappa = None

    if need_kappa:
        F_rhs_t = assemble_F_fast_scalar(batch["node_index"], ihd_module.Fe_1d, n_nodes, rho)
        kappa = ihd_module.TO_homo_1d(batch, ihd_module.Ke_1d, F_rhs_t, ihd_module.X0_1d, batch_size=1)

    if need_ch:
        F_rhs_m = assemble_F_fast(batch["node_index"], ihd_module.Fe_3d, n_nodes, rho)
        CH = ihd_module.TO_homo_3d(batch, ihd_module.Ke_3d, F_rhs_m, ihd_module.X0_3d, batch_size=1)

    out: Dict[str, float] = {}
    if CH is not None:
        if "stiffness" in wanted:
            out["stiffness"] = float(stiffness_objective(CH).item())
        if "shear" in wanted:
            out["shear"] = float(shear_objective(CH).item())
    if kappa is not None and "thermal" in wanted:
        out["thermal"] = float(thermal_objective(kappa).item())
    return out


# -----------------------------
# Optimization
# -----------------------------
@dataclass
class RunResult:
    J_raw_init: float
    J_raw_final: float
    J_total_init: float
    J_total_final: float
    P_print_init: float
    P_print_final: float
    P_cavity_init: float
    P_cavity_final: float
    iters_run: int
    converged: bool
    rho_final: np.ndarray
    log_csv: Optional[str]
    summary_json: Optional[str]
    meshes_dir: str
    rho_final_npy: str
    final_obj: str


def optimize_inverse_homogenization(
    ihd_module: IHD,
    batch: Dict[str, torch.Tensor],
    *,
    res: int,
    objective: str,
    init_mode: str,
    volfrac: float,
    penal: float,
    rho_min: float,
    move: float,
    xmin: float,
    iters: int,
    seed: int,
    symmetry: str,
    lambda_print: float,
    overhang_deg: float,
    lambda_cavity: float,
    cavity_thresh: float,
    cavity_beta: float,
    min_channel_rad: int,
    cavity_rad: int,
    cavity_seed_faces: str,
    cavity_iters: int,
    cavity_tiles: int,
    cavity_tiles_xyz: str | None,
    cavity_focus_center: bool,
    filter_k: int,
    output_dir: str,
    mesh_every: int,
    mesh_res: int,
    mesh_iso: float,
    mesh_pad: int,
    save_iter_log: bool = True,
    save_summary: bool = True,
    save_initial_mesh: bool = True,
    verbose: bool = True,
) -> RunResult:
    device = batch["rho"].device
    setup_seed(seed)

    os.makedirs(output_dir, exist_ok=True)
    meshes_dir = os.path.join(output_dir, "meshes")
    os.makedirs(meshes_dir, exist_ok=True)
    log_csv = os.path.join(output_dir, "iter_log.csv") if save_iter_log else None
    summary_json = os.path.join(output_dir, "summary.json") if save_summary else None
    final_rho_npy = os.path.join(output_dir, "rho_final.npy")
    final_obj = os.path.join(meshes_dir, "final.obj")

    elem_coords = batch["elem_coords"]
    sym_kind, sym_axes = parse_symmetry(symmetry)
    use_print = float(lambda_print) > 0.0
    use_cavity = float(lambda_cavity) > 0.0

    # --- init x ---
    if init_mode == "noise":
        x = make_random_x_init(res=res, volfrac=volfrac, xmin=xmin, device=device, iters=5)
    else:
        rho0 = batch["rho"].float().reshape(-1, 1).clamp(rho_min, 1.0)
        x = ((rho0 - rho_min) / (1.0 - rho_min)).clamp(0.0, 1.0).pow(1.0 / penal)
        x = x.clamp(xmin, 1.0)
        if volfrac <= 0:
            volfrac = float(x.mean().item())
    if sym_kind != "none":
        with torch.no_grad():
            x = symmetrize_vec(x, elem_coords, res=res, kind=sym_kind, axes=sym_axes).clamp(xmin, 1.0)

    rho_init = (rho_min + (1.0 - rho_min) * (x ** penal)).reshape(-1, 1).clamp(rho_min, 1.0)
    J_raw_init = evaluate_J(ihd_module, batch, rho_init, obj=objective)
    if use_print:
        P_print_init_t = printability_penalty_overhang(
            rho_init.reshape(res, res, res), overhang_deg=float(overhang_deg), periodic_xy=True
        )
        P_print_init = float(P_print_init_t.item())
    else:
        P_print_init = 0.0
    if use_cavity:
        rho_grid_init = rho_init.reshape(res, res, res)
        tx, ty, tz = _parse_tiles_xyz(cavity_tiles_xyz, cavity_tiles)
        if max(tx, ty, tz) > 1:
            P_cavity_init_t = cavity_penalty_periodic_array(
                rho_grid_init,
                tiles_xyz=(tx, ty, tz),
                focus_center=bool(cavity_focus_center),
                void_thresh=float(cavity_thresh),
                beta=float(cavity_beta),
                min_channel_rad=int(min_channel_rad),
                rad=int(cavity_rad),
                iters=int(cavity_iters),
                seed_faces=str(cavity_seed_faces),
            )
        else:
                P_cavity_init_t = cavity_penalty_closed_void(
                rho_grid_init,
                void_thresh=float(cavity_thresh),
                beta=float(cavity_beta),
                min_channel_rad=int(min_channel_rad),
                    rad=int(cavity_rad),
                    iters=int(cavity_iters),
                periodic_xyz=True,
                seed_faces=str(cavity_seed_faces),
            )
        P_cavity_init = float(P_cavity_init_t.item())
    else:
        P_cavity_init = 0.0
    J_total_init = float(J_raw_init) - float(lambda_print) * float(P_print_init) - float(lambda_cavity) * float(P_cavity_init)

    # write header
    if save_iter_log and log_csv is not None:
        with open(log_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["iter", "J_total", "J_raw", "P_print", "P_cavity", "mean_x", "mean_rho", "dx_max", "dx_mean", "time_sec"])

    # initial mesh
    rho_grid0 = rho_init.reshape(res, res, res).detach().cpu().numpy().astype(np.float32)
    if save_initial_mesh:
        save_mesh_from_rho(rho_grid0, os.path.join(meshes_dir, "it0000.obj"), out_res=mesh_res, iso=mesh_iso, pad=mesh_pad)

    t0 = time.time()
    J_prev: Optional[float] = None
    stall = 0
    converged = False
    it_run = 0
    tol_x = 1e-3
    tol_J = 1e-4
    patience = 10
    min_iters = 10

    for it in range(iters):
        it_run = it + 1
        x_old = x.detach().reshape(-1, 1)
        x_var = x_old.detach().requires_grad_(True)

        x_c = x_var.clamp(xmin, 1.0)
        x_used = symmetrize_vec(x_c, elem_coords, res=res, kind=sym_kind, axes=sym_axes).clamp(xmin, 1.0)

        rho = (rho_min + (1.0 - rho_min) * (x_used ** penal)).reshape(-1, 1)
        batch["rho"] = rho

        n_nodes = batch["feature"].shape[0]


        need_CH, need_kappa = objective_requires(objective)

        CH = None
        kappa = None

        if need_kappa:
            F_rhs_t = assemble_F_fast_scalar(batch["node_index"], ihd_module.Fe_1d, n_nodes, rho)
            kappa = ihd_module.TO_homo_1d(batch, ihd_module.Ke_1d, F_rhs_t, ihd_module.X0_1d, batch_size=1)

        if need_CH:
            F_rhs_m = assemble_F_fast(batch["node_index"], ihd_module.Fe_3d, n_nodes, rho)
            CH = ihd_module.TO_homo_3d(batch, ihd_module.Ke_3d, F_rhs_m, ihd_module.X0_3d, batch_size=1)

        J_raw = objective_from_spec(obj=objective, CH=CH, kappa=kappa, normalize_by=None, return_parts=False)

        if use_print:
            rho_grid = rho.reshape(res, res, res)
            P_print = printability_penalty_overhang(
                rho_grid, overhang_deg=float(overhang_deg), periodic_xy=True
            )
        else:
            P_print = rho.new_zeros(())

        if use_cavity:
            rho_grid = rho.reshape(res, res, res)
            tx, ty, tz = _parse_tiles_xyz(cavity_tiles_xyz, cavity_tiles)
            if max(tx, ty, tz) > 1:
                P_cavity = cavity_penalty_periodic_array(
                    rho_grid,
                    tiles_xyz=(tx, ty, tz),
                    focus_center=bool(cavity_focus_center),
                    void_thresh=float(cavity_thresh),
                    beta=float(cavity_beta),
                min_channel_rad=int(min_channel_rad),
                    rad=int(cavity_rad),
                    iters=int(cavity_iters),
                    seed_faces=str(cavity_seed_faces),
                )
            else:
                P_cavity = cavity_penalty_closed_void(
                    rho_grid,
                    void_thresh=float(cavity_thresh),
                    beta=float(cavity_beta),
                min_channel_rad=int(min_channel_rad),
                    rad=int(cavity_rad),
                    iters=int(cavity_iters),
                    periodic_xyz=True,
                    seed_faces=str(cavity_seed_faces),
                )
        else:
            P_cavity = rho.new_zeros(())

        J_total = J_raw - float(lambda_print) * P_print - float(lambda_cavity) * P_cavity

        (dJdx,) = torch.autograd.grad(J_total, x_var, retain_graph=False, create_graph=False)

        if filter_k and int(filter_k) > 1:
            dJdx = avg_filter3d_x_periodic(dJdx, elem_coords, res=res, k=int(filter_k))

        x_new = oc_update(x_old, dJdx.detach(), volfrac=float(volfrac), move=float(move), xmin=float(xmin))

        with torch.no_grad():
            x_next = symmetrize_vec(x_new, elem_coords, res=res, kind=sym_kind, axes=sym_axes).clamp(xmin, 1.0)

            dx = (x_next - x_old).abs()
            dx_max = float(dx.max().item())
            dx_mean = float(dx.mean().item())

            J_total_val = float(J_total.item())
            J_raw_val = float(J_raw.item())
            P_print_val = float(P_print.item()) if use_print else 0.0
            P_cavity_val = float(P_cavity.item()) if use_cavity else 0.0
            if J_prev is None:
                dJ_rel = float("inf")
            else:
                dJ_rel = abs(J_total_val - J_prev) / max(abs(J_prev), 1e-12)

            if (it >= min_iters) and (dx_max < tol_x) and (dJ_rel < tol_J):
                stall += 1
            else:
                stall = 0

            x = x_next
            J_prev = J_total_val

        rho_mean = float(rho.mean().item())
        x_mean = float(x_used.mean().item())
        t_now = time.time() - t0

        if save_iter_log and log_csv is not None:
            with open(log_csv, "a", newline="") as f:
                w = csv.writer(f)
                w.writerow([it_run, J_prev, J_raw_val, P_print_val, P_cavity_val, x_mean, rho_mean, dx_max, dx_mean, t_now])

        if verbose:
            print(
                f"it={it_run:04d}  J_total={J_prev:.6e}  J_raw={J_raw_val:.6e}  P_print={P_print_val:.6e}  P_cavity={P_cavity_val:.6e}  "
                f"mean_x={x_mean:.4f}  mean_rho={rho_mean:.4f}  "
                f"dx_max={dx_max:.3e}  dx_mean={dx_mean:.3e}  stall={stall}/{patience}"
            )

        # export mesh
        if (mesh_every > 0) and (it_run % mesh_every == 0):
            with torch.no_grad():
                x_vis = symmetrize_vec(x.reshape(-1, 1), elem_coords, res=res, kind=sym_kind, axes=sym_axes).clamp(xmin, 1.0)
                rho_vis = (rho_min + (1.0 - rho_min) * (x_vis ** penal)).reshape(-1, 1)

                if filter_k and int(filter_k) > 1:
                    rho_vis = avg_filter3d_x_periodic(rho_vis, elem_coords, res=res, k=int(filter_k))

                rho_vis_grid = rho_vis.reshape(res, res, res).detach().cpu().numpy().astype(np.float32)

            save_mesh_from_rho(
                rho_vis_grid,
                os.path.join(meshes_dir, f"it{it_run:04d}.obj"),
                out_res=mesh_res,
                iso=mesh_iso,
                pad=mesh_pad,
            )

        if stall >= patience:
            converged = True
            if verbose:
                print(f"[EARLY STOP] Converged at it={it_run} for {patience} consecutive checks.")
            break

    # final
    with torch.no_grad():
        x = symmetrize_vec(x, elem_coords, res=res, kind=sym_kind, axes=sym_axes).clamp(xmin, 1.0)
        rho_final = (rho_min + (1.0 - rho_min) * (x ** penal)).reshape(-1, 1).clamp(rho_min, 1.0)
        J_raw_final = evaluate_J(ihd_module, batch, rho_final, obj=objective)
        if use_print:
            P_print_final_t = printability_penalty_overhang(
                rho_final.reshape(res, res, res), overhang_deg=float(overhang_deg), periodic_xy=True
            )
            P_print_final = float(P_print_final_t.item())
        else:
            P_print_final = 0.0

        if use_cavity:
            rho_grid_final = rho_final.reshape(res, res, res)
            tx, ty, tz = _parse_tiles_xyz(cavity_tiles_xyz, cavity_tiles)
            if max(tx, ty, tz) > 1:
                P_cavity_final_t = cavity_penalty_periodic_array(
                    rho_grid_final,
                    tiles_xyz=(tx, ty, tz),
                    focus_center=bool(cavity_focus_center),
                    void_thresh=float(cavity_thresh),
                    beta=float(cavity_beta),
                min_channel_rad=int(min_channel_rad),
                    rad=int(cavity_rad),
                    iters=int(cavity_iters),
                    seed_faces=str(cavity_seed_faces),
                )
            else:
                P_cavity_final_t = cavity_penalty_closed_void(
                    rho_grid_final,
                    void_thresh=float(cavity_thresh),
                    beta=float(cavity_beta),
                min_channel_rad=int(min_channel_rad),
                    rad=int(cavity_rad),
                    iters=int(cavity_iters),
                    periodic_xyz=True,
                    seed_faces=str(cavity_seed_faces),
                )
            P_cavity_final = float(P_cavity_final_t.item())
        else:
            P_cavity_final = 0.0

        J_total_final = (
            float(J_raw_final)
            - float(lambda_print) * float(P_print_final)
            - float(lambda_cavity) * float(P_cavity_final)
        )
        rho_final_grid = rho_final.reshape(res, res, res).detach().cpu().numpy().astype(np.float32)

    np.save(final_rho_npy, rho_final_grid)

    # final mesh
    rho_final_mesh = rho_final
    if filter_k and int(filter_k) > 1:
        rho_final_mesh = avg_filter3d_x_periodic(rho_final_mesh, elem_coords, res=res, k=int(filter_k))
    rho_final_mesh_grid = rho_final_mesh.reshape(res, res, res).detach().cpu().numpy().astype(np.float32)

    save_mesh_from_rho(
        rho_final_mesh_grid,
        final_obj,
        out_res=mesh_res,
        iso=mesh_iso,
        pad=mesh_pad,
    )

    # summary
    summary = {
        "objective": objective,
        "init_mode": init_mode,
        "symmetry": str(symmetry),
        "lambda_print": float(lambda_print),
        "overhang_deg": float(overhang_deg),
        "lambda_cavity": float(lambda_cavity),
        "cavity_thresh": float(cavity_thresh),
        "cavity_beta": float(cavity_beta),
        "min_channel_rad": int(min_channel_rad),
        "cavity_rad": int(cavity_rad),
        "cavity_seed_faces": str(cavity_seed_faces),
        "cavity_iters": int(cavity_iters),
        "cavity_tiles": int(cavity_tiles),
        "cavity_tiles_xyz": (None if cavity_tiles_xyz is None else str(cavity_tiles_xyz)),
        "cavity_focus_center": bool(cavity_focus_center),
        "resolution": res,
        "iters_requested": iters,
        "iters_run": int(it_run),
        "converged": bool(converged),
        "seed": int(seed),
        "volfrac_target": float(volfrac),
        "penal": float(penal),
        "rho_min": float(rho_min),
        "move": float(move),
        "xmin": float(xmin),
        "mesh_every": int(mesh_every),
        "mesh_res": int(mesh_res),
        "mesh_iso": float(mesh_iso),
        "mesh_pad": int(mesh_pad),
        "J_raw_init": float(J_raw_init),
        "J_raw_final": float(J_raw_final),
        "J_total_init": float(J_total_init),
        "J_total_final": float(J_total_final),
        "P_print_init": float(P_print_init),
        "P_print_final": float(P_print_final),
        "P_cavity_init": float(P_cavity_init),
        "P_cavity_final": float(P_cavity_final),
        "log_csv": (None if log_csv is None else os.path.abspath(log_csv)),
        "rho_final_npy": os.path.abspath(final_rho_npy),
        "meshes_dir": os.path.abspath(meshes_dir),
        "time_sec": float(time.time() - t0),
    }
    if save_summary and summary_json is not None:
        with open(summary_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    return RunResult(
        J_raw_init=float(J_raw_init),
        J_raw_final=float(J_raw_final),
        J_total_init=float(J_total_init),
        J_total_final=float(J_total_final),
        P_print_init=float(P_print_init),
        P_print_final=float(P_print_final),
        P_cavity_init=float(P_cavity_init),
        P_cavity_final=float(P_cavity_final),
        iters_run=int(it_run),
        converged=bool(converged),
        rho_final=rho_final_grid,
        log_csv=log_csv,
        summary_json=summary_json,
        meshes_dir=meshes_dir,
        rho_final_npy=final_rho_npy,
        final_obj=final_obj,
    )


# -----------------------------
# CLI / entrypoint
# -----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Inverse homogenization TO (single run, CLI-only).")

    # init
    p.add_argument("--init", type=str, default="voxel", choices=["voxel", "noise"], help="Initialization mode.")
    p.add_argument("--voxel_npy", type=str, default=None, help="Voxel .npy path (0/1), for --init voxel.")
    p.add_argument("--seed", type=int, default=0, help="Random seed (for noise init).")

    # problem size
    p.add_argument("--resolution", type=int, default=64, help="Voxel resolution r (r^3 elements).")
    # optimization params
    p.add_argument("--objective", type=str, default="shear", help=("Objective spec. Single: shear|stiffness|thermal. Multi: e.g. 'stiffness:0.4,thermal:0.6' or '0.4*stiffness+0.6*thermal'.") )
    p.add_argument("--iters", type=int, default=400, help="Number of TO iterations.")
    p.add_argument("--volfrac", type=float, default=0.0,
                   help="Target volume fraction (mean(x)). If 0 and init=voxel, uses mean(init).")
    p.add_argument("--penal", type=float, default=3.0, help="SIMP penalty p.")
    p.add_argument("--rho_min", type=float, default=1.0e-5, help="Minimum physical density in SIMP.")
    p.add_argument("--move", type=float, default=0.05, help="OC move limit per iteration.")
    p.add_argument("--xmin", type=float, default=1.0e-5, help="Minimum design variable x.")
    p.add_argument("--filter_k", type=int, default=3, help="Periodic box filter size (odd). Set <=1 to disable.")
    p.add_argument(
        "--symmetry",
        type=str,
        default="none",
        choices=["none", "x", "y", "z", "xy", "xz", "yz", "xyz", "octant", "cubic", "iso"],
        help=("Symmetry projection. Mirror modes: x/y/z/xy/xz/yz/xyz (xyz==octant==1/8). Cubic modes: cubic/iso enforce full 48 cube symmetries (x=y=z directions)."),
    )

    # additive manufacturing / printability (45° overhang)
    p.add_argument(
        "--lambda_print",
        type=float,
        default=0.0,
        help="Weight for 45° printability penalty (0 disables).",
    )
    p.add_argument(
        "--overhang_deg",
        type=float,
        default=45.0,
        help="Overhang angle constraint in degrees (currently supports 45° best).",
    )

    # closed-cavity (trapped void) penalty
    p.add_argument(
        "--lambda_cavity",
        type=float,
        default=0.0,
        help="Weight for closed-cavity (trapped void) penalty (0 disables).",
    )
    p.add_argument("--cavity_thresh", type=float, default=0.5, help="Void threshold (rho < thresh) used for cavity detection.")
    p.add_argument("--cavity_beta", type=float, default=20.0, help="Sigmoid steepness for soft void indicator.")
    p.add_argument("--min_channel_rad", type=int, default=0, help="Minimum void-channel radius (in voxels) required for a void path to count as connected to outside (0 disables). Roughly enforces diameter ~ (2*rad+1) voxels.")
    p.add_argument("--cavity_rad", type=int, default=1, help="Connectivity radius in voxels (1 => 3x3x3 neighborhood).")
    p.add_argument(
        "--cavity_seed_faces",
        type=str,
        default="all",
        choices=["all", "x", "y", "z", "xy"],
        help="Which boundary faces count as 'outside' for cavity connectivity. 'xy' means +/-z faces only.",
    )
    p.add_argument(
        "--cavity_iters",
        type=int,
        default=128,
        help="Flood-fill iterations (0 => use resolution steps).",
    )

    p.add_argument("--cavity_tiles", type=int, default=1,
                   help="Tile count per axis for *array* cavity detection (1 => single-cell). Try 3 for interior check.")
    p.add_argument("--cavity_tiles_xyz", type=str, default=None,
                   help="Override tiling as 'tx,ty,tz' (e.g., 3,3,3). If set, overrides --cavity_tiles.")
    p.add_argument("--cavity_focus_center", dest="cavity_focus_center", action="store_true",
                   help="When using tiling (>=3), compute cavity penalty only on the center unit cell (recommended).")
    p.add_argument("--no-cavity_focus_center", dest="cavity_focus_center", action="store_false",
                   help="Disable center-only focus; average over whole supercell.")
    p.set_defaults(cavity_focus_center=True)

    # physical meaning for voxel->rho
    p.add_argument("--rho_void", type=float, default=1.0e-5, help="Density for void voxels in voxel init.")
    p.add_argument("--rho_solid", type=float, default=0.9999, help="Density for solid voxels in voxel init.")

    # GMG params needed by IHD.TO_homo
    p.add_argument("--gmg_smooth_iter", type=int, nargs="+", default=[2, 2, 3, 4, 5],
                   help="GMG smooth iterations per level, e.g. 2 2 3 4 5.")
    p.add_argument("--gmg_max_cycle", type=int, default=1, help="GMG max cycles.")

    # output
    p.add_argument("--output_dir", type=str, default="./ihd_out", help="Output directory.")
    p.add_argument("--mesh_every", type=int, default=10, help="Export mesh every N iterations.")
    p.add_argument("--mesh_res", type=int, default=128, help="Upsampled mesh resolution.")
    p.add_argument("--mesh_iso", type=float, default=0.3, help="Marching cubes iso threshold.")
    p.add_argument("--mesh_pad", type=int, default=2, help="Padding for marching cubes volume.")
    p.add_argument("--quiet", action="store_true", help="Reduce stdout logging.")

    return p.parse_args()


def _load_init_voxel(args: argparse.Namespace) -> np.ndarray:
    """Load voxel initialization from a .npy.

    Supported shapes:
      - (n,n,n)
      - (n^3,) flat vector

    If n != resolution, we nearest-resample to (resolution,resolution,resolution).
    """
    if not args.voxel_npy:
        raise ValueError("--init voxel requires --voxel_npy <path_to.npy>. (Single-voxel init removed.)")

    r = int(args.resolution)
    arr = np.load(args.voxel_npy)
    arr = np.asarray(arr)
    if arr.ndim == 1:
        n = int(round(arr.size ** (1.0 / 3)))
        if n ** 3 != arr.size:
            raise ValueError(f"voxel npy is 1D but size={arr.size} is not a perfect cube")
        arr = arr.reshape(n, n, n)
    elif arr.ndim != 3:
        raise ValueError(f"voxel npy must be 3D (n,n,n) or 1D (n^3,), got shape={arr.shape}")

    voxel_bool = (arr > 0)
    if voxel_bool.shape != (r, r, r):
        voxel_bool = _resample_voxel_nearest(voxel_bool, out_res=r)
    return voxel_bool.astype(np.uint8)


def main() -> None:
    args = parse_args()
    verbose = not args.quiet

    setup_seed(int(args.seed))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # build minimal cfg for IHD (no OmegaConf)
    cfg = SimpleNamespace(
        resolution=int(args.resolution),
        GMG=SimpleNamespace(
            smooth_iter=list(args.gmg_smooth_iter),
            max_cycle=int(args.gmg_max_cycle),
        ),
    )

    ihd_module = IHD(cfg, device).to(device)

    # init rho_grid
    r = int(args.resolution)
    if args.init == "noise":
        vf = float(args.volfrac)
        rho_grid = torch.rand((r, r, r), device=device, dtype=torch.float32)
    else:
        voxel = _load_init_voxel(args)
        rho_grid = torch.full((r, r, r), float(args.rho_void), device=device, dtype=torch.float32)
        rho_grid[torch.from_numpy(voxel.astype(bool)).to(device)] = float(args.rho_solid)

    # The unit-cell solver is periodic by construction.
    _enforce_periodic_vol_(rho_grid)

    batch = build_unitcell_batch(device=device, res=r, rho_grid=rho_grid)

    init_mode = args.init
    volfrac = float(args.volfrac)
    if init_mode == "noise" and volfrac <= 0:
        raise ValueError("--init noise requires --volfrac > 0 (target mean(x)).")

    result = optimize_inverse_homogenization(
        ihd_module,
        batch,
        res=r,
        objective   =str(args.objective),
        init_mode   =str(init_mode),
        volfrac     =float(volfrac),
        penal       =float(args.penal),
        rho_min     =float(args.rho_min),
        move        =float(args.move),
        xmin        =float(args.xmin),
        iters       =int(args.iters),
        seed        =int(args.seed),
        symmetry    =str(args.symmetry),
        lambda_print=float(args.lambda_print),
        overhang_deg=float(args.overhang_deg),
        lambda_cavity=float(args.lambda_cavity),
        cavity_thresh=float(args.cavity_thresh),
        cavity_beta=float(args.cavity_beta),
        min_channel_rad=int(args.min_channel_rad),
        cavity_rad=int(args.cavity_rad),
        cavity_seed_faces=str(args.cavity_seed_faces),
        cavity_iters=int(args.cavity_iters),
        cavity_tiles=int(args.cavity_tiles),
        cavity_tiles_xyz=(None if args.cavity_tiles_xyz is None else str(args.cavity_tiles_xyz)),
        cavity_focus_center=bool(args.cavity_focus_center),
        filter_k    =int(args.filter_k),
        output_dir  =str(args.output_dir),
        mesh_every  =int(args.mesh_every),
        mesh_res    =int(args.mesh_res),
        mesh_iso    =float(args.mesh_iso),
        mesh_pad    =int(args.mesh_pad),
        verbose     =verbose,
    )

    if verbose:
        print("\n[DONE]")
        print(f"J_total_init  = {result.J_total_init:.6e}")
        print(f"J_total_final = {result.J_total_final:.6e}")
        print(f"J_raw_init    = {result.J_raw_init:.6e}")
        print(f"J_raw_final   = {result.J_raw_final:.6e}")
        if float(args.lambda_print) > 0:
            print(f"P_print_init  = {result.P_print_init:.6e}")
            print(f"P_print_final = {result.P_print_final:.6e}")
        if float(args.lambda_cavity) > 0:
            print(f"P_cavity_init  = {result.P_cavity_init:.6e}")
            print(f"P_cavity_final = {result.P_cavity_final:.6e}")
        print(f"iters   = {result.iters_run} (converged={result.converged})")
        if result.log_csv is not None:
            print("log:", os.path.abspath(result.log_csv))
        if result.summary_json is not None:
            print("summary:", os.path.abspath(result.summary_json))
        print("meshes:", os.path.abspath(result.meshes_dir))


if __name__ == "__main__":
    main()

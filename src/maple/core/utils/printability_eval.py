from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from skimage import measure, morphology


@dataclass(frozen=True)
class OverhangMetrics:
    penalty: float
    score: float
    unsupported_voxels: int
    considered_solid_voxels: int
    support_radius: int


@dataclass(frozen=True)
class ChannelMetrics:
    penalty: float
    score: float
    trapped_void_voxels: int
    void_voxels: int
    boundary_reachable_void_voxels: int
    has_cavity: bool
    sealed_boundary: bool


@dataclass(frozen=True)
class ConnectivityMetrics:
    penalty: float
    score: float
    component_count: int
    is_single_component: bool


def _validate_cubic_binary_grid(grid: np.ndarray) -> np.ndarray:
    arr = np.asarray(grid)
    if arr.ndim != 3:
        raise ValueError(f"expected a 3D array, got shape {tuple(arr.shape)}")
    if arr.shape[0] != arr.shape[1] or arr.shape[0] != arr.shape[2]:
        raise ValueError(f"expected a cubic grid, got shape {tuple(arr.shape)}")
    return arr.astype(bool, copy=False)


def binarize_density_grid(rho_grid: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    rho = np.asarray(rho_grid, dtype=np.float32)
    if rho.ndim != 3:
        raise ValueError(f"expected a 3D density grid, got shape {tuple(rho.shape)}")
    return rho >= float(threshold)


def _support_radius_from_overhang(overhang_deg: float) -> int:
    od = float(np.clip(overhang_deg, 1.0, 89.0))
    rad = float(np.tan(np.deg2rad(90.0 - od)))
    return max(1, int(np.ceil(rad)))


def _dilate_xy_periodic(mask_xy: np.ndarray, rad: int) -> np.ndarray:
    out = np.zeros_like(mask_xy, dtype=bool)
    for dx in range(-rad, rad + 1):
        for dy in range(-rad, rad + 1):
            out |= np.roll(np.roll(mask_xy, shift=dx, axis=0), shift=dy, axis=1)
    return out


def _dilate_xy_nonperiodic(mask_xy: np.ndarray, rad: int) -> np.ndarray:
    h, w = mask_xy.shape
    out = np.zeros_like(mask_xy, dtype=bool)
    for dx in range(-rad, rad + 1):
        src_x0 = max(0, -dx)
        src_x1 = min(h, h - dx)
        dst_x0 = max(0, dx)
        dst_x1 = min(h, h + dx)
        for dy in range(-rad, rad + 1):
            src_y0 = max(0, -dy)
            src_y1 = min(w, w - dy)
            dst_y0 = max(0, dy)
            dst_y1 = min(w, w + dy)
            out[dst_x0:dst_x1, dst_y0:dst_y1] |= mask_xy[src_x0:src_x1, src_y0:src_y1]
    return out


def overhang_metrics_binary(
    solid_grid: np.ndarray,
    *,
    overhang_deg: float = 45.0,
    periodic_xy: bool = True,
) -> OverhangMetrics:
    solid = _validate_cubic_binary_grid(solid_grid)
    res = int(solid.shape[0])
    rad = _support_radius_from_overhang(overhang_deg)

    support = solid[:, :, 0].copy()
    unsupported_count = 0
    considered_count = 0

    for z in range(1, res):
        solid_z = solid[:, :, z]
        solid_count = int(solid_z.sum())
        considered_count += solid_count
        if solid_count == 0:
            support = solid_z
            continue

        if periodic_xy:
            supported_region = _dilate_xy_periodic(support, rad)
        else:
            supported_region = _dilate_xy_nonperiodic(support, rad)

        unsupported = solid_z & (~supported_region)
        supported = solid_z & supported_region

        unsupported_count += int(unsupported.sum())
        support = supported

    penalty = (float(unsupported_count) / float(considered_count)) if considered_count > 0 else 0.0
    penalty = float(np.clip(penalty, 0.0, 1.0))
    return OverhangMetrics(
        penalty=penalty,
        score=100.0 * (1.0 - penalty),
        unsupported_voxels=unsupported_count,
        considered_solid_voxels=considered_count,
        support_radius=rad,
    )


def _parse_tiles_xyz(tiles: int, tiles_xyz: Tuple[int, int, int] | None) -> Tuple[int, int, int]:
    if tiles_xyz is not None:
        tx, ty, tz = tiles_xyz
    else:
        tx = ty = tz = int(tiles)
    return max(1, int(tx)), max(1, int(ty)), max(1, int(tz))


def _center_slices(res: int, tx: int, ty: int, tz: int) -> Tuple[slice, slice, slice]:
    cx = tx // 2
    cy = ty // 2
    cz = tz // 2
    return (
        slice(cx * res, (cx + 1) * res),
        slice(cy * res, (cy + 1) * res),
        slice(cz * res, (cz + 1) * res),
    )


def _boundary_labels(labels: np.ndarray, seed_faces: str) -> np.ndarray:
    sf = str(seed_faces).lower()
    if sf not in {"all", "x", "y", "z", "xy"}:
        raise ValueError("seed_faces must be one of: all, x, y, z, xy")

    faces = []
    if sf in {"all", "x"}:
        faces.extend([labels[0, :, :], labels[-1, :, :]])
    if sf in {"all", "y"}:
        faces.extend([labels[:, 0, :], labels[:, -1, :]])
    if sf in {"all", "z", "xy"}:
        faces.extend([labels[:, :, 0], labels[:, :, -1]])

    if not faces:
        return np.empty((0,), dtype=labels.dtype)
    boundary = np.unique(np.concatenate([face.reshape(-1) for face in faces]))
    return boundary[boundary > 0]


def _boundary_mask(mask: np.ndarray, seed_faces: str) -> np.ndarray:
    sf = str(seed_faces).lower()
    if sf not in {"all", "x", "y", "z", "xy"}:
        raise ValueError("seed_faces must be one of: all, x, y, z, xy")

    out = np.zeros_like(mask, dtype=bool)
    if sf in {"all", "x"}:
        out[0, :, :] = True
        out[-1, :, :] = True
    if sf in {"all", "y"}:
        out[:, 0, :] = True
        out[:, -1, :] = True
    if sf in {"all", "z", "xy"}:
        out[:, :, 0] = True
        out[:, :, -1] = True
    return out


def _disk_offsets(rad: int) -> list[tuple[int, int]]:
    r = int(rad)
    return [
        (du, dv)
        for du in range(-r, r + 1)
        for dv in range(-r, r + 1)
        if du * du + dv * dv <= r * r
    ]


def _erode2d_nonperiodic(mask: np.ndarray, rad: int) -> np.ndarray:
    if int(rad) <= 0:
        return np.asarray(mask, dtype=bool)

    src = np.asarray(mask, dtype=bool)
    h, w = src.shape
    out = np.ones_like(src, dtype=bool)
    for du, dv in _disk_offsets(int(rad)):
        shifted = np.zeros_like(src, dtype=bool)
        src_u0 = max(0, -du)
        src_u1 = min(h, h - du)
        dst_u0 = max(0, du)
        dst_u1 = min(h, h + du)
        src_v0 = max(0, -dv)
        src_v1 = min(w, w - dv)
        dst_v0 = max(0, dv)
        dst_v1 = min(w, w + dv)
        shifted[dst_u0:dst_u1, dst_v0:dst_v1] = src[src_u0:src_u1, src_v0:src_v1]
        out &= shifted
    return out


def _boundary_channel_seed(void_mask: np.ndarray, seed_faces: str, min_channel_rad: int) -> np.ndarray:
    """Boundary void locations whose face aperture can pass the requested radius."""

    sf = str(seed_faces).lower()
    if sf not in {"all", "x", "y", "z", "xy"}:
        raise ValueError("seed_faces must be one of: all, x, y, z, xy")

    rad = max(0, int(min_channel_rad))
    seed = np.zeros_like(void_mask, dtype=bool)

    if sf in {"all", "x"}:
        face = _erode2d_nonperiodic(void_mask[0, :, :], rad)
        seed[0, :, :] |= face
        face = _erode2d_nonperiodic(void_mask[-1, :, :], rad)
        seed[-1, :, :] |= face

    if sf in {"all", "y"}:
        face = _erode2d_nonperiodic(void_mask[:, 0, :], rad)
        seed[:, 0, :] |= face
        face = _erode2d_nonperiodic(void_mask[:, -1, :], rad)
        seed[:, -1, :] |= face

    if sf in {"all", "z", "xy"}:
        face = _erode2d_nonperiodic(void_mask[:, :, 0], rad)
        seed[:, :, 0] |= face
        face = _erode2d_nonperiodic(void_mask[:, :, -1], rad)
        seed[:, :, -1] |= face

    return seed


def channel_metrics_binary(
    solid_grid: np.ndarray,
    *,
    min_channel_rad: int = 3,
    tiles: int = 3,
    tiles_xyz: Tuple[int, int, int] | None = None,
    seed_faces: str = "all",
    connectivity: int = 1,
) -> ChannelMetrics:
    solid = _validate_cubic_binary_grid(solid_grid)
    res = int(solid.shape[0])
    tx, ty, tz = _parse_tiles_xyz(tiles, tiles_xyz)

    void_uc = ~solid
    void_super = np.tile(void_uc, (tx, ty, tz))
    if int(min_channel_rad) > 0:
        footprint = morphology.ball(int(min_channel_rad))
        void_conn = morphology.binary_opening(void_super, footprint=footprint)
    else:
        void_conn = void_super.copy()

    xs, ys, zs = _center_slices(res, tx, ty, tz)
    center_void_raw = void_super[xs, ys, zs]
    center_void_conn = void_conn[xs, ys, zs]

    void_voxels = int(center_void_raw.sum())

    # Hard guard: if all selected exterior seed faces are sealed by solid
    # material, any void inside the evaluated unit cell is trapped regardless of
    # labeling. This catches hollow-box cases whose outermost voxel shell is
    # fully solid.
    seed_boundary = _boundary_mask(void_super, seed_faces)
    sealed_boundary = bool(void_voxels > 0 and not np.any(void_super & seed_boundary))
    if sealed_boundary:
        return ChannelMetrics(
            penalty=1.0,
            score=0.0,
            trapped_void_voxels=void_voxels,
            void_voxels=void_voxels,
            boundary_reachable_void_voxels=0,
            has_cavity=(void_voxels > 0),
            sealed_boundary=True,
        )

    labels = measure.label(void_conn, connectivity=int(connectivity))
    if int(min_channel_rad) > 0:
        entrance_seed = _boundary_channel_seed(void_super, seed_faces, int(min_channel_rad))
        outside = np.unique(labels[entrance_seed & (labels > 0)])
    else:
        outside = _boundary_labels(labels, seed_faces)

    if void_voxels > 0 and outside.size == 0:
        return ChannelMetrics(
            penalty=1.0,
            score=0.0,
            trapped_void_voxels=void_voxels,
            void_voxels=void_voxels,
            boundary_reachable_void_voxels=0,
            has_cavity=(void_voxels > 0),
            sealed_boundary=False,
        )

    center_labels = labels[xs, ys, zs]
    reachable = center_void_conn & np.isin(center_labels, outside)
    trapped = center_void_raw & (~reachable)

    trapped_voxels = int(trapped.sum())
    reachable_voxels = int(reachable.sum())

    penalty = (float(trapped_voxels) / float(void_voxels)) if void_voxels > 0 else 0.0
    penalty = float(np.clip(penalty, 0.0, 1.0))

    return ChannelMetrics(
        penalty=penalty,
        score=100.0 * (1.0 - penalty),
        trapped_void_voxels=trapped_voxels,
        void_voxels=void_voxels,
        boundary_reachable_void_voxels=reachable_voxels,
        has_cavity=(trapped_voxels > 0),
        sealed_boundary=False,
    )


def connectivity_metrics_binary(
    solid_grid: np.ndarray,
    *,
    tiles: int = 3,
    tiles_xyz: Tuple[int, int, int] | None = None,
    connectivity: int = 1,
) -> ConnectivityMetrics:
    solid = _validate_cubic_binary_grid(solid_grid)
    res = int(solid.shape[0])
    tx, ty, tz = _parse_tiles_xyz(tiles, tiles_xyz)

    solid_super = np.tile(solid, (tx, ty, tz))
    labels = measure.label(solid_super, connectivity=int(connectivity))

    xs, ys, zs = _center_slices(res, tx, ty, tz)
    center_labels = np.unique(labels[xs, ys, zs])
    center_labels = center_labels[center_labels > 0]
    n_components = int(center_labels.size)

    if n_components == 0:
        penalty = 1.0
    elif n_components == 1:
        penalty = 0.0
    else:
        penalty = 1.0 - (1.0 / float(n_components))

    penalty = float(np.clip(penalty, 0.0, 1.0))
    return ConnectivityMetrics(
        penalty=penalty,
        score=100.0 * (1.0 - penalty),
        component_count=n_components,
        is_single_component=(n_components == 1),
    )


def printability_score_binary(
    rho_grid: np.ndarray,
    *,
    threshold: float = 0.5,
    overhang_deg: float = 45.0,
    periodic_xy: bool = True,
    min_channel_rad: int = 3,
    channel_tiles: int = 3,
    channel_tiles_xyz: Tuple[int, int, int] | None = None,
    connectivity_tiles: int = 3,
    connectivity_tiles_xyz: Tuple[int, int, int] | None = None,
    seed_faces: str = "all",
    label_connectivity: int = 1,
    lambda_oh: float = 1.0,
    lambda_chan: float = 1.0,
    lambda_conn: float = 1.0,
    oh_tol: float = 0.0,
    chan_tol: float = 0.0,
) -> Dict[str, float | int | bool]:
    solid = binarize_density_grid(rho_grid, threshold=threshold)
    overhang = overhang_metrics_binary(solid, overhang_deg=overhang_deg, periodic_xy=periodic_xy)
    channel = channel_metrics_binary(
        solid,
        min_channel_rad=min_channel_rad,
        tiles=channel_tiles,
        tiles_xyz=channel_tiles_xyz,
        seed_faces=seed_faces,
        connectivity=label_connectivity,
    )
    connectivity = connectivity_metrics_binary(
        solid,
        tiles=connectivity_tiles,
        tiles_xyz=connectivity_tiles_xyz,
        connectivity=label_connectivity,
    )

    lam_oh = float(lambda_oh)
    lam_chan = float(lambda_chan)
    lam_conn = float(lambda_conn)
    wsum = lam_oh + lam_chan + lam_conn
    if wsum <= 0.0:
        wsum = 1.0

    total_penalty = (
        lam_oh * overhang.penalty
        + lam_chan * channel.penalty
        + lam_conn * connectivity.penalty
    ) / wsum
    total_penalty = float(np.clip(total_penalty, 0.0, 1.0))
    total_score = 100.0 * (1.0 - total_penalty)

    overhang_ok = bool(overhang.penalty <= float(oh_tol))
    channel_ok = bool(channel.penalty <= float(chan_tol))
    connectivity_ok = bool(connectivity.is_single_component)
    printable_ok = bool(overhang_ok and channel_ok and connectivity_ok)

    return {
        "resolution": int(solid.shape[0]),
        "binary_threshold": float(threshold),
        "vf_binary": float(solid.mean()),
        "support_radius": int(overhang.support_radius),
        "unsupported_voxels": int(overhang.unsupported_voxels),
        "considered_solid_voxels": int(overhang.considered_solid_voxels),
        "P_oh": float(overhang.penalty),
        "S_oh": float(overhang.score),
        "trapped_void_voxels": int(channel.trapped_void_voxels),
        "void_voxels": int(channel.void_voxels),
        "boundary_reachable_void_voxels": int(channel.boundary_reachable_void_voxels),
        "has_cavity": bool(channel.has_cavity),
        "sealed_boundary": bool(channel.sealed_boundary),
        "P_chan": float(channel.penalty),
        "S_chan": float(channel.score),
        "solid_component_count": int(connectivity.component_count),
        "P_conn": float(connectivity.penalty),
        "S_conn": float(connectivity.score),
        "lambda_oh": lam_oh,
        "lambda_chan": lam_chan,
        "lambda_conn": lam_conn,
        "P_print": float(total_penalty),
        "S_print": float(total_score),
        "overhang_ok": overhang_ok,
        "channel_ok": channel_ok,
        "connectivity_ok": connectivity_ok,
        "printable_ok": printable_ok,
    }

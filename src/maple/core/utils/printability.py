import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple
# -----------------------------
# Soft morphology helpers (differentiable)
# -----------------------------
def _soft_erode3d(
    x: torch.Tensor,
    rad: int,
    *,
    periodic_xyz: bool,
    pad_val: float = 1.0,
) -> torch.Tensor:
    """Differentiable 3D erosion via min-pooling.

    x is assumed in [0,1]. For erosion we use:
        erode(x) = -max_pool(-x)

    Padding:
      - periodic_xyz=True  -> circular padding (wrap-around)
      - periodic_xyz=False -> constant padding with pad_val (default 1.0 so boundary is not artificially eroded)
    """
    rad_i = int(max(0, rad))
    if rad_i == 0:
        return x
    k = 2 * rad_i + 1
    pad = rad_i

    t = x[None, None]  # (1,1,D,H,W)
    if periodic_xyz:
        t = F.pad(t, (pad, pad, pad, pad, pad, pad), mode="circular")
    else:
        t = F.pad(t, (pad, pad, pad, pad, pad, pad), mode="constant", value=float(pad_val))
    er = -F.max_pool3d(-t, kernel_size=k, stride=1)[0, 0]
    return er


def _soft_dilate3d(
    x: torch.Tensor,
    rad: int,
    *,
    periodic_xyz: bool,
    pad_val: float = 0.0,
) -> torch.Tensor:
    """Differentiable 3D dilation via max-pooling."""
    rad_i = int(max(0, rad))
    if rad_i == 0:
        return x
    k = 2 * rad_i + 1
    pad = rad_i

    t = x[None, None]
    if periodic_xyz:
        t = F.pad(t, (pad, pad, pad, pad, pad, pad), mode="circular")
    else:
        t = F.pad(t, (pad, pad, pad, pad, pad, pad), mode="constant", value=float(pad_val))
    di = F.max_pool3d(t, kernel_size=k, stride=1)[0, 0]
    return di


def _soft_open3d(x: torch.Tensor, rad: int, *, periodic_xyz: bool) -> torch.Tensor:
    """Opening = erosion then dilation.

    Used here to remove narrow void connections while largely preserving the size of large void regions.
    """
    if int(max(0, rad)) == 0:
        return x
    er = _soft_erode3d(x, rad, periodic_xyz=periodic_xyz, pad_val=1.0)
    di = _soft_dilate3d(er, rad, periodic_xyz=periodic_xyz, pad_val=0.0)
    return di
# -----------------------------
# Additive manufacturing printability (overhang) penalty
# -----------------------------
def _support_radius_from_overhang(overhang_deg: float) -> int:
    """Compute lateral support radius (in voxels) per one-layer step for a target overhang angle.

    Convention here: overhang_deg is measured from the build plate (horizontal).
    - 45° => radius 1 (one voxel sideways per one voxel up)
    - smaller angles => stricter (larger radius)
    - larger angles => looser (radius tends to 1)
    """
    # avoid tan blowing up
    od = float(np.clip(overhang_deg, 1.0, 89.0))
    # complementary angle (from vertical)
    rad = float(np.tan(np.deg2rad(90.0 - od)))
    r = int(np.ceil(rad))
    return max(1, r)


def printability_penalty_overhang(
    rho_grid: torch.Tensor,
    *,
    overhang_deg: float = 45.0,
    periodic_xy: bool = True,
) -> torch.Tensor:
    """Differentiable overhang penalty on a density field.

    rho_grid: (r,r,r) in [0,1], build direction assumed +z (axis=2).

    A layer z is considered "supported" if there is material in layer z-1 within a lateral
    neighborhood defined by the overhang angle (45° -> 3x3 neighborhood).

    We propagate only the *supported* fraction upward:
      support(z) = min( rho(z), dilate_xy(support(z-1)) )
      penalty(z) = mean( relu(rho(z) - dilate_xy(support(z-1))) )
    """
    if rho_grid.ndim != 3 or rho_grid.shape[0] != rho_grid.shape[1] or rho_grid.shape[0] != rho_grid.shape[2]:
        raise ValueError(f"rho_grid must be cubic (r,r,r), got {tuple(rho_grid.shape)}")
    r = int(rho_grid.shape[0])
    rad = _support_radius_from_overhang(overhang_deg)
    k = 2 * rad + 1

    # bottom layer assumed fully supported by build plate
    support = rho_grid[:, :, 0]
    penalty = rho_grid.new_zeros(())

    for z in range(1, r):
        s = support[None, None]  # (1,1,r,r)
        if periodic_xy:
            s = F.pad(s, (rad, rad, rad, rad), mode="circular")
        else:
            s = F.pad(s, (rad, rad, rad, rad), mode="constant", value=0.0)
        s = F.max_pool2d(s, kernel_size=k, stride=1)
        s2d = s[0, 0]

        rho_z = rho_grid[:, :, z]
        unsupported = F.relu(rho_z - s2d)
        penalty = penalty + unsupported.mean()

        # only the supported part can support upper layers
        support = torch.minimum(rho_z, s2d)

    return penalty / max(r - 1, 1)

# -----------------------------
# Closed-cavity (trapped void) penalty
# -----------------------------
def cavity_penalty_closed_void(
    rho_grid: torch.Tensor,
    *,
    void_thresh: float = 0.5,
    beta: float = 20.0,
    min_channel_rad: int = 0,
    rad: int = 1,
    iters: int = 0,
    periodic_xyz: bool = True,
    seed_faces: str = "all",  # all | x | y | z | xy (z-only)
) -> torch.Tensor:
    """Penalty for *closed* (trapped) voids that are not connected to the unit-cell boundary.

    This is useful to discourage enclosed cavities (e.g., trapped powder in AM).

    Implementation: soft void indicator + masked flood-fill from boundary faces using max-pool dilation.

    Args:
        rho_grid: (r,r,r) density in [0,1]
        void_thresh: threshold for void (rho < void_thresh). Used in a sigmoid.
        beta: sigmoid steepness; larger -> sharper void/solid transition.
        rad: neighborhood radius for connectivity propagation (1 => 3x3x3).
        iters: number of flood-fill iterations; 0 => use r steps.
        periodic_xyz: if True, use circular padding (periodic connectivity).
        seed_faces: which boundary faces are treated as "outside" seeds:
            - "all": any of the 6 faces
            - "x"/"y"/"z": only +/- that axis faces
            - "xy": only +/-z faces (common for drainage along build direction)

    Returns:
        Scalar tensor ~ mean fraction of void not reachable from seeded boundary faces.
    """
    if rho_grid.ndim != 3 or rho_grid.shape[0] != rho_grid.shape[1] or rho_grid.shape[0] != rho_grid.shape[2]:
        raise ValueError(f"rho_grid must be cubic (r,r,r), got {tuple(rho_grid.shape)}")
    r = int(rho_grid.shape[0])
    if iters is None:
        iters = 0
    nsteps = int(iters) if int(iters) > 0 else r

    # soft void indicator: ~1 in void, ~0 in solid
    void = torch.sigmoid(float(beta) * (float(void_thresh) - rho_grid))

    # Optional minimum-channel constraint: remove narrow void connections for reachability.
    void_conn = void if int(max(0, min_channel_rad)) == 0 else _soft_open3d(void, int(min_channel_rad), periodic_xyz=bool(periodic_xyz))

    # seed mask on boundary faces
    seed = torch.zeros_like(void)
    sf = str(seed_faces).lower()
    if sf == "all":
        seed[0, :, :] = 1; seed[-1, :, :] = 1
        seed[:, 0, :] = 1; seed[:, -1, :] = 1
        seed[:, :, 0] = 1; seed[:, :, -1] = 1
    elif sf == "x":
        seed[0, :, :] = 1; seed[-1, :, :] = 1
    elif sf == "y":
        seed[:, 0, :] = 1; seed[:, -1, :] = 1
    elif sf == "z":
        seed[:, :, 0] = 1; seed[:, :, -1] = 1
    elif sf == "xy":
        # "xy" means seed only on +/-z faces (drainage along z)
        seed[:, :, 0] = 1; seed[:, :, -1] = 1
    else:
        raise ValueError("seed_faces must be one of: all, x, y, z, xy")

    reach = void_conn * seed  # reachable void starting from boundary

    rad_i = int(max(0, rad))
    k = 2 * rad_i + 1
    pad = rad_i

    if rad_i == 0:
        trapped = torch.relu(void - reach)
        return trapped.mean()

    for _ in range(nsteps):
        x = reach[None, None]  # (1,1,r,r,r)
        if periodic_xyz:
            x = F.pad(x, (pad, pad, pad, pad, pad, pad), mode="circular")
        else:
            x = F.pad(x, (pad, pad, pad, pad, pad, pad), mode="constant", value=0.0)
        dil = F.max_pool3d(x, kernel_size=k, stride=1)[0, 0]  # (r,r,r)
        reach = torch.minimum(void_conn, dil)

    trapped = torch.relu(void - reach)
    return trapped.mean()


# -----------------------------
# Periodic array (supercell) trapped-void penalty
# -----------------------------
def cavity_penalty_periodic_array(
    rho_grid: torch.Tensor,
    *,
    tiles: int = 3,
    tiles_xyz: Optional[Tuple[int, int, int]] = None,
    focus_center: bool = True,
    void_thresh: float = 0.5,
    beta: float = 20.0,
    min_channel_rad: int = 0,
    rad: int = 1,
    iters: int = 0,
    seed_faces: str = "all",
) -> torch.Tensor:
    """Detect trapped voids that may *only* appear when unit cells are assembled into a finite periodic array.

    Idea: tile the unit cell into a supercell (e.g., 3x3x3), treat the *outer* boundary of the supercell as "outside",
    then flood-fill void connected to that outside. Any void not connected is considered trapped.

    Why tiling helps:
      A void that touches the unit-cell boundary might become *internal* after tiling (e.g., corner/edge pockets),
      creating a trapped cavity in the assembled lattice even if a single cell looks cavity-free.

    Args:
        rho_grid: (r,r,r) unit-cell density in [0,1]
        tiles / tiles_xyz: tiling counts along x,y,z. If tiles_xyz is provided it overrides tiles.
        focus_center: if True and tiles >=3, compute penalty only over the center unit cell region.
            This reduces boundary artifacts and better reflects "interior" behavior of larger arrays.
        Other args mirror cavity_penalty_closed_void.

    Notes:
        - Flood fill is done with *non-periodic* padding on the supercell, because the supercell boundary is the only outside.
        - Cost grows quickly with tiles and resolution: O((tiles*r)^3). Start with tiles=3 and res<=64.
    """
    if rho_grid.ndim != 3 or rho_grid.shape[0] != rho_grid.shape[1] or rho_grid.shape[0] != rho_grid.shape[2]:
        raise ValueError(f"rho_grid must be cubic (r,r,r), got {tuple(rho_grid.shape)}")
    r = int(rho_grid.shape[0])

    if tiles_xyz is None:
        tx = ty = tz = int(tiles)
    else:
        tx, ty, tz = (int(tiles_xyz[0]), int(tiles_xyz[1]), int(tiles_xyz[2]))
    tx = max(1, tx); ty = max(1, ty); tz = max(1, tz)

    # soft void indicator in the unit cell then tile
    void_uc = torch.sigmoid(float(beta) * (float(void_thresh) - rho_grid))
    void = void_uc.repeat(tx, ty, tz)  # (tx*r, ty*r, tz*r)

    void_conn = void if int(max(0, min_channel_rad)) == 0 else _soft_open3d(void, int(min_channel_rad), periodic_xyz=False)
    Rx, Ry, Rz = void.shape

    # seed mask on *outer* boundary faces of supercell
    seed = torch.zeros_like(void)
    sf = str(seed_faces).lower()
    if sf == "all":
        seed[0, :, :] = 1; seed[-1, :, :] = 1
        seed[:, 0, :] = 1; seed[:, -1, :] = 1
        seed[:, :, 0] = 1; seed[:, :, -1] = 1
    elif sf == "x":
        seed[0, :, :] = 1; seed[-1, :, :] = 1
    elif sf == "y":
        seed[:, 0, :] = 1; seed[:, -1, :] = 1
    elif sf == "z":
        seed[:, :, 0] = 1; seed[:, :, -1] = 1
    elif sf == "xy":
        seed[:, :, 0] = 1; seed[:, :, -1] = 1
    else:
        raise ValueError("seed_faces must be one of: all, x, y, z, xy")

    reach = void_conn * seed

    rad_i = int(max(0, rad))
    if rad_i == 0:
        trapped = torch.relu(void - reach)
    else:
        k = 2 * rad_i + 1
        pad = rad_i
        nsteps = int(iters) if int(iters) > 0 else int(max(Rx, Ry, Rz))

        for _ in range(nsteps):
            x = reach[None, None]  # (1,1,Rx,Ry,Rz)
            # IMPORTANT: non-periodic padding for supercell outside
            x = F.pad(x, (pad, pad, pad, pad, pad, pad), mode="constant", value=0.0)
            dil = F.max_pool3d(x, kernel_size=k, stride=1)[0, 0]
            reach = torch.minimum(void_conn, dil)

        trapped = torch.relu(void - reach)

    if focus_center and (tx >= 3 and ty >= 3 and tz >= 3):
        cx, cy, cz = tx // 2, ty // 2, tz // 2
        xs = slice(cx * r, (cx + 1) * r)
        ys = slice(cy * r, (cy + 1) * r)
        zs = slice(cz * r, (cz + 1) * r)
        trapped = trapped[xs, ys, zs]

    return trapped.mean()

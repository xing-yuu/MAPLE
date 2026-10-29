import numpy as np
import torch
import os
import torch.nn.functional as F
# marching cubes backend
try:
    import mcubes  # PyMCubes
    _HAS_MCUBES = True
except Exception:  # pragma: no cover
    mcubes = None
    _HAS_MCUBES = False
    from skimage import measure  # type: ignore


# -----------------------------
# IO: mesh export
# -----------------------------
def upsample_trilinear(vol: np.ndarray, out_res: int, align_corners: bool = False) -> np.ndarray:
    """(r,r,r) -> (out_res,out_res,out_res) trilinear upsample."""
    x = torch.from_numpy(vol[None, None].astype(np.float32))  # (1,1,r,r,r)
    y = F.interpolate(x, size=(out_res, out_res, out_res), mode="trilinear", align_corners=align_corners)
    return y[0, 0].cpu().numpy()



def _export_obj(vertices: np.ndarray, triangles: np.ndarray, out_obj_path: str) -> None:
    """Write a minimal OBJ file (vertices + faces)."""
    vertices = np.asarray(vertices, dtype=np.float32)
    triangles = np.asarray(triangles, dtype=np.int64)
    with open(out_obj_path, "w", encoding="utf-8") as f:
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        # OBJ is 1-indexed
        for tri in triangles:
            f.write(f"f {tri[0]+1} {tri[1]+1} {tri[2]+1}\n")


def export_rho_mesh(vol: np.ndarray, out_obj_path: str, iso: float = 0.5, pad: int = 2) -> None:
    """
    vol: (res,res,res) density field in [0,1]
    iso: marching-cubes threshold
    pad: zero-padding to reduce boundary artifacts

    Backend:
      - Prefer PyMCubes (import mcubes)
      - Fallback to skimage.measure.marching_cubes if mcubes is unavailable
    """
    vol = np.asarray(vol, dtype=np.float32)
    if pad > 0:
        data = np.zeros((vol.shape[0] + 2 * pad, vol.shape[1] + 2 * pad, vol.shape[2] + 2 * pad), dtype=np.float32)
        data[pad:pad + vol.shape[0], pad:pad + vol.shape[1], pad:pad + vol.shape[2]] = vol
    else:
        data = vol

    try:
        if _HAS_MCUBES:
            vertices, triangles = mcubes.marching_cubes(data, iso)
        else:
            # skimage returns (verts, faces, normals, values)
            vertices, triangles, _, _ = measure.marching_cubes(data, level=iso)

        vmin = vertices.min(axis=0)
        vmax = vertices.max(axis=0)
        center = 0.5 * (vmax + vmin)
        scale = max((vmax - vmin).max() / 2.0, 1e-12)
        vertices = (vertices - center) / scale

        os.makedirs(os.path.dirname(out_obj_path), exist_ok=True)
        _export_obj(vertices, triangles, out_obj_path)
        print(f"[MESH] saved: {out_obj_path}")
    except Exception as e:
        print(f"[MESH] skip : {e}")
    # normalize vertices to roughly [-1,1] for convenient visualization
def save_mesh_from_rho(
    rho_grid: np.ndarray,
    out_obj_path: str,
    out_res: int = 256,
    iso: float = 0.5,
    pad: int = 2) -> None:
    """rho_grid (r,r,r) -> upsample -> marching cubes -> export obj."""
    rho_up = upsample_trilinear(rho_grid.astype(np.float32), out_res=out_res, align_corners=False)
    export_rho_mesh(rho_up, out_obj_path, iso=iso, pad=pad)

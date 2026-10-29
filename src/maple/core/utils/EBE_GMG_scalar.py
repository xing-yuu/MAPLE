import time

import torch
import torch.nn as nn

try:
    from omegaconf import ListConfig  # optional
except Exception:  # pragma: no cover
    ListConfig = tuple  # fallback for type checks

# optional (only needed if you pass spconv.SparseConvTensor as init_list)
try:
    import spconv.pytorch as spconv  # type: ignore
except Exception:  # pragma: no cover
    spconv = None


class ProlongStencil(nn.Module):
    """Stateless matrix-free prolongation / restriction.

    Notes:
      - Works for ANY dof_dim (scalar field -> dof_dim=1).
      - c_idx / w second dimension is the stencil length K
        (3D trilinear => K=8; 2D bilinear => K=4; 1D linear => K=2).
    """

    def __init__(self):
        super().__init__()

    @staticmethod
    def _align_stencil(c_idx: torch.Tensor, w: torch.Tensor, device, w_dtype):
        if c_idx.device != device:
            c_idx = c_idx.to(device=device)
        if c_idx.dtype != torch.long:
            c_idx = c_idx.to(dtype=torch.long)
        if w.device != device or w.dtype != w_dtype:
            w = w.to(device=device, dtype=w_dtype)
        return c_idx, w

    def prolong(self, u_coarse: torch.Tensor, c_idx: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """u_f = P * u_c

        u_coarse: (nC, dof_dim) or (nC,)
        c_idx/w : (nF, K)
        """
        if u_coarse.dim() == 1:
            u_coarse = u_coarse.unsqueeze(-1)

        nC = u_coarse.shape[0]
        dof_dim = u_coarse.shape[1]
        nF = c_idx.shape[0]

        if nF == 0:
            return u_coarse.new_zeros((0, dof_dim))
        if nC == 0:
            return u_coarse.new_zeros((nF, dof_dim))

        c_idx, w = self._align_stencil(c_idx, w, u_coarse.device, u_coarse.dtype)

        ok = (c_idx >= 0) & (c_idx < nC)  # guard
        idx_safe = c_idx.clamp(0, max(nC - 1, 0))

        uc = u_coarse[idx_safe]  # (nF,K,dof_dim)
        contrib = uc * w.unsqueeze(-1)  # (nF,K,dof_dim)
        contrib = contrib * ok.unsqueeze(-1)  # invalid -> 0
        return contrib.sum(dim=1)  # (nF,dof_dim)

    def restrict_PT(self, r_fine: torch.Tensor, nC: int, c_idx: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """r_c = P^T * r_f

        r_fine : (nF, dof_dim) or (nF,)
        return : (nC, dof_dim)
        """
        if r_fine.dim() == 1:
            r_fine = r_fine.unsqueeze(-1)

        dof_dim = r_fine.shape[1]
        if nC == 0:
            return r_fine.new_zeros((0, dof_dim))

        nF = c_idx.shape[0]
        if nF == 0:
            return r_fine.new_zeros((nC, dof_dim))

        c_idx, w = self._align_stencil(c_idx, w, r_fine.device, r_fine.dtype)

        ok = (c_idx >= 0) & (c_idx < nC)
        idx_safe = c_idx.clamp(0, max(nC - 1, 0))

        contrib = r_fine.unsqueeze(1) * w.unsqueeze(-1)  # (nF,K,dof_dim)
        contrib = contrib * ok.unsqueeze(-1)

        rc = torch.zeros((nC, dof_dim), device=r_fine.device, dtype=r_fine.dtype)
        rc.index_add_(0, idx_safe.reshape(-1), contrib.reshape(-1, dof_dim))
        return rc


def get_trilinear_prolongation_matrix_scalar(device, dtype):
    """Local trilinear prolongation for scalar field (3D).

    Fine patch: 3x3x3 nodes => 27 DOF
    Coarse elem: 2x2x2 corners => 8 DOF
    Return P_loc: (27, 8)
    """
    fine_coords = torch.tensor(
        [[x, y, z] for x in range(3) for y in range(3) for z in range(3)],
        dtype=dtype,
        device=device,
    )
    rst = fine_coords - 1.0

    node_signs = torch.tensor(
        [
            [-1, -1, -1],
            [1, -1, -1],
            [1, 1, -1],
            [-1, 1, -1],
            [-1, -1, 1],
            [1, -1, 1],
            [1, 1, 1],
            [-1, 1, 1],
        ],
        dtype=dtype,
        device=device,
    )

    N_mat = torch.zeros((27, 8), dtype=dtype, device=device)
    for i in range(8):
        ri, si, ti = node_signs[i]
        N_mat[:, i] = 0.125 * (1 + rst[:, 0] * ri) * (1 + rst[:, 1] * si) * (1 + rst[:, 2] * ti)

    return N_mat  # (27,8)


def sparse_coarsen_ke_scalar(fine_ke: torch.Tensor, fine_elem_coords: torch.Tensor):
    """Projection-based physical coarsening for scalar trilinear voxel elements.

    fine_ke:
      - (N, 8, 8) : per-voxel scalar element stiffness (8 corner nodes)
    fine_elem_coords:
      - (N,3) : [x,y,z] or
      - (N,4) : [b,x,y,z]  (batch used only for grouping)

    Coarsening rule:
      - 2x2x2 fine elements -> 1 coarse element
      - patch nodes are 3x3x3 => 27 DOF
      - local prolongation P: (27,8) (trilinear)
      - coarse Ke = P^T K_patch P  => (8,8)
    """
    device = fine_ke.device
    dtype = fine_ke.dtype

    if fine_ke.numel() == 0:
        out_dim = fine_elem_coords.shape[1]
        return fine_ke.new_zeros((0, 8, 8)), fine_elem_coords.new_zeros((0, out_dim))

    fine_elem_coords = fine_elem_coords.to(device=device, dtype=torch.long)

    # 1) coarse coords & local child id
    if fine_elem_coords.shape[1] == 4:
        b = fine_elem_coords[:, 0:1]
        xyz = fine_elem_coords[:, 1:4]
        coarse_xyz = xyz // 2
        remainders = xyz % 2
        coarse_coords = torch.cat([b, coarse_xyz], dim=1)
    else:
        xyz = fine_elem_coords
        coarse_coords = xyz // 2
        remainders = xyz % 2

    local_indices = remainders[:, 0] * 4 + remainders[:, 1] * 2 + remainders[:, 2]  # 0..7

    unique_coarse, inverse_idx = torch.unique(coarse_coords, dim=0, return_inverse=True)
    M = unique_coarse.shape[0]

    # 2) collect 8 children Ke into (M,8,8,8)
    ke_grouped = fine_ke.new_zeros((M, 8, 8, 8))
    for i in range(8):
        mask = local_indices == i
        if mask.any():
            coarse_ids = inverse_idx[mask]
            ke_grouped[:, i].index_add_(0, coarse_ids, fine_ke[mask])

    # 3) assemble patch stiffness K_patch (M,27,27)
    K_patch = fine_ke.new_zeros((M, 27, 27))

    child_offsets = torch.tensor(
        [
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 1],
            [1, 0, 0],
            [1, 0, 1],
            [1, 1, 0],
            [1, 1, 1],
        ],
        device=device,
        dtype=torch.long,
    )
    node_offsets = torch.tensor(
        [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1],
        ],
        device=device,
        dtype=torch.long,
    )

    xyz_child = child_offsets.unsqueeze(1) + node_offsets.unsqueeze(0)  # (8,8,3)
    patch_node_id = 9 * xyz_child[..., 0] + 3 * xyz_child[..., 1] + xyz_child[..., 2]  # (8,8)

    for i in range(8):
        sub_ke = ke_grouped[:, i]  # (M,8,8)
        di = patch_node_id[i]  # (8,)
        rows = di.view(-1, 1).repeat(1, 8)
        cols = di.view(1, -1).repeat(8, 1)
        K_patch[:, rows, cols] += sub_ke

    # 4) Galerkin projection
    P = get_trilinear_prolongation_matrix_scalar(device, dtype)  # (27,8)
    coarse_ke = torch.matmul(P.t(), torch.matmul(K_patch, P))  # (M,8,8)

    return coarse_ke, unique_coarse


class EBEOperatorScalar(nn.Module):
    """Element-by-Element operator for scalar field (1 DOF per node)."""

    def __init__(self):
        super().__init__()

    @staticmethod
    def _compute_colors(coords: torch.Tensor):
        """Graph coloring on structured grid based on parity.

        coords:
          - (N,3): [x,y,z] -> 8 colors
          - (N,4): [b,x,y,z] -> 8 colors per batch (batch excluded from parity)
          - (N,2): -> 4 colors
          - (N,1): -> 2 colors
        """
        coords = coords.long()
        if coords.numel() == 0:
            return []

        if coords.shape[1] == 4:
            xyz = coords[:, 1:]
        else:
            xyz = coords

        d = xyz.shape[1]
        if d < 1:
            return []

        color_ids = torch.zeros((xyz.shape[0],), device=xyz.device, dtype=torch.long)
        for i in range(d):
            bit = (xyz[:, i] % 2).long()
            color_ids += bit * (2 ** (d - 1 - i))

        n_colors = int(2**d)
        return [torch.where(color_ids == c)[0] for c in range(n_colors)]

    @staticmethod
    def _compute_diag_inverse(Ke: torch.Tensor, num_nodes: int, dof_indices: torch.Tensor, fixed_mask=None):
        """Inverse of assembled diagonal (scalar DOF).

        Ke: (E,8,8)
        dof_indices: (E,8) global node indices
        fixed_mask: (num_nodes,) or (num_nodes,1) bool
        """
        device = Ke.device
        dtype = Ke.dtype

        elem_diag = torch.diagonal(Ke, dim1=-2, dim2=-1)  # (E,8)
        global_diag = torch.zeros((num_nodes,), device=device, dtype=dtype)
        global_diag.index_add_(0, dof_indices.reshape(-1), elem_diag.reshape(-1))

        global_diag[global_diag == 0] = 1.0
        if fixed_mask is not None:
            fixed_flat = fixed_mask.reshape(-1).to(device=device)
            global_diag[fixed_flat] = 1.0

        return 1.0 / global_diag

    @staticmethod
    def matvec(Ke: torch.Tensor, u: torch.Tensor, dof_indices: torch.Tensor) -> torch.Tensor:
        """Global matvec y = K u (scalar) via EBE.

        u: (nNodes,1) or (nNodes,)
        returns: (nNodes,1)
        """
        u_flat = u.reshape(-1).to(dtype=Ke.dtype)

        u_elem = u_flat[dof_indices]  # (E,8)
        f_elem = torch.bmm(Ke, u_elem.unsqueeze(-1)).squeeze(-1)  # (E,8)

        v_flat = torch.zeros_like(u_flat)
        v_flat.index_add_(0, dof_indices.reshape(-1), f_elem.reshape(-1))
        return v_flat.view(-1, 1)

    def smooth_gs_color(
        self,
        Ke: torch.Tensor,
        u: torch.Tensor,
        f: torch.Tensor,
        node_coords: torch.Tensor,
        dof_indices: torch.Tensor,
        color_indices,
        diag_inv=None,
        fixed_mask=None,
        omega: float = 1.0,
        iterations: int = 1,
    ):
        """Colored Gauss-Seidel smoother (scalar)."""
        u_work = u.view(-1, 1).to(dtype=Ke.dtype)
        f_work = f.view(-1, 1).to(dtype=Ke.dtype)

        if diag_inv is None:
            diag_inv = self._compute_diag_inverse(Ke, node_coords.shape[0], dof_indices, fixed_mask=fixed_mask)
        diag_inv = diag_inv.view(-1, 1)

        for _ in range(iterations):
            for nodes_c in color_indices:
                if nodes_c.numel() == 0:
                    continue
                r = f_work - self.matvec(Ke, u_work, dof_indices)
                u_work[nodes_c] += omega * diag_inv[nodes_c] * r[nodes_c]

            if fixed_mask is not None:
                u_work = u_work.clone()
                u_work[fixed_mask.reshape(-1)] = 0.0

        return u_work


class GMGSolverScalar(nn.Module):
    """Geometric multigrid V-cycle solver for scalar field on voxel grid."""

    def __init__(self):
        super().__init__()
        self.P_op = ProlongStencil()

    def _generate_next_level_topology(self, elem_coords: torch.Tensor):
        """Voxel element -> 8 corner nodes (periodic wrap).

        Supports:
          - elem_coords: (E,3)
          - elem_coords: (E,4)=[b,x,y,z]
        """
        device = elem_coords.device

        offsets = torch.tensor(
            [
                [0, 0, 0],
                [1, 0, 0],
                [1, 1, 0],
                [0, 1, 0],
                [0, 0, 1],
                [1, 0, 1],
                [1, 1, 1],
                [0, 1, 1],
            ],
            device=device,
            dtype=elem_coords.dtype,
        )

        if elem_coords.numel() == 0:
            topo_indices = torch.empty((0, 8), device=device, dtype=torch.long)
            out_dim = elem_coords.shape[1]
            unique_nodes = torch.empty((0, out_dim), device=device, dtype=elem_coords.dtype)
            return topo_indices, unique_nodes

        if elem_coords.shape[1] == 4:
            b = elem_coords[:, 0:1].to(device=device)
            xyz = elem_coords[:, 1:4].to(device=device)

            dims = (xyz.max(dim=0).values + 1).clamp(min=1)
            all_xyz = (xyz.unsqueeze(1) + offsets.unsqueeze(0)) % dims.view(1, 1, 3)
            b_rep = b.unsqueeze(1).expand(-1, 8, 1)
            all_node_coords = torch.cat([b_rep, all_xyz], dim=2)  # (E,8,4)

            flat = all_node_coords.reshape(-1, 4)
            unique_nodes, inverse = torch.unique(flat, dim=0, return_inverse=True)
            topo_indices = inverse.reshape(-1, 8).to(torch.long)
            return topo_indices, unique_nodes

        xyz = elem_coords.to(device=device)
        dims = (xyz.max(dim=0).values + 1).clamp(min=1)

        all_node_coords = (xyz.unsqueeze(1) + offsets.unsqueeze(0)) % dims.view(1, 1, 3)
        flat = all_node_coords.reshape(-1, 3)

        unique_nodes, inverse = torch.unique(flat, dim=0, return_inverse=True)
        topo_indices = inverse.reshape(-1, 8).to(torch.long)
        return topo_indices, unique_nodes

    def _build_prolongation_operator_robust(self, fine_nodes, coarse_nodes):
        """Periodic trilinear interpolation stencil (fine <- coarse) for 3D voxel grid.

        Returns:
          c_idx: (nF,8), w: (nF,8)
        """
        device = fine_nodes.device
        dtype = torch.float

        fine_nodes = fine_nodes.to(device=device, dtype=torch.long)
        coarse_nodes = coarse_nodes.to(device=device, dtype=torch.long)

        # --- batched path ---
        if fine_nodes.shape[1] == 4 or coarse_nodes.shape[1] == 4:
            if fine_nodes.shape[1] == 3:
                fine_nodes = torch.cat(
                    [torch.zeros((fine_nodes.shape[0], 1), device=device, dtype=torch.long), fine_nodes], dim=1
                )
            if coarse_nodes.shape[1] == 3:
                coarse_nodes = torch.cat(
                    [torch.zeros((coarse_nodes.shape[0], 1), device=device, dtype=torch.long), coarse_nodes], dim=1
                )

            bF = fine_nodes[:, 0]
            xyzF = fine_nodes[:, 1:]
            bC = coarse_nodes[:, 0]
            xyzC = coarse_nodes[:, 1:]

            coarse_dims = (xyzC.max(dim=0).values + 1).clamp(min=1)
            Dx, Dy, Dz = [int(x.item()) for x in coarse_dims]
            stride_yz = Dy * Dz
            stride_b = Dx * Dy * Dz

            linC = bC * stride_b + xyzC[:, 0] * stride_yz + xyzC[:, 1] * Dz + xyzC[:, 2]
            sort_idx = torch.argsort(linC)
            linC_sorted = linC[sort_idx]

            base = (xyzF // 2) % coarse_dims.view(1, 3)
            remainder = xyzF % 2

            offsets8 = torch.tensor(
                [
                    [0, 0, 0],
                    [0, 0, 1],
                    [0, 1, 0],
                    [0, 1, 1],
                    [1, 0, 0],
                    [1, 0, 1],
                    [1, 1, 0],
                    [1, 1, 1],
                ],
                device=device,
                dtype=torch.long,
            )

            c_idx = torch.full((xyzF.shape[0], 8), -1, device=device, dtype=torch.long)
            w = torch.zeros((xyzF.shape[0], 8), device=device, dtype=dtype)

            for k in range(8):
                off = offsets8[k].view(1, 3)
                target_xyz = (base + off) % coarse_dims.view(1, 3)
                linT = bF * stride_b + target_xyz[:, 0] * stride_yz + target_xyz[:, 1] * Dz + target_xyz[:, 2]

                pos = torch.searchsorted(linC_sorted, linT)
                valid = pos < linC_sorted.numel()
                pos_v = pos[valid]
                linT_v = linT[valid]
                ok = linC_sorted[pos_v] == linT_v
                if ok.any():
                    fine_rows = torch.nonzero(valid, as_tuple=False).squeeze(1)[ok]
                    coarse_rows = sort_idx[pos_v[ok]]

                    d_vec = torch.abs(0.5 * remainder[fine_rows].to(torch.float32) - off.to(torch.float32))
                    w_k = (1.0 - d_vec).prod(dim=1).to(dtype)

                    c_idx[fine_rows, k] = coarse_rows
                    w[fine_rows, k] = w_k

            return (c_idx, w)

        # --- unbatched path ---
        coarse_dims = (coarse_nodes.max(dim=0).values + 1).clamp(min=1)

        coarse_lookup = torch.full(
            (int(coarse_dims[0].item()), int(coarse_dims[1].item()), int(coarse_dims[2].item())),
            -1,
            device=device,
            dtype=torch.long,
        )
        coarse_lookup[coarse_nodes[:, 0], coarse_nodes[:, 1], coarse_nodes[:, 2]] = torch.arange(
            coarse_nodes.shape[0], device=device, dtype=torch.long
        )

        base_coarse = fine_nodes // 2
        remainder = fine_nodes % 2

        offsets8 = torch.tensor(
            [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]],
            device=device,
            dtype=torch.long,
        )

        c_idx = torch.full((fine_nodes.shape[0], 8), -1, device=device, dtype=torch.long)
        w = torch.zeros((fine_nodes.shape[0], 8), device=device, dtype=dtype)

        for k in range(8):
            off = offsets8[k].view(1, 3)
            target = (base_coarse + off) % coarse_dims.view(1, 3)
            c_k = coarse_lookup[target[:, 0], target[:, 1], target[:, 2]]

            d_vec = torch.abs(0.5 * remainder.to(torch.float32) - off.to(torch.float32))
            w_k = (1.0 - d_vec).prod(dim=1).to(dtype)

            ok = c_k >= 0
            if ok.any():
                c_idx[ok, k] = c_k[ok]
                w[ok, k] = w_k[ok]

        return (c_idx, w)

    def _spconv_to_dense_u(self, sp_tensor, op_levels, level: int, batch_idx=None, feat_slice=slice(0, 1)):
        """Align spconv.SparseConvTensor to GMG node order -> dense (nNodes,1)."""
        if sp_tensor is None:
            node_coords = op_levels[level]["node_coords"]
            return torch.zeros((node_coords.shape[0], 1), device=node_coords.device, dtype=torch.float32)

        device = sp_tensor.features.device
        dtype = sp_tensor.features.dtype
        node_coords = op_levels[level]["node_coords"].to(device=device, dtype=torch.long)
        n_nodes = node_coords.shape[0]
        u0 = torch.zeros((n_nodes, 1), device=device, dtype=dtype)

        idx = getattr(sp_tensor, "indices", None)
        feat = getattr(sp_tensor, "features", None)
        if idx is None or feat is None:
            raise TypeError("Need an object with .indices/.features (spconv.SparseConvTensor or similar).")
        if idx.numel() == 0:
            return u0

        idx = idx.to(device=device, dtype=torch.long)
        vals = feat.to(device=device)

        # node coords
        if node_coords.shape[1] == 4:
            b_nodes = node_coords[:, 0]
            xyz_nodes = node_coords[:, 1:]
        else:
            b_nodes = torch.zeros((n_nodes,), device=device, dtype=torch.long)
            xyz_nodes = node_coords

        # sparse coords
        if idx.shape[1] == 4:
            b_sp = idx[:, 0]
            xyz_sp = idx[:, 1:]
        else:
            b_sp = torch.zeros((idx.shape[0],), device=device, dtype=torch.long)
            xyz_sp = idx

        if batch_idx is not None:
            keep = b_sp == int(batch_idx)
            b_sp = b_sp[keep]
            xyz_sp = xyz_sp[keep]
            vals = vals[keep]

        vals = vals[:, feat_slice]
        if vals.shape[1] < 1:
            raise ValueError(f"init_list feature dim < 1: got {vals.shape}")
        vals = vals[:, :1]

        dims = (xyz_nodes.max(dim=0).values + 1).clamp(min=1)
        xyz_sp = xyz_sp % dims.view(1, 3)

        Dx, Dy, Dz = [int(x.item()) for x in dims]
        stride_yz = Dy * Dz
        stride_b = Dx * Dy * Dz

        lin_nodes = b_nodes * stride_b + xyz_nodes[:, 0] * stride_yz + xyz_nodes[:, 1] * Dz + xyz_nodes[:, 2]
        lin_sp = b_sp * stride_b + xyz_sp[:, 0] * stride_yz + xyz_sp[:, 1] * Dz + xyz_sp[:, 2]

        sort_idx = torch.argsort(lin_nodes)
        lin_sorted = lin_nodes[sort_idx]

        pos = torch.searchsorted(lin_sorted, lin_sp)
        valid = pos < lin_sorted.numel()
        pos = pos[valid]
        lin_sp_v = lin_sp[valid]

        valid2 = lin_sorted[pos] == lin_sp_v
        if not valid2.any():
            return u0

        rows = sort_idx[pos[valid2]]
        vals_v = vals[valid][valid2].to(device=device, dtype=dtype)

        u0.index_add_(0, rows, vals_v)
        return u0

    def _prepare_init_levels(self, init_list, op_levels, batch_idx=None, feat_slice=slice(0, 1)):
        """Convert init_list (spconv tensors per level) to GMG-aligned dense init per level."""
        if init_list is None:
            return None
        if not isinstance(init_list, (list, tuple)):
            raise TypeError("init_list must be list/tuple of SparseConvTensor for each level.")

        L = len(op_levels)
        init_levels = [None] * L
        for gmg_level in range(min(L, len(init_list))):
            sp_level = init_list[-1 - gmg_level]  # finest -> level0
            init_levels[gmg_level] = self._spconv_to_dense_u(
                sp_level, op_levels, level=gmg_level, batch_idx=batch_idx, feat_slice=feat_slice
            )
        return init_levels

    def solve(
        self,
        fine_ke: torch.Tensor,
        global_f: torch.Tensor,
        node_coords_L0: torch.Tensor,
        fine_elem_coords: torch.Tensor,
        fine_topo_indices: torch.Tensor,
        max_iter: int = 50,
        smooth_iter=1,
        init_list=None,
        init_batch=None,
        init_feat_slice=slice(0, 1),
        init_is_solution: bool = False,
        max_levels: int = 5,
        fixed_mask=None,
        dtype=torch.float,
    ):
        """MG solve for scalar field.

        Required shapes:
          fine_ke:        (nElem, 8, 8)
          fine_topo_idx:  (nElem, 8)
          global_f:       (nNodes, 1) or (nNodes,)
        """
        device = fine_ke.device

        global_f = global_f.view(-1, 1)
        if global_f.dtype != dtype:
            global_f = global_f.to(dtype=dtype, device=device)

        if fixed_mask is not None:
            fixed_mask = fixed_mask.to(device=device, dtype=torch.bool).reshape(-1)

        op_levels = []

        # Level 0
        op0 = EBEOperatorScalar()
        dof_indices_L0 = fine_topo_indices.long().to(device=device)
        num_nodes_L0 = global_f.shape[0]

        op_levels.append(
            {
                "op": op0,
                "node_coords": node_coords_L0,
                "elem_coords": fine_elem_coords,
                "P": None,
                "Ke": fine_ke,
                "dof_indices": dof_indices_L0,
                "diag_inv": op0._compute_diag_inverse(fine_ke, num_nodes_L0, dof_indices_L0, fixed_mask=fixed_mask),
                "color_indices": op0._compute_colors(node_coords_L0),
            }
        )

        current_ke = fine_ke
        current_elem_coords = fine_elem_coords
        current_node_coords = node_coords_L0

        # Build coarser levels
        for _l in range(1, max_levels):
            next_ke, next_elem_coords = sparse_coarsen_ke_scalar(current_ke, current_elem_coords)
            if next_ke.shape[0] == 0:
                break

            next_topo_indices, next_node_coords = self._generate_next_level_topology(next_elem_coords)
            op_next = EBEOperatorScalar()

            op_levels.append(
                {
                    "op": op_next,
                    "node_coords": next_node_coords,
                    "elem_coords": next_elem_coords,
                    "P": self._build_prolongation_operator_robust(current_node_coords, next_node_coords),
                    "Ke": next_ke,
                    "dof_indices": next_topo_indices.long(),
                    "diag_inv": op_next._compute_diag_inverse(next_ke, next_node_coords.shape[0], next_topo_indices.long(), fixed_mask=None),
                    "color_indices": op_next._compute_colors(next_node_coords),
                }
            )

            current_ke = next_ke
            current_elem_coords = next_elem_coords
            current_node_coords = next_node_coords

        # Init from network (optional)
        u0_levels = self._prepare_init_levels(init_list, op_levels, batch_idx=init_batch, feat_slice=init_feat_slice)

        if u0_levels is not None and u0_levels[0] is not None:
            u = u0_levels[0]
        else:
            u = torch.zeros_like(global_f)

        if fixed_mask is not None:
            u = u.clone()
            u[fixed_mask] = 0.0

        for _ in range(max_iter):
            u = self._v_cycle(
                0,
                u,
                global_f,
                u0_levels=u0_levels,
                op_levels=op_levels,
                smooth_iter=smooth_iter,
                init_is_solution=init_is_solution,
                fixed_mask_L0=fixed_mask,
            )

        r = global_f - op_levels[0]["op"].matvec(op_levels[0]["Ke"], u, op_levels[0]["dof_indices"])
        return u, r

    def _v_cycle(
        self,
        level: int,
        u: torch.Tensor,
        f: torch.Tensor,
        u0_levels=None,
        op_levels=None,
        smooth_iter=1,
        init_is_solution: bool = True,
        fixed_mask_L0=None,
    ):
        op = op_levels[level]["op"]
        lvl = op_levels[level]
        fixed = fixed_mask_L0 if level == 0 else None

        if isinstance(smooth_iter, (list, tuple, ListConfig)):
            if len(smooth_iter) == 0:
                iter_smooth = 1
            elif level < len(smooth_iter):
                iter_smooth = smooth_iter[level]
            else:
                iter_smooth = smooth_iter[-1]
        else:
            iter_smooth = int(smooth_iter)

        # 1) Pre-smoothing
        u = op.smooth_gs_color(
            lvl["Ke"],
            u,
            f,
            lvl["node_coords"],
            lvl["dof_indices"],
            lvl["color_indices"],
            diag_inv=lvl["diag_inv"],
            fixed_mask=fixed,
            iterations=iter_smooth,
        )

        # Coarsest level
        if level == len(op_levels) - 1:
            return op.smooth_gs_color(
                lvl["Ke"],
                u,
                f,
                lvl["node_coords"],
                lvl["dof_indices"],
                lvl["color_indices"],
                diag_inv=lvl["diag_inv"],
                fixed_mask=fixed,
                iterations=1,
            )

        # 2) Residual
        r = f - op.matvec(lvl["Ke"], u, lvl["dof_indices"])
        if fixed is not None:
            r = r.clone()
            r[fixed] = 0.0

        # 3) Restriction
        c_idx, w = op_levels[level + 1]["P"]
        nC = op_levels[level + 1]["node_coords"].shape[0]
        r_coarse = self.P_op.restrict_PT(r, nC=nC, c_idx=c_idx, w=w)

        # 4) Coarse init injection (optional)
        if u0_levels is not None and (level + 1) < len(u0_levels) and u0_levels[level + 1] is not None:
            u_pred_coarse = u0_levels[level + 1].to(device=r_coarse.device, dtype=r_coarse.dtype)
            if init_is_solution:
                u_restricted = self.P_op.restrict_PT(u, nC=nC, c_idx=c_idx, w=w).to(device=r_coarse.device, dtype=r_coarse.dtype)
                u_coarse = u_pred_coarse - u_restricted
            else:
                u_coarse = u_pred_coarse
        else:
            u_coarse = torch.zeros_like(r_coarse)

        # 5) Recursion
        u_coarse = self._v_cycle(
            level + 1,
            u_coarse,
            r_coarse,
            u0_levels=u0_levels,
            op_levels=op_levels,
            smooth_iter=smooth_iter,
            init_is_solution=init_is_solution,
            fixed_mask_L0=None,
        )

        # 6) Prolongation
        correction = self.P_op.prolong(u_coarse, c_idx=c_idx, w=w)
        u_new = u + correction
        if fixed is not None:
            u_new = u_new.clone()
            u_new[fixed] = 0.0

        # 7) Post-smoothing
        u_new = op.smooth_gs_color(
            lvl["Ke"],
            u_new,
            f,
            lvl["node_coords"],
            lvl["dof_indices"],
            lvl["color_indices"],
            diag_inv=lvl["diag_inv"],
            fixed_mask=fixed,
            iterations=iter_smooth,
        )
        return u_new


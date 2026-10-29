import torch
import torch.nn as nn
import numpy as np
import time
import copy
try:
    from torch_scatter import scatter_add, scatter_mean  # optional, faster
except Exception:  # pragma: no cover
    scatter_add = None
    scatter_mean = None

    def _scatter_add_fallback(src: torch.Tensor, index: torch.Tensor, dim: int = 0, dim_size: int = None) -> torch.Tensor:
        if dim != 0:
            raise NotImplementedError('fallback scatter_add only supports dim=0')
        if dim_size is None:
            dim_size = int(index.max().item()) + 1 if index.numel() > 0 else 0
        out_shape = (dim_size,) + src.shape[1:]
        out = src.new_zeros(out_shape)
        out.index_add_(0, index, src)
        return out

    def _scatter_mean_fallback(src: torch.Tensor, index: torch.Tensor, dim: int = 0, dim_size: int = None) -> torch.Tensor:
        out = _scatter_add_fallback(src, index, dim=dim, dim_size=dim_size)
        if dim != 0:
            raise NotImplementedError('fallback scatter_mean only supports dim=0')
        if dim_size is None:
            dim_size = out.shape[0]
        count = src.new_zeros((dim_size,))
        ones = src.new_ones((index.shape[0],))
        count.index_add_(0, index, ones)
        count = count.clamp(min=1.0).reshape(-1, *([1] * (src.dim() - 1)))
        return out / count

    def scatter_add(src: torch.Tensor, index: torch.Tensor, dim: int = 0, dim_size: int = None) -> torch.Tensor:
        return _scatter_add_fallback(src, index, dim=dim, dim_size=dim_size)

    def scatter_mean(src: torch.Tensor, index: torch.Tensor, dim: int = 0, dim_size: int = None) -> torch.Tensor:
        return _scatter_mean_fallback(src, index, dim=dim, dim_size=dim_size)
def isotropic_elastic_tensor(E, v):
    Lambda = v / (1. + v) / (1 - 2. * v)*E
    Mu = 1. / (2.*(1. + v))*E
    return torch.as_tensor([
        [Lambda + 2 * Mu, Lambda, Lambda, 0, 0, 0],
        [Lambda, Lambda + 2 * Mu, Lambda, 0, 0, 0],
        [Lambda, Lambda, Lambda + 2 * Mu, 0, 0, 0],
        [0, 0, 0, Mu, 0, 0],
        [0, 0, 0, 0, Mu, 0],
        [0, 0, 0, 0, 0, Mu]])
def hexahedron(r,C):
    device = C.device
    dx = 1. / r / 2
    dy = 1. / r / 2
    dz = 1. / r / 2

    pp = torch.as_tensor(
        [-pow(3 / 5, 0.5), 0, pow(3 / 5, 0.5)], dtype=C.dtype, device=device)
    ww = torch.as_tensor([5 / 9, 8 / 9, 5 / 9],
                            dtype=C.dtype, device=device)
    Ke = torch.zeros(24, 24, dtype=C.dtype, device=device)
    Fe = torch.zeros(24, 6, dtype=C.dtype, device=device)

    dxdydz = torch.as_tensor(
        [[-dx, dx, dx, -dx, -dx, dx, dx, -dx], [-dy, -dy, dy, dy, -dy, -dy, dy, dy],
            [-dz, -dz, -dz, -dz, dz, dz, dz, dz]], dtype=C.dtype, device=device).t()
    for i in range(3):
        for j in range(3):
            for k in range(3):
                x = pp[i]
                y = pp[j]
                z = pp[k]
                qxqyqz = torch.as_tensor(
                    [[-((y - 1) * (z - 1)) / 8, ((y - 1) * (z - 1)) / 8, -((y + 1) * (z - 1)) / 8,
                        ((y + 1) * (z - 1)) / 8, ((y - 1) *
                                                (z + 1)) / 8, -((y - 1) * (z + 1)) / 8,
                        ((y + 1) * (z + 1)) / 8, -((y + 1) * (z + 1)) / 8],
                        [-((x - 1) * (z - 1)) / 8, ((x + 1) * (z - 1)) / 8, -((x + 1) * (z - 1)) / 8,
                        ((x - 1) * (z - 1)) / 8, ((x - 1) * (z + 1)) /
                        8, -((x + 1) * (z + 1)) / 8,
                        ((x + 1) * (z + 1)) / 8, -((x - 1) * (z + 1)) / 8],
                        [-((x - 1) * (y - 1)) / 8, ((x + 1) * (y - 1)) / 8, -((x + 1) * (y + 1)) / 8,
                        ((x - 1) * (y + 1)) / 8, ((x - 1) * (y - 1)) /
                        8, -((x + 1) * (y - 1)) / 8,
                        ((x + 1) * (y + 1)) / 8, -((x - 1) * (y + 1)) / 8]], dtype=C.dtype, device=device)

                J = qxqyqz @ dxdydz
                invJ = torch.inverse(J)
                qxyz = invJ @ qxqyqz
                B = torch.zeros(6, 24, dtype=C.dtype, device=device)

                for i_B in range(8):
                    B[:, i_B * 3:(i_B + 1) * 3] = torch.as_tensor(
                        [[qxyz[0, i_B], 0, 0],
                            [0, qxyz[1, i_B], 0],
                            [0, 0, qxyz[2, i_B]],
                            [qxyz[1, i_B],
                            qxyz[0, i_B], 0],
                            [0, qxyz[2, i_B],
                            qxyz[1, i_B]],
                            [qxyz[2, i_B], 0, qxyz[0, i_B]]], dtype=C.dtype,
                        device=device)

                weight = J.det() * ww[i] * ww[j] * ww[k]
                Ke = Ke + weight * B.transpose(0, 1) @ C @ B
                Fe = Fe + weight * B.transpose(0, 1) @ C
    idx = torch.ones(24, dtype=torch.bool, device=device)
    idx[[0, 1, 2, 4, 5, 11]] = False
    X0 = torch.zeros(24, 6, dtype=C.dtype, device=device)
    X0[idx, :] = torch.inverse(Ke[idx, :][:, idx])@Fe[idx, :]
    return Ke, Fe, X0


# -----------------------------
# 8-node hexahedral element for scalar conduction (Laplace)
# -----------------------------

def hexahedron_scalar(r: int, kappa: float = 1.0):
    """Build element matrices for scalar thermal conduction.

    Returns:
        Ke: (8,8)   element stiffness for Laplace (unit conductivity * kappa)
        Fe: (8,3)   element RHS for 3 unit macroscopic gradients (x/y/z)
        X0: (8,3)   reference nodal field for solid element (anchor one node)

    Notes:
      - This is the scalar analogue of `hexahedron()` used for elasticity.
      - We anchor node 0 (fix one DOF) to make the local solve well-posed.
    """
    device = torch.device('cpu')
    dtype = torch.float32
    # support passing device via kappa as tensor (optional)
    if isinstance(kappa, torch.Tensor):
        device = kappa.device
        dtype = kappa.dtype
        kappa_val = float(kappa.item())
    else:
        kappa_val = float(kappa)

    dx = 1.0 / r / 2
    dy = 1.0 / r / 2
    dz = 1.0 / r / 2

    pp = torch.as_tensor([-pow(3 / 5, 0.5), 0.0, pow(3 / 5, 0.5)], dtype=dtype, device=device)
    ww = torch.as_tensor([5 / 9, 8 / 9, 5 / 9], dtype=dtype, device=device)

    Ke = torch.zeros(8, 8, dtype=dtype, device=device)
    Fe = torch.zeros(8, 3, dtype=dtype, device=device)

    dxdydz = torch.as_tensor(
        [[-dx, dx, dx, -dx, -dx, dx, dx, -dx],
         [-dy, -dy, dy, dy, -dy, -dy, dy, dy],
         [-dz, -dz, -dz, -dz, dz, dz, dz, dz]],
        dtype=dtype,
        device=device,
    ).t()  # (8,3)

    for i in range(3):
        for j in range(3):
            for k in range(3):
                x = pp[i]
                y = pp[j]
                z = pp[k]
                qxqyqz = torch.as_tensor(
                    [[-((y - 1) * (z - 1)) / 8, ((y - 1) * (z - 1)) / 8, -((y + 1) * (z - 1)) / 8,
                      ((y + 1) * (z - 1)) / 8, ((y - 1) * (z + 1)) / 8, -((y - 1) * (z + 1)) / 8,
                      ((y + 1) * (z + 1)) / 8, -((y + 1) * (z + 1)) / 8],
                     [-((x - 1) * (z - 1)) / 8, ((x + 1) * (z - 1)) / 8, -((x + 1) * (z - 1)) / 8,
                      ((x - 1) * (z - 1)) / 8, ((x - 1) * (z + 1)) / 8, -((x + 1) * (z + 1)) / 8,
                      ((x + 1) * (z + 1)) / 8, -((x - 1) * (z + 1)) / 8],
                     [-((x - 1) * (y - 1)) / 8, ((x + 1) * (y - 1)) / 8, -((x + 1) * (y + 1)) / 8,
                      ((x - 1) * (y + 1)) / 8, ((x - 1) * (y - 1)) / 8, -((x + 1) * (y - 1)) / 8,
                      ((x + 1) * (y + 1)) / 8, -((x - 1) * (y + 1)) / 8]],
                    dtype=dtype,
                    device=device,
                )  # (3,8)

                J = qxqyqz @ dxdydz  # (3,3)
                invJ = torch.inverse(J)
                qxyz = invJ @ qxqyqz  # (3,8)

                weight = J.det() * ww[i] * ww[j] * ww[k]

                # Ke = ∫ (∇N)^T k ∇N dV
                Ke = Ke + weight * kappa_val * (qxyz.transpose(0, 1) @ qxyz)

                # Fe = ∫ (∇N)^T k * I dV  -> (8,3)
                Fe = Fe + weight * kappa_val * qxyz.transpose(0, 1)

    # Build X0 by anchoring one node to remove the constant nullspace
    idx = torch.ones(8, dtype=torch.bool, device=device)
    idx[0] = False
    X0 = torch.zeros(8, 3, dtype=dtype, device=device)
    Ke_ff = Ke[idx, :][:, idx]
    Fe_ff = Fe[idx, :]
    # Solve Ke_ff * X0_ff = Fe_ff
    X0[idx, :] = torch.linalg.solve(Ke_ff, Fe_ff)
    return Ke, Fe, X0

def assembly(voxel, Ke, Fe,anchor=False):
        """assemble stiffness matrix K and force matrix for solid cell

        Args:
           voxel (torch.cuda.FloatTensor): Voxel, it should be N*N*N 
           Ke (torch.cuda.FloatTensor): Elelment stiffness matrix, it should be 24*24 
           Fe (torch.cuda.FloatTensor): Elelment macrostrain-force matrix, it should be 24*6 

        Returns:
            K(torch.cuda.Soarse_COO_Tensor)
            F(torch.cuda.FloatTensor)
        """
        device = voxel.device
        voxel_numel=torch.count_nonzero(voxel)
        voxel_coo=torch.nonzero(voxel, as_tuple=False)
        node=voxel.clone().long()
        hex=torch.tensor([[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1]],dtype=torch.int,device=device)
        for i in hex:
            tmp=(voxel_coo+i+voxel.shape[0])%voxel.shape[0]
            node[tmp[:, 0], tmp[:, 1], tmp[:, 2]] = 1
        node_numel=torch.count_nonzero(node)
        node_coo=torch.nonzero(node, as_tuple=False)
        nodeidx = torch.arange(0, node_numel,device=device)
        # print(node_coo.shape,node.shape,torch.max(node_coo))
        node[node_coo[:, 0], node_coo[:, 1], node_coo[:, 2]]=nodeidx

        dof = torch.zeros(voxel_numel,8,3, device=device)
        kij = torch.zeros(2,voxel_numel * 8 * 3 * 8 * 3,dtype=torch.float, device=device)
        for i in range(0,8):
            now_node_coo=(voxel_coo+hex[i]+voxel.shape[0])%voxel.shape[0]
            dof[:,i,0] = node[now_node_coo[:,0],now_node_coo[:,1],now_node_coo[:,2]]*3
        kij[0,:]=dof.reshape(-1,1).repeat(1, 8).reshape(-1)
        kij[1,:]=dof.repeat_interleave(8, dim=0).reshape(-1) 

        fij = torch.zeros(2,voxel_numel*8*3, dtype=torch.float, device=device)
        fij[0,:]=dof.reshape(-1,1).repeat(1, 3).reshape(-1)
        fij[1,:]=torch.arange(0, 3, device=device).reshape(1,1,-1).repeat(voxel_numel, 8, 1).contiguous().view(-1)

        vK = Ke.repeat(voxel_numel, 1, 1).reshape(-1)
        vF = Fe.repeat(voxel_numel, 1, 1).reshape(-1)

        if anchor :
            mask=(torch.logical_or(kij[0,:]==0 , kij[1,:]==0 ))
            vK[mask]=0
            mask=( fij[0,:]==0)
            vF[mask]=0

        K = torch.sparse_coo_tensor(kij, vK.contiguous().view(-1), (node_numel,node_numel), device=device).coalesce()
        
        F = torch.sparse_coo_tensor(fij, vF.contiguous().view(-1), (node_numel,3), device=device).coalesce().to_dense()

        K = 0.5 * (K.transpose(0, 1) + K)

        return K.coalesce() , F
def assembly_F(voxel, Fe,anchor=False):
        """assemble stiffness matrix K and force matrix for solid cell

        Args:
           voxel (torch.cuda.FloatTensor): Voxel, it should be N*N*N 
           Ke (torch.cuda.FloatTensor): Elelment stiffness matrix, it should be 24*24 
           Fe (torch.cuda.FloatTensor): Elelment macrostrain-force matrix, it should be 24*6 

        Returns:
            K(torch.cuda.Soarse_COO_Tensor)
            F(torch.cuda.FloatTensor)
        """
        device = voxel.device
        voxel_numel=torch.count_nonzero(voxel)
        voxel_coo=torch.nonzero(voxel, as_tuple=False)
        node=voxel.clone().long()
        hex=torch.tensor([[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1]],dtype=torch.int,device=device)
        for i in hex:
            tmp=(voxel_coo+i+voxel.shape[0])%voxel.shape[0]
            node[tmp[:, 0], tmp[:, 1], tmp[:, 2]] = 1
        node_numel=torch.count_nonzero(node)
        node_coo=torch.nonzero(node, as_tuple=False)
        nodeidx = torch.arange(0, node_numel,device=device)
        # print(node_coo.shape,node.shape,torch.max(node_coo))
        node[node_coo[:, 0], node_coo[:, 1], node_coo[:, 2]]=nodeidx

        dof = torch.zeros(voxel_numel,8, device=device)
        for i in range(0,8):
            now_node_coo=(voxel_coo+hex[i]+voxel.shape[0])%voxel.shape[0]
            dof[:,i] = node[now_node_coo[:,0],now_node_coo[:,1],now_node_coo[:,2]]

        fij = torch.zeros(2,voxel_numel * 8 * 3 * 6, dtype=torch.float, device=device)
        fij[0,:]=dof.reshape(-1,1).repeat(1, 3 * 6).reshape(-1)
        fij[1,:]=torch.arange(0, 3 * 6, device=device).reshape(1,1,-1).repeat(voxel_numel, 8, 1).contiguous().view(-1)

        vF = Fe.repeat(voxel_numel, 1, 1).reshape(-1)

        if anchor :
            mask=( fij[0,:]==0)
            vF[mask]=0
        
        F = torch.sparse_coo_tensor(fij, vF.contiguous().view(-1), (node_numel,3*6), device=device).coalesce().to_dense()

        return F

def get_order(indices_list):
    batch_size=len(indices_list)
    order_list=[]

    for i in range (batch_size):
        indices = indices_list[i].t().reshape(-1)
        indices = torch.unique(indices, sorted = False) 
        order_list.append(indices)
    return order_list
        

        



def smooth_PCG_sp_assembly(K,f,u0,maxit): 
    indices = K.indices()
    # rows, cols = indices[0], indices[1]
    values = K.values()
    n = K.size(0)
    
    # 提取对角线元素构造预条件子M_diag
    diag_mask = (indices[0] == indices[1])
    diag_rows = indices[0][diag_mask]
    diag_values = values[diag_mask]

    diag = torch.zeros(n, dtype=values.dtype, device=K.device)
    diag[diag_rows] = diag_values
    M_diag = 1.0 / diag

    r =  f- torch.sparse.mm(K, u0)
    z = M_diag.unsqueeze(1) * r
    p = z.clone()
    rho_old = torch.sum(r * z,dim=0)

    for _ in range(maxit):
        Ap = torch.sparse.mm(K, p)
        alpha = torch.div(rho_old , torch.sum(p* Ap,dim=0)) 
        u0 = u0 + alpha * p
        u0 = u0 - torch.mean(u0, dim = 0)
        r = r - alpha * Ap
        z_new = M_diag.unsqueeze(1) * r
        rho_new = torch.sum(r*z_new,dim=0)
        beta = rho_new / (rho_old )
        p = z_new + beta * p
        rho_old = rho_new
         
    return u0

# def chebyshev_preconditioner(Ke, node_index, b, eig_min, eig_max, degree=3):
#     """
#     Applies Chebyshev polynomial preconditioning to vector b.
    
#     Args:
#         A (Tensor): Symmetric positive-definite matrix.
#         b (Tensor): Right-hand side vector.
#         eig_min (float): Estimate of the smallest eigenvalue of A.
#         eig_max (float): Estimate of the largest eigenvalue of A.
#         degree (int): Degree of the Chebyshev polynomial.
    
#     Returns:
#         Tensor: Preconditioned vector.
#     """
#     d = (eig_max + eig_min) / 2
#     c = (eig_max - eig_min) / 2

#     x0 = torch.zeros_like(b)
#     x1 = b / d

#     if degree == 1:
#         return x1

#     for k in range(2, degree + 1):
#         alpha = 2 * c / d
#         beta = -1
#         x2 = alpha * scatter_add(torch.matmul(Ke.reshape(24,24).unsqueeze(0), x1[node_index.reshape(-1),:].reshape(-1,24,6)).reshape(-1,6),
#                                     node_index,
#                                     dim=0) + beta * x0 + b / d
#         x0, x1 = x1, x2

#     return x1

# def EBE_PCG_chebyshev(Ke,F,u,node_type,node_index,min_eigen,max_eigen,max_iter): 
#     # node_type 每个节点周围8节点体素情况
#     Ke = Ke.to(torch.float64)
#     F = F.to(torch.float64)
#     u = u.to(torch.float64)

#     node_index=node_index.to(torch.int64).view(-1)
#     node_index = node_index.repeat(3, 1).T
#     # print(torch.max(node_index),torch.min(node_index),u.shape)
#     node_index[:,0] = node_index[:,0]*3
#     node_index[:,1] = node_index[:,1]*3+1
#     node_index[:,2] = node_index[:,2]*3+2
#     node_index = node_index.reshape(-1)

#     u = u.reshape(-1,6,3).transpose(-1,-2).reshape(-1,6)

#     # r (-1,3,6)
#     r = (F.reshape(-1,6) - scatter_add(torch.matmul(Ke.reshape(24,24).unsqueeze(0), 
#                                     u[node_index.reshape(-1),:].reshape(-1,24,6)).reshape(-1,6),node_index,dim=0))
#     z =  chebyshev_preconditioner(Ke, node_index, r, min_eigen, max_eigen)

#     p = z.clone()

#     rsold = torch.sum(r * z,dim=0)

#     for i in range(max_iter):
#         Ap = torch.matmul(Ke.reshape(24,24).unsqueeze(0), p[node_index.reshape(-1),:].reshape(-1,24,6))
#         Ap = scatter_add(Ap.reshape(-1,6),node_index,dim=0)
#         alpha = rsold / (torch.sum(p * Ap,dim=0) ) 
#         u = u + alpha.unsqueeze(0) * p
#         u = (u.reshape(-1,18) - torch.mean(u.reshape(-1,18), dim = 0)).reshape(-1,6)
#         r = r - alpha.unsqueeze(0) * Ap
#         z  =  chebyshev_preconditioner(Ke, node_index, r, min_eigen, max_eigen)
#         rsnew = torch.sum(r * z,dim=0)
#         p = z + (rsnew / rsold) * p
#         rsold = rsnew
#     u = u.reshape(-1,3,6).transpose(-1,-2).reshape(-1,18)
#     return u

def calculate_residual(Ke,F,u,node_index):
    node_index=node_index.to(torch.int64).view(-1)
    node_index = node_index.repeat(3, 1).T * 3
    # print(torch.max(node_index),torch.min(node_index),u.shape)
    node_index[:,1] = node_index[:,1]+1
    node_index[:,2] = node_index[:,2]+2
    node_index = node_index.reshape(-1)
    Ke = Ke.to(torch.float64)
    F = F.to(torch.float64)
    u = u.to(torch.float64)
    u = u.reshape(-1)
    r = (F.reshape(-1) - scatter_add((torch.matmul(Ke.reshape(24,24), 
                                    u[node_index.reshape(-1)].reshape(-1,24).T).T).reshape(-1),
                                    node_index))
    return r.reshape(-1,3)

def calculate_residual_batch(sparse_tensor,batch,Ke,F,batch_size): 
    coo =sparse_tensor.indices.int()
    u   =sparse_tensor.features
    node_index = batch['node_index']
    voxel_batch_id = batch['voxel_batch_id']
    F=F.reshape(-1,3,6)[:,:,0]
    
    # u = u.reshape(-1,6,3).transpose(-1,-2).reshape(-1,18)
    r = torch.zeros_like(u)

    for i in range (0,batch_size):
        mask=coo[:,0]==i
        mask_voxel=voxel_batch_id==i
        r[mask]=calculate_residual(Ke,F[mask,:],u[mask],node_index[mask_voxel])
        
    return r

import torch

def assemble_F_fast_scalar(node_index: torch.Tensor, Fe: torch.Tensor, n_nodes: int, rho: torch.Tensor):
    """
    Thermal / scalar RHS assembly (3 load cases).

    node_index: (E, 8)   每个单元的8个节点全局编号
    Fe:         (8, 3)   单元节点 RHS，标量自由度，3个工况
    rho:        (E,)     每个单元的密度/系数（用于缩放 RHS）
    return:     (n_nodes, 3)
    """
    if Fe.dim() != 2 or Fe.shape[0] != 8:
        raise ValueError(f"Expected Fe shape (8,3), got {tuple(Fe.shape)}")
    if node_index.dim() != 2 or node_index.shape[1] != 8:
        raise ValueError(f"Expected node_index shape (E,8), got {tuple(node_index.shape)}")

    E = node_index.shape[0]
    rho = rho.reshape(E, 1, 1).to(dtype=Fe.dtype, device=Fe.device)  # (E,1,1)

    idx = node_index.reshape(-1).to(torch.int64)  # (E*8,)

    # (E,8,3) scaled RHS -> (E*8,3)
    val = Fe.unsqueeze(0).expand(E, 8, 3) * rho
    val = val.reshape(-1, 3)

    F = Fe.new_zeros((n_nodes, 3))
    F.index_add_(0, idx, val)
    return F

def assemble_F_fast(node_index: torch.Tensor, Fe: torch.Tensor, n_nodes: int, rho):
    """
    node_index: (Etot, 8)  每个单元的8个节点，全局编号（collate已加node_base）
    Fe: (24, 6)
    返回: (n_nodes, 3) —— 对应你后面只用 macrostrain 的第0列
    """
    # 只取 macrostrain=0: Fe -> (8,3,6) -> (8,3)
    # Fe0 = Fe.view(8, 3, 6)[:, :, 0]                # (8,3)

    # E = node_index.shape[0]
    # idx = node_index.reshape(-1)                    # (E*8,)
    # val = Fe0.unsqueeze(0).expand(E, 8, 3).reshape(-1, 3)  # (E*8,3)

    # F = Fe.new_zeros((n_nodes, 3))
    # F.index_add_(0, idx, val)
    # return F
    Fe8 = Fe.view(8, 3, 6)

    E = node_index.shape[0]
    idx = node_index.reshape(-1)  # (E*8,)

    # 把每个单元的 (8,3,6) 复制 E 份 -> (E,8,3,6) -> (E*8,3,6)
    val = Fe8.unsqueeze(0).expand(E, 8, 3, 6) * rho.reshape(-1,1,1,1)
    # val = Fe8.unsqueeze(0).expand(E, 8, 3, 6)
    val = val.reshape(-1, 3, 6)
    F = Fe.new_zeros((n_nodes, 3, 6))
    F.index_add_(0, idx, val)
    return F

def matvec_from_Ke(x, Ke, node_index):
    """
    x: (Ndof, ncol)  这里 ncol=6 在你代码里
    Ke: (24, 24) 或 (1,24,24)（与原始代码一致）
    node_index: (Ne*24,) 全局 DOF 索引（已按 3分量展开）
    返回: A @ x  (Ndof, ncol)
    """
    Xloc = x.index_select(0, node_index).reshape(-1, 24)  # (Ne,24,ncol)
    # print(Ke.unsqueeze(0).shape,Xloc.shape)
    Yloc = torch.matmul(Ke, Xloc.T).T# (Ne,24,ncol)（广播 KeB）
    y = scatter_add(Yloc.reshape(-1), node_index)  # (Ndof, ncol)
    return y

# --- 3) 正确的 Chebyshev 预条件器（Jacobi 基础） ---

def chebyshev_preconditioner(r, matvec, eig_min, eig_max, degree=3):
    """
    y_{k+1} = y_k + w_k
    w_k = ( D^{-1}(r - A y_k) + β w_{k-1} ) / d
    d=(λmax+λmin)/2, c=(λmax-λmin)/2, β=(c/(2d))^2
    r: (Ndof, ncol)
    """
    assert degree >= 1
    d = 0.5 * (eig_max + eig_min)
    c = 0.5 * (eig_max - eig_min)
    beta = (c / (2.0 * d)) ** 2

    y = r / d              
    w_prev = y.clone()     

    for _ in range(1, degree):
        Ay = matvec(y)
        t = (r - Ay)
        w = (t + beta * w_prev) / d
        y = y + w
        w_prev = w
    return y

def EBE_PCG_chebyshev(Ke, F, u, node_type, node_index, min_eigen, max_eigen, max_iter, tol=1e-8):
    # max_iter = 80
    Ke = Ke.to(torch.float64)
    F  = F.to(torch.float64)
    u  = u.to(torch.float64)

    # --- 按你原始逻辑展开 DOF 索引（xyz 三分量展开） ---
    node_index = node_index.to(torch.int64).view(-1)          # (Ne*8*?) -> (Ne*8)
    node_index = node_index.repeat(3, 1).T                    # (Ne*8,3)
    node_index[:,0] = node_index[:,0]*3
    node_index[:,1] = node_index[:,1]*3 + 1
    node_index[:,2] = node_index[:,2]*3 + 2
    node_index = node_index.reshape(-1)                       # (Ne*24,)

    # 变量形状与原始一致
    u = u.reshape(-1, 3).reshape(-1)  # (Ndof,6)
    b = F.reshape(-1, 6)[:,0]                                    # (Ndof,6)
    Ndof = u.shape[0]

    mv = lambda x: matvec_from_Ke(x, Ke.reshape(24, 24), node_index)

    r = b - mv(u)                                             # (Ndof,6)
    z = chebyshev_preconditioner(r, mv, min_eigen, max_eigen, degree=3)
    p = z.clone()
    rz_old = torch.sum(r * z)                               # (6,)

    for it in range(max_iter):
        Ap = mv(p)
        denom = torch.sum(p * Ap)         # (6,)
        alpha = rz_old / denom                                 # (6,)
        u = u + alpha * p
        r = r - alpha * Ap
        # print(torch.linalg.norm(r).item())
        if torch.linalg.norm(r).item() < 3e-5 :
            break
        # print('iter: ',it,'r',torch.linalg.norm(r).item())
        z = chebyshev_preconditioner(r, mv, min_eigen, max_eigen, degree=3)
        rz_new = torch.sum(r * z)
        beta = (rz_new / rz_old)                # (1,6)
        p = z + beta * p
        rz_old = rz_new
    # exit()    # 还原回你的形状
    u = u.reshape(-1, 3)
    return u


def EBE_PCG(Ke,F,u,node_type,node_index,max_iter): 
    # node_type 每个节点周围8节点体素情况
    Ke = Ke.to(torch.float64)
    F = F.to(torch.float64)
    u = u.to(torch.float64)

    diag=torch.zeros(u.shape[0],3,dtype=u.dtype,device=u.device)
    diag_Ke=Ke.reshape(8,3,8,3).transpose(1,2)
    # 编号顺序z-y-x
    # hex=torch.tensor([[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1]],dtype=torch.int,device=voxel.device)
    Ke_index=[6,2,1,5,7,3,0,4]
    for i in range (0,8):
        diag+=node_type[:,i].unsqueeze(1) * (diag_Ke[Ke_index[i],Ke_index[i]].diagonal())
    M_diag = torch.ones_like(1.0 / diag.reshape(-1))
    node_index=node_index.to(torch.int64).view(-1)
    node_index = node_index.repeat(3, 1).T
    # print(torch.max(node_index),torch.min(node_index),u.shape)
    node_index[:,0] = node_index[:,0] * 3
    node_index[:,1] = node_index[:,1] * 3 + 1
    node_index[:,2] = node_index[:,2] * 3 + 2
    node_index = node_index.reshape(-1)
    

    u = torch.zeros_like(u.reshape(-1,6,3).transpose(-1,-2).reshape(-1,6))

    # r (-1,3,6)
    r = (F.reshape(-1,6) - scatter_add(torch.matmul(Ke.reshape(24,24).unsqueeze(0), 
                                    u[node_index.reshape(-1),:].reshape(-1,24,6)).reshape(-1,6),
                                    node_index,
                                    dim=0))
    z = M_diag.unsqueeze(1) * r
    # z = r
    p = z.clone()

    rsold = torch.sum(r * z,dim=0)
    for it in range(max_iter):
        Ap = torch.matmul(Ke.reshape(24,24).unsqueeze(0), p[node_index.reshape(-1),:].reshape(-1,24,6))
        Ap = scatter_add(Ap.reshape(-1,6),node_index,dim=0)
        alpha = rsold / (torch.sum(p * Ap,dim=0) ) 
        u = u + alpha.unsqueeze(0) * p
        u = (u.reshape(-1,18) - torch.mean(u.reshape(-1,18), dim = 0)).reshape(-1,6)
        r = r - alpha.unsqueeze(0) * Ap
        print('iter: ',it,'r',torch.linalg.norm(r).item())
        z_new  =  M_diag.unsqueeze(1) * r
        rsnew = torch.sum(r * z_new,dim=0)
        p = z_new + (rsnew / rsold) * p
        rsold = rsnew
    u = u.reshape(-1,3,6).transpose(-1,-2).reshape(-1,18)
    exit()
    # u = u - torch.mean(u, dim = 0)
    return u



def matvec_from_K_node(x, K_node_matrix,adj_coo, res):
    K_node_matrix = K_node_matrix.reshape(res,res,res,27,3,3)
    adj_u = x[adj_coo[:,0],adj_coo[:,1],adj_coo[:,2],:].view(27,-1,3).transpose(0,1).reshape(res,res,res,27,3)
    valid_u=torch.einsum('xyznji,xyzni->xyzj',K_node_matrix,adj_u)
    return valid_u


def PCG(K_node_matrix,F_node_matrix,u,maxit = 10):
    device = K_node_matrix.device
    res = K_node_matrix.shape[0]
    K_node_matrix = K_node_matrix.reshape(res,res,res,27,3,3)

    xs = torch.arange(res,device=device)
    ys = torch.arange(res,device=device)
    zs = torch.arange(res,device=device)
    X, Y, Z = torch.meshgrid(xs, ys, zs, indexing='ij')
    coo = torch.stack([X, Y, Z], dim=-1).reshape(-1, 3)  # 形状: (n^3, 3)
    
    offsets = torch.tensor([[i,j,k] for i in [-1,0,1] for j in [-1,0,1] for k in [-1,0,1]],device=device)
    adj_coo = (coo.unsqueeze(0) + offsets.view(27,1,3)+ res) % res
    adj_coo = adj_coo.reshape(-1,3)

    
    M_diag = 1.0 / (K_node_matrix[:,:,:,13,:,:].diagonal(dim1=-2, dim2=-1)+1e-12)

    mv_GMG = lambda x: matvec_from_K_node(x, K_node_matrix, adj_coo, res)

    r = F_node_matrix - mv_GMG(u)

    z = M_diag * r
    p = z.clone()
    rsold = torch.sum(r * z)

    for it in range(maxit):
        Ap = mv_GMG(p)
        alpha = rsold / torch.sum(p * Ap) 
        # print('sp', u.shape,p.shape)
        u = u + alpha * p
        r = r - alpha * Ap
        z  =  M_diag * r
        rsnew = torch.sum(r * z)
        p = z + (rsnew / rsold) * p
        rsold = rsnew
    return u
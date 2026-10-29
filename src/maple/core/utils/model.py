import torch
import torch.nn as nn

from maple.core.utils.EBE_GMG import GMGSolver as GMG_3D
from maple.core.utils.EBE_GMG_scalar import GMGSolverScalar as GMG_1D
from maple.core.utils._utils import (
    hexahedron,
    hexahedron_scalar,
    isotropic_elastic_tensor,
)
    
class IHD(nn.Module):
    def __init__(self,cfg,device):
        super(IHD, self).__init__()
        self.cfg=cfg
        self.r=cfg.resolution
        self.resolution = cfg.resolution
        self.C = isotropic_elastic_tensor(1.0, 0.3).to(device)
        self.Ke_3d, self.Fe_3d, self.X0_3d = hexahedron(r=self.resolution, C=self.C)
        self.Ke_3d = self.Ke_3d.to(device)
        self.Fe_3d = self.Fe_3d.to(device)
        self.X0_3d = self.X0_3d.to(device)

        # --- scalar thermal conduction element (8-node, 3 load cases) ---
        Ke1, Fe1, X01 = hexahedron_scalar(r=self.resolution, kappa=torch.as_tensor(1.0, device=device, dtype=torch.float32))
        self.Ke_1d = Ke1.to(device)
        self.Fe_1d = Fe1.to(device)
        self.X0_1d = X01.to(device)

        self.device=device  
        self.gmg_solver_3d=GMG_3D()
        self.gmg_solver_1d=GMG_1D()
    def homogenized_batch_3d(self, U, ke_hard, X0,noda_batch_id, node_index, voxel_batch_id,batch_size):
        volume = 1.
        CH = torch.zeros(batch_size, 6, 6, dtype=U.dtype, device=U.device) 
        for i in range (batch_size):
            mask = noda_batch_id==i 
            mask_voxel = voxel_batch_id == i 
            u_=U[mask].reshape(-1,18).reshape(-1,6,3).transpose(-1,-2)
            node_index_=node_index[mask_voxel].reshape(-1).to(torch.int64)
            u_element=u_[node_index_.reshape(-1),:].reshape(-1,8,3,6).reshape(-1,24,6)
            L = X0-u_element
            CH[i,:,:] = 1 / volume * torch.einsum('bij,bik,bkl-> jl',L,ke_hard,L)
        return CH
    import torch

    def homogenized_batch_thermal(
        self,
        U,                  # (N_nodes_total, C)  scalar temperature field for C load cases
        ke_hard,             # (N_vox_total, 8, 8)
        X0,                  # (N_vox_total, 8, C)
        noda_batch_id,       # (N_nodes_total,)
        node_index,          # (N_vox_total, 8)
        voxel_batch_id,      # (N_vox_total,)
        batch_size
    ):
        volume = 1.0

        # 保证 U 至少是 (N, C)
        if U.dim() == 1:
            U = U[:, None]

        C = U.shape[1]  # 工况数：3D 导热通常是 3（x/y/z 三个单位梯度）
        CH = torch.zeros(batch_size, C, C, dtype=U.dtype, device=U.device)

        for i in range(batch_size):
            mask_node = (noda_batch_id == i)
            mask_vox  = (voxel_batch_id == i)

            # 当前 batch 的节点温度 (n_nodes_i, C)
            u_nodes = U[mask_node]  # (n_nodes_i, C)

            # 当前 batch 的单元节点索引 (n_vox_i, 8)
            node_index_i = node_index[mask_vox].to(torch.int64)  # (n_vox_i, 8)

            # gather 单元 8 节点温度 -> (n_vox_i, 8, C)
            u_elem = u_nodes[node_index_i.reshape(-1), :].reshape(-1, 8, C)

            # 当前 batch 的 ke 与 X0
            ke_i = ke_hard[mask_vox]  # (n_vox_i, 8, 8)

            # X0 can be (8,C) (broadcast) or (N_vox,8,C) (per-voxel)
            if X0.dim() == 2:
                X0_i = X0.unsqueeze(0).expand(ke_i.shape[0], -1, -1)
            else:
                X0_i = X0[mask_vox]

            # L = X0 - u_elem, (n_vox_i, 8, C)
            L = X0_i - u_elem

            # CH = sum_e (L_e^T ke_e L_e) / volume
            # einsum: (e,i,c) (e,i,j) (e,j,d) -> (c,d)
            CH[i] = torch.einsum("eic,eij,ejd->cd", L, ke_i, L) / volume

        return CH


    def TO_homo_3d(self,batch,Ke,F,X0,batch_size):
        device = self.device
        F = F.reshape(-1,3,6)
        coo,feature=batch["coord"].int(),batch["feature"]
        # feature.requires_grad_(True)

        


        E = batch['node_index'].shape[0]
        Ke_ = Ke.unsqueeze(0).expand(E, -1, -1) * batch["rho"].reshape(-1,1,1)
        six_list_u = []
        six_list_r = 0
        for i in range(6):
            u, r = self.gmg_solver_3d.solve(
                                        fine_ke             = Ke_,
                                        global_f            = F[:,:,i],
                                        node_coords_L0      = coo,
                                        fine_elem_coords    = batch['elem_coords'],
                                        fine_topo_indices   = batch['node_index'],
                                        max_iter            = self.cfg.GMG.max_cycle,
                                        smooth_iter         = self.cfg.GMG.smooth_iter,
                                        init_list           = None,
                                        max_levels          = len(self.cfg.GMG.smooth_iter),
                                        dtype               = torch.float32
                                    )
            six_list_u.append(u)  

        out_u = torch.cat(six_list_u, dim=1)


        CH = self.homogenized_batch_3d(out_u, Ke_, X0, batch["coord"][:,0], batch['node_index'],batch['voxel_batch_id'],batch_size)[0]
        return  CH
    def TO_homo_1d(self,batch,Ke,F,X0,batch_size):
        device = self.device
        F = F.reshape(-1,3)
        coo,feature=batch["coord"].int(),batch["feature"]

        E = batch['node_index'].shape[0]
        Ke_ = Ke.unsqueeze(0).expand(E, -1, -1) * batch["rho"].reshape(-1,1,1)
        six_list_u = []
        six_list_r = 0
        for i in range(3):
            u, r = self.gmg_solver_1d.solve(
                                        fine_ke             = Ke_,
                                        global_f            = F[:,i],
                                        node_coords_L0      = coo,
                                        fine_elem_coords    = batch['elem_coords'],
                                        fine_topo_indices   = batch['node_index'],
                                        max_iter            = self.cfg.GMG.max_cycle,
                                        smooth_iter         = self.cfg.GMG.smooth_iter,
                                        init_list           = None,
                                        max_levels          = len(self.cfg.GMG.smooth_iter),
                                        dtype               = torch.float32
                                    )
            six_list_u.append(u)  

        out_u = torch.cat(six_list_u, dim=1)


        CH = self.homogenized_batch_thermal(out_u, Ke_, X0, batch["coord"][:,0], batch['node_index'], batch['voxel_batch_id'], batch_size)[0]
        return  CH

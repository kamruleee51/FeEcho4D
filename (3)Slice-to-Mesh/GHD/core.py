import torch
import torch.nn as nn
import torch.nn.functional as F


from torch_scatter import scatter

from pytorch3d.io import load_obj, load_objs_as_meshes
from pytorch3d.structures import Meshes
from pytorch3d.ops import cot_laplacian
from torch_geometric.utils import to_undirected, add_self_loops, get_laplacian
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle

from scipy.sparse.linalg import eigsh
from scipy.sparse import coo_matrix

from .utils import laplician_symetric, get_mix_laplacian

import trimesh
import numpy as np



class GHD_config:
    def __init__(self, **args):
        self.base_shape_path = None 
        self.num_basis = 7*7+1
        self.device = "cuda:0"
        self.mix_laplacian_tradeoff = {'cotlap': 1.0, 'dislap': 1e-3, 'stdlap': 1e-3}
        self.if_lap_nomalize = True
        self.eign_path = None
        self.__dict__.update(args)
        

    def print(self):
        print("GHD config:")
        for k,v in self.__dict__.items():
            print(k, v)
    #
    def clone(self):
        print("num_basis", self.num_basis)
        print("mix_laplacian_tradeoff", self.mix_laplacian_tradeoff)
        return GHD_config(**self.__dict__)
        



class GHDmesh:

    def __init__(self,  cfg: GHD_config = GHD_config(), base_shape: Meshes = None):
        """
        base_shape: pytorch3d.structures.Meshes
        """
        self.num_basis = cfg.num_basis
        self.cfg = cfg.clone()
        self.device = cfg.device
        if base_shape is None:
            self.base_shape = load_objs_as_meshes([cfg.base_shape_path], device=cfg.device)
            self.cfg.eign_path = None
        else:
            self.base_shape = base_shape

        self.mix_laplacian, self.cot_lap, self.distant_lap, self.stander_lap = get_mix_laplacian(self.base_shape, 
                                                                                                 cfg.mix_laplacian_tradeoff, 
                                                                                                 if_return_scipy=cfg.if_return_scipy,
                                                                                                 if_nomalize=cfg.if_lap_nomalize)
        if cfg.eign_path is not None:
            self.GH_eigval, self.GH_eigvec = torch.load(cfg.eign_path, map_location=cfg.device)
        else:
            self.GH_eigval, self.GH_eigvec = eigsh(self.mix_laplacian, cfg.num_basis, which='SM')
            self.GH_eigval = torch.from_numpy(self.GH_eigval).to(cfg.device).float().unsqueeze(0)
            self.GH_eigvec = torch.from_numpy(self.GH_eigvec).to(cfg.device).float().unsqueeze(0)

        self.reset_affine_param()
        self.reset_GHD_param()

    def GHD_uppdate(self, new_gh_param):
        """
        update the GHD parameters
        """
        new_gh_mesh = self.__class__(self.cfg, self.base_shape)
        new_gh_mesh.GHD_param.data = new_gh_param
        return new_gh_mesh
        


    def reset_affine_param(self):
        """
        reset the affine parameters
        """
        self.R = nn.Parameter(torch.zeros(1,3, device=self.device))
        self.s = nn.Parameter(torch.tensor([1.], device=self.device).unsqueeze(0))
        self.T = nn.Parameter(torch.zeros(1,3, device=self.device))

    def reset_GHD_param(self):
        """
        reset the GHD parameters (1, num_basis, 3)
        """
        self.GHD_param = nn.Parameter(torch.zeros((1, self.num_basis, 3), dtype=torch.float32, device=self.device, requires_grad=True))

    def random_init_ghd(self, std=0.1):
        """
        random initialize the GHD parameters with the given std
        """
        self.GHD_param.data = torch.randn_like(self.GHD_param.data)*std
    
    
    def ghf_to_vertf(self, gh_feature):
        """
        recover the GH coefficients to the GH basis
        GH_feature: (B, num_basis, D) or (num_basis, D)
        return: (B, N_verts, D)
        """
        assert gh_feature.shape[-2] == self.num_basis
        
        try:
            B = gh_feature.shape[-3]
        except:
            B = 1
            gh_feature = gh_feature.unsqueeze(0)

        return self.GH_eigvec.matmul(gh_feature)

        
    
    def vertf_to_ghf(self, vert_feature):
        """
        project the input feature to the GHB eigen space
        input_feature: (B, N_verts, d) or (N_verts, d)
        return: (B, num_basis, d)
        """
        assert vert_feature.shape[-2] == self.base_shape.verts_padded().shape[-2]

        try:
            B = vert_feature.shape[-3]
        except:
            B = 1
            vert_feature = vert_feature.unsqueeze(0)
            
        return self.GH_eigvec.transpose(-1,-2).matmul(vert_feature)
    

    
    def rendering(self, GHD_param=None, Affine_param=None):
        """
        render the mesh by the current GHD parameters
        """
        
        deformation = self.__call__(gh_feature=GHD_param)  # (B, N_verts, 3)

        if Affine_param is None:
            R_matrix = axis_angle_to_matrix(self.R)
            s = self.s
            T = self.T
        else:
            R_matrix = axis_angle_to_matrix(Affine_param[0])
            s = Affine_param[1]
            T = Affine_param[2]

        new_verts = (self.base_shape.verts_padded()+deformation) @ R_matrix.transpose(-1,-2)*s + T
        
        output_shape = self.base_shape.update_padded(new_verts)

        return output_shape
    

    def __call__(self, gh_feature = None):
        """
        retur the feature map on the base shape
        GHD_coefficient: (1, num_basis, F) if None, use the current GHD parameters (1, num_basis, 3)
        """
        if gh_feature is None:
            gh_feature = self.GHD_param # (1, num_basis, 3)
    
        return self.ghf_to_vertf(gh_feature)
    
class Normal_iterative_GHDmesh(GHDmesh):
        
    def __init__(self, cfg: GHD_config = GHD_config(), base_shape: Meshes = None, num_iter=10):
        """
        base_shape: pytorch3d.structures.Meshes
        """
        self.num_iter = num_iter
        super().__init__(cfg, base_shape)
        


    def reset_GHD_param(self):
        """
        reset the GHD parameters (1, num_basis, T)
        """
        self.GHD_param = nn.Parameter(torch.zeros((1, self.num_basis, self.num_iter), dtype=torch.float32, device=self.device, requires_grad=True))

    def rendering(self, GHD_param=None, step_length=1.0):

        if GHD_param is None:
            gh_feature = self.GHD_param # (1, num_basis, T)
        else:
            gh_feature = GHD_param.view(1, self.num_basis, -1)
        
        deformation_lengths = self.ghf_to_vertf(gh_feature)*step_length # (1, N_verts, T)

        current_mesh = self.base_shape.clone()
        for t in range(self.num_iter):
            current_normal = current_mesh.verts_normals_padded() # (1, N_verts, 3)
            deformation = deformation_lengths[:,:,t:t+1]*current_normal
            current_mesh = current_mesh.offset_verts(deformation.view(-1,3))

        R_matrix = axis_angle_to_matrix(self.R) # (1, 3, 3)

        current_mesh = current_mesh.update_padded(current_mesh.verts_padded()@ R_matrix.transpose(-1,-2)*self.s.view(1,1,1) + self.T.view(1,1,3))

        return current_mesh


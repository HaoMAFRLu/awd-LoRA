"""This script is used to analyze pretrained LLaMA models using RPCA.
"""
import torch
import torch.distributed as dist
from loguru import logger

from salad.utils import *
from salad.ialm import fit_torch

LAYER_TYPE = ['self_attn.q_proj', 'self_attn.k_proj', 
              'self_attn.v_proj', 'self_attn.o_proj',
              'mlp.down_proj', 'mlp.up_proj', 'mlp.gate_proj']

class StaticRPCA:
    def __init__(self,
                 model: torch.nn.Module,
                 path_folder: str=None) -> None:
        self.path_folder = path_folder
        self.rank, self.world_size = self._init_distributed()

        if self.rank == 0:
            print(f'Total rank: {self.world_size}')

        torch.cuda.set_device(self.rank % torch.cuda.device_count())
        self.device = torch.device(f'cuda:{self.rank % torch.cuda.device_count()}')
        dev_idx = torch.cuda.current_device()
        props   = torch.cuda.get_device_properties(dev_idx)
        logger.info(f"[Rank {self.rank}] using {props.name}, {props.total_memory / (1024 ** 3):.2f} GiB")       

        # distribute the layers
        self.nr_layers = model.config.num_hidden_layers

        if self.rank == 0:
            all_layers = [f'layers.{i}.{layer_type}' for i in range(self.nr_layers) for layer_type in LAYER_TYPE]
            # split the layers among the ranks
            self.layers_per_rank = [[] for _ in range(self.world_size)]
            for idx, layer_name in enumerate(all_layers):
                self.layers_per_rank[idx % self.world_size].append(layer_name)
        else:
            self.layers_per_rank = None


        self.ddp_model = DDP(model.cuda(), device_ids=[torch.cuda.current_device()])
        dist.broadcast_object_list(self.layers_per_rank, src=0)
        self.assigned_layers = self.layers_per_rank[self.rank]

        
    def _init_distributed(self):
        """Initialize distributed environment"""
        dist.init_process_group(backend='nccl')
        rank = dist.get_rank()
        world = dist.get_world_size()
        return rank, world
    
    def load_LS(self, path_folder: str = None):
        """Load the low-rank and sparse components from the pretrained analysis."""
        LL = {}
        SS = {}
        files = os.listdir(path_folder)
        rank_files = [f for f in files if f.startswith('matrix')]
        for f in rank_files:
            LL_part, SS_part = get_lowspa_layers(os.path.join(path_folder, f))
            for key in LL_part:
                LL[key] = LL_part[key]
                SS[key] = SS_part[key]
        return LL, SS
    
    def recover_LS(self):
        """Recover the low-rank and sparse components for the assigned layers."""
        L_pretrained, S_pretrained = self.load_LS(self.path_folder)
        svs = {}
        SS = {}

        for layer_name in self.assigned_layers:
            L = L_pretrained[layer_name].to(torch.float32).to(self.device)
            S = S_pretrained[layer_name].to(torch.float32).to(self.device)
            X = L + S
            m, n = X.shape

            L_, S_ = fit_torch(X, lambda_ = 1.0 / np.sqrt(max(m, n)),
                                device=self.device, 
                                dtype=torch.float32, 
                                epsilon1=1e-2, 
                                epsilon2=1e-2)
            
            _, sv, _ = torch.linalg.svd(L_, full_matrices=False)
            svs[layer_name] = sv.to('cpu')
            SS[layer_name] = S_.to('cpu')
        
        data = {'svs': svs, 'SS': SS}
        atomic_pickle_dump(data, os.path.join(self.path_folder, f'rpca_LS_rank_{self.rank}.pkl'))

    def recover_X(self):
        """Recover the low-rank and sparse components for the assigned layers."""
        svs = {}
        SS = {}
        for layer_name in self.assigned_layers:
            X = get_weight(self.ddp_model, layer_name).detach().to(torch.float32).to(self.device)
            m, n = X.shape
            L, S = fit_torch(X, lambda_ = 1.0 / np.sqrt(max(m, n)),
                            device=self.device, 
                            dtype=torch.float32, 
                            epsilon1=1e-2, 
                            epsilon2=1e-2)
            
            _, sv, _ = torch.linalg.svd(L, full_matrices=False)
            svs[layer_name] = sv.to('cpu')
            SS[layer_name] = S.to('cpu')
        
        data = {'svs': svs, 'SS': SS}
        atomic_pickle_dump(data, os.path.join(self.path_folder, f'rpca_X_rank_{self.rank}.pkl'))

    def destroy(self):
        dist.destroy_process_group()
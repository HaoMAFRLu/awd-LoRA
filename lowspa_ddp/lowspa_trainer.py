import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import functools
import os
import pickle
from tqdm import tqdm

from lowspa_ddp.layer_solver import ADMMSolver
from lowspa_ddp.utils import *

class LowSpaTrainer():
    def __init__(self, 
                 model: nn.Module,
                 data_loader: torch.utils.data.DataLoader,
                 config: dict):
        """
        Args:
            model: the nn.Module to train
            config: dict specifying:
                - layers: list of layer names to apply SVD
                - solvers: mapping layer_name -> solver class or params
                - gpu_map: optional mapping layer_name -> gpu id
                - training: dict with optimizer, lr, num_epochs
        """
        self.model = model
        self.config = config
        self.model_type = config.get('model_type', 'CNN')
        self.data_loader = data_loader

        self.rank, self.world_size = self._init_distributed()
        torch.cuda.set_device(self.rank % torch.cuda.device_count())

        # Wrap model in DDP
        self.model.cuda()
        self.ddp_model = DDP(self.model, device_ids=[torch.cuda.current_device()])

        self.training_setup(self.model_type,
                            config.get('optimizer'),
                            config.get('scheduler'))
        
        # get all the names of the model layers
        self.names_model_layers = self.get_model_layer_names(self.model)
        # get specified layers in the config
        self.cfg_layers = self.get_cfg_layers(self.config, self.names_model_layers)
        # assign layers to different GPUs
        self.assigned_layers = self.assign_layers(self.cfg_layers, self.rank, self.world_size)
        # get all the weights for the assigned layers
        XX = {entry: self.get_weight(self.ddp_model, 'module.'+entry) for entry in self.assigned_layers}
        
        self.LL = {entry: torch.zeros_like(XX[entry]) for entry in self.assigned_layers}
        self.SS = {entry: torch.zeros_like(XX[entry]) for entry in self.assigned_layers}
        self.YY = {entry: torch.zeros_like(XX[entry]) for entry in self.assigned_layers}
        
        self.layer_info = {entry: {
            'loss': [],
            'rank': [],
            'nonzero': [],
            'total_rank': [],
            'total_elements': []
        } for entry in self.assigned_layers}
        self.layer_info['loss'] = []
        self.layer_info['loss1'] = []
        self.layer_info['loss2'] = []

        # initialize the ADMM solvers
        self.ADMM_solvers = []
        for entry in self.cfg_layers:
            name = entry['name']
            params = entry['params']
            solver = ADMMSolver(name, params, XX[name])
            solver.layer_gpu_map = self.rank if name in self.assigned_layers else -1
            self.ADMM_solvers.append(solver)

    @staticmethod
    def get_name_and_params(_params: dict):
        """
        Extract name and parameters from a config dict.
        """
        if not isinstance(_params, dict):
            raise ValueError("Expected a dictionary for config data")
        
        name = _params.get('name')
        if not name:
            raise ValueError("Config must contain a 'name' key")
        
        params = _params.get('params', {})
        return name, params
    
    def training_setup(self,
                       model_type: str,
                       params_optimizer: dict,
                       params_scheduler: dict):
        self.loss_fn = get_loss_fn(model_type)
        self.optimizer = get_optimizer(*self.get_name_and_params(params_optimizer), self.ddp_model)
        self.scheduler = get_scheduler(*self.get_name_and_params(params_scheduler), self.optimizer)

    @staticmethod
    def get_weight(model: torch.nn.Module, 
                   layer_name: str) -> torch.Tensor:
        module, attr = layer_name.rsplit('.', 1)
        sub = functools.reduce(getattr, module.split('.'), model)
        return getattr(sub, attr) 

    @staticmethod
    def get_cfg_layers(config: dict,
                       names_model_layers) -> list:
        """Extract layer names from config"""
        layers = config.get('layers')

        if not isinstance(layers, list):
            raise ValueError("Config 'layers' must be a list of layer names")

        for entry in layers:
            name = entry.get('name')        
            if name not in names_model_layers:
                raise KeyError(f"Layer {name} not found in model")

        return layers
    
    @staticmethod
    def assign_layers(layers: dict, 
                      rank: int,
                      world_size: int) -> dict: 
        """
        Assign layers to GPUs in a round-robin fashion. 
        Args:
            layers: list of layer names
            rank: current process rank
            world_size: total number of processes
        Returns:
            dict mapping layer names to GPU ids
        """ 
        assigned_layers = [
            entry['name'] for idx, entry in enumerate(layers)
            if idx % world_size == rank
        ]
        return assigned_layers

    @staticmethod
    def get_model_layer_names(model: torch.nn.Module):
        """
        Recursively collect all layer names in the model.
        Returns a list of parameter names.
        """
        return {name for name, _ in model.named_parameters()} 

    def _init_distributed(self):
        """Initialize distributed environment"""
        dist.init_process_group(backend='nccl')
        rank = dist.get_rank()
        world = dist.get_world_size()
        return rank, world

    def get_penalty_loss(self):
        """User-defined loss; can be overridden or passed via config."""
        loss = 0
        for solver in self.ADMM_solvers:
            loss += solver.get_loss_term()
        return loss

    def gather_results(self, local_results):
        """Gather dicts from all ranks to rank 0"""
        gathered = [None] * self.world_size
        dist.all_gather_object(gathered, local_results)
        if self.rank == 0:
            for p in gathered:
                for layer_name, data in p.items():
                    self.LL[layer_name] = data['L']
                    self.SS[layer_name] = data['S']
                    self.YY[layer_name] = data['Y']
                    self.layer_info[layer_name]['loss'].append(data['avg_loss'])
                    self.layer_info[layer_name]['rank'].append(data['nr_rank'])
                    self.layer_info[layer_name]['nonzero'].append(data['nr_nonzero'])
                    self.layer_info[layer_name]['total_rank'].append(data['nr_total_rank'])
                    self.layer_info[layer_name]['total_elements'].append(data['nr_elements'])
 
    def broadcast_results(self, results):
        """Broadcast a dict of factors from rank 0 to all ranks"""
        # factors is a dict or None
        brd = [results]
        dist.broadcast_object_list(brd, src=0)
        return brd[0]

    def single_step_train(self, data, target):
        """
        Single training step for the model.
        Args:
            data: Input data batch
            target: Target labels
        """
        
        data = data.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        self.optimizer.zero_grad()
        output = self.ddp_model(data)
        
        loss1 = self.loss_fn(output, target)
        loss2 = self.get_penalty_loss()
        loss = loss1 + loss2

        loss.backward()
        self.optimizer.step()

        return loss.item(), loss1.item(), loss2.item()
    
    def save_results(self, path_folder):
        """
        Save the results of the training.
        Args:
            path_folder: Path to save the results
        """
        torch.save(self.ddp_model.state_dict(), os.path.join(path_folder, 'model.pth'))
        data = {
            'LL': self.LL,
            'SS': self.SS,
            'YY': self.YY,
            'layer_info': self.layer_info,
        }
        with open(os.path.join(path_folder, 'results.pkl'), 'wb') as f:
            pickle.dump(data, f)

    def update_local_state(self):
        """
        Update local state with the broadcasted results.
        This is called after gathering results from all ranks.
        """
        for solver in self.ADMM_solvers:
            if solver.layer_gpu_map != self.rank:
                solver.L = self.LL[solver.layer_name]
                solver.S = self.SS[solver.layer_name]
                solver.Y = self.YY[solver.layer_name]

    def get_local_results(self):
        """
        Get local results for the current rank.
        Returns:
            dict with layer names and their corresponding results.
        """
        local_results = {}
        for solver in self.ADMM_solvers:
            if solver.layer_gpu_map == self.rank:
                solver.run()
                local_results[solver.layer_name] = solver.results
        return local_results

    def solvers_reset(self):
        """
        Reset all solvers for a new training epoch.
        """
        for solver in self.ADMM_solvers:
            solver.reset()

    def print_info(self,
                   epoch: int,
                   total_epochs: int,
                   layer_info: dict,
                   lr: float):
        """
        Print training information for the current epoch.
        Args:
            epoch: Current epoch number
            total_epochs: Total number of epochs
            layer_info: Dictionary containing layer statistics
        """
        losses = {'loss': layer_info['loss'][-1],
                  'loss1': layer_info['loss1'][-1],
                  'loss2': layer_info['loss2'][-1]}
        
        layer_stats = [{'name': entry['name'],
                        'loss': layer_info[entry['name']]['loss'][-1],
                        'non_zero': layer_info[entry['name']]['nonzero'][-1],
                        'rank': layer_info[entry['name']]['rank'][-1],
                        'total_rank': layer_info[entry['name']]['total_rank'][-1],
                        'total_elements': layer_info[entry['name']]['total_elements'][-1]} for entry in self.cfg_layers]
        
        print_epoch(epoch, total_epochs, lr, losses, layer_stats)

    def train(self, 
              num_epochs: int, 
              path_folder: str=None):
        
        self.ddp_model.train()
        
        for epoch in range(num_epochs):
            ep_loss, ep_loss1, ep_loss2 = 0.0, 0.0, 0.0
            self.solvers_reset()

            # training over data
            for data, target in tqdm(self.data_loader):
                loss, loss1, loss2 = self.single_step_train(data, target)
                ep_loss += loss
                ep_loss1 += loss1
                ep_loss2 += loss2

            # average losses
            ep_loss /= len(self.data_loader)
            ep_loss1 /= len(self.data_loader)
            ep_loss2 /= len(self.data_loader)    
            self.layer_info['loss'].append(ep_loss)
            self.layer_info['loss1'].append(ep_loss1)
            self.layer_info['loss2'].append(ep_loss2)
            
            self.scheduler.step()
            # run ADMM solvers and get local results
            local_results = self.get_local_results()
            # gather results from all ranks
            self.gather_results(local_results)
            # broadcast results to all ranks
            self.LL, self.SS, self.YY = self.broadcast_results((self.LL, self.SS, self.YY))
            # update local state
            self.update_local_state()
            # print and save results
            if self.rank == 0:
                self.print_info(epoch, num_epochs, self.layer_info, self.scheduler.get_last_lr()[0])
                if path_folder is not None:
                    self.save_results(path_folder)

        dist.destroy_process_group()


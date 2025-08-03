import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import os, socket
import pickle
from tqdm import tqdm
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

from lowspa_ddp.utils import *

class Trainer():
    def __init__(self, 
                 model: nn.Module,
                 model_type: str,
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
        self.model_type = model_type
        self.data_loader = data_loader

        self.loss_list = []

        self.rank, self.world_size = self._init_distributed()
        torch.cuda.set_device(self.rank % torch.cuda.device_count())
        self.device = torch.device(f'cuda:{self.rank % torch.cuda.device_count()}')

        # print device info
        dev_idx = torch.cuda.current_device()
        props   = torch.cuda.get_device_properties(dev_idx)
        print(f"[Rank {self.rank}] using {props.name}, {props.total_memory / (1024 ** 3):.2f} GiB")       

        # Wrap model in DDP
        self.model.cuda()
        self.ddp_model = DDP(self.model, device_ids=[torch.cuda.current_device()])

        self.training_setup(self.model_type,
                            config.get('optimizer'),
                            config.get('scheduler'))

    def training_setup(self,
                       model_type: str,
                       params_optimizer: dict,
                       params_scheduler: dict):
        self.loss_fn = get_loss_fn(model_type)
        self.optimizer = get_optimizer(*self.get_name_and_params(params_optimizer), self.ddp_model)
        # self.scheduler = get_scheduler(*self.get_name_and_params(params_scheduler), self.optimizer)
        warmup_iters = 2000
        max_iters    = 600_000
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[
                LinearLR(self.optimizer, start_factor=1e-8, end_factor=1.0, total_iters=warmup_iters),
                CosineAnnealingLR(self.optimizer, T_max=max_iters - warmup_iters, eta_min=6e-5),
            ],
            milestones=[warmup_iters],
        )

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
    
    def _init_distributed(self):
        """Initialize distributed environment"""
        dist.init_process_group(backend='nccl')
        rank = dist.get_rank()
        world = dist.get_world_size()
        return rank, world

    def get_global_loss(self, log_loss):
        """
        Get the global loss across all ranks.
        Args:
            loss: local loss tensor
        Returns:
            global loss value
        """
        with torch.no_grad():
            dist.all_reduce(log_loss, op=dist.ReduceOp.SUM)
            log_loss = log_loss / self.world_size
        return log_loss.item()

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
        
        loss = self.loss_fn(output, target)
        loss.backward()
        self.optimizer.step()

        global_loss_mean = self.get_global_loss(loss.detach())
        return global_loss_mean
    
    def save_results(self, path_folder):
        """
        Save the results of the training.
        Args:
            path_folder: Path to save the results
        """
        torch.save(self.ddp_model.state_dict(), os.path.join(path_folder, 'model.pth'))
        with open(os.path.join(path_folder, 'results.pkl'), 'wb') as f:
            pickle.dump(self.loss_list, f)

    def print_info(self,
                   epoch: int,
                   total_epochs: int,
                   loss: float,
                   lr: float):
        """
        Print training information for the current epoch.
        Args:
            epoch: Current epoch number
            total_epochs: Total number of epochs
            layer_info: Dictionary containing layer statistics
        """
        header = (f"Epoch {epoch}/{total_epochs} | "
                 f"Lr: {lr:.6f} | "
                 f"Loss: {loss:.6f} | ")
        print(header)

    def train(self, 
              num_epochs: int, 
              path_folder: str=None):

        def is_main_process():
            if self.rank == 0:
                return tqdm(self.data_loader)
            else:
                return self.data_loader
            
        self.ddp_model.train()
        
        for epoch in range(num_epochs):
            ep_loss = 0.0
            # important for DDP randomness
            if self.model_type == 'GPT':
                self.data_loader.set_epoch(epoch)
            # training over data
            for data, target in is_main_process():
                loss = self.single_step_train(data, target)
                ep_loss += loss
                # update learning rate
                self.scheduler.step()
            # average losses
            ep_loss /= len(self.data_loader)  
            self.loss_list.append(ep_loss)
            

            if self.rank == 0:
                self.print_info(epoch, num_epochs, self.loss_list[-1], self.scheduler.get_last_lr()[0])
                if path_folder is not None:
                    self.save_results(path_folder)

        dist.destroy_process_group()


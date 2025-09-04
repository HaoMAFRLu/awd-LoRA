import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import datasets
import datasets.distributed
from loguru import logger

from salad.salad_solver import SALAD
from salad.utils import *
from salad.register import *

class SALADTrainer():
    def __init__(self, 
                 model: nn.Module,
                 data: datasets.Dataset,
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

        self.num_warmup_steps = 40
        self.num_total_iters = config.get('num_total_iters', 1000)
        self.num_freq = config.get('num_freq', 1)
        self.is_clip = config.get('is_clip', 1.0)
        self.max_length = config.get('max_length', 256)
        self.num_workers = config.get('num_workers', 4)

        self.rank, self.world_size = self._init_distributed()

        # fk the error 429!!!!!!
        hf_login_once()

        torch.cuda.set_device(self.rank % torch.cuda.device_count())
        self.device = torch.device(f'cuda:{self.rank % torch.cuda.device_count()}')
        if self.rank == 0:
            print_setting(config)

        # self.batch_size = config.get('batch_size', 32)
        self.batch_size = int(config.get('batch_size', 32)/self.world_size) + 1

        # print device info
        dev_idx = torch.cuda.current_device()
        props   = torch.cuda.get_device_properties(dev_idx)
        logger.info(f"[Rank {self.rank}] using {props.name}, {props.total_memory / (1024 ** 3):.2f} GiB")       

        # Wrap model in DDP
        self.model.cuda()
        # get all the names of the model layers
        self.names_model_layers = get_linear_layers_name(self.model)
        # get specified layers in the config
        self.cfg_layers = self.get_cfg_layers(self.config, self.names_model_layers)

        for entry in self.cfg_layers:
            name = entry['name']
            params = entry['params']
            W = self.get_weight(self.model, 'model.'+name)
            rate_rank = params.get('rate_rank', 0.5)
            # truncate the rank of X
            U, s, Vt = torch.linalg.svd(W, full_matrices=False)
            idx = int(len(s) * rate_rank)
            _W = (U[:, :idx] * s[:idx]) @ Vt[:idx, :]
            with torch.no_grad():
                W.copy_(_W.to(W.dtype))

        self.ddp_model = DDP(self.model, device_ids=[torch.cuda.current_device()])

        data = datasets.distributed.split_dataset_by_node(data, rank=self.rank, world_size=self.world_size)

        tokenizer = get_tokenizer(self.max_length)
        dataset = PreprocessedIterableDataset(data, tokenizer, 
                                              batch_size=self.batch_size, 
                                              max_length=self.max_length)
        self.dataloader = torch.utils.data.DataLoader(dataset, 
                                                      batch_size=None, 
                                                      num_workers=self.num_workers)
        self.pad_idx = tokenizer.pad_token_id

        self.optimizer = get_optimizer(*self.get_name_and_params(config['optimizer']), self.ddp_model)
        self.lr_scheduler = get_scheduler(self.optimizer,
                                        scheduler_type=config['scheduler']['name'],
                                        num_training_steps=self.num_total_iters,
                                        warmup_steps=config['scheduler']['params'].get('warmup_steps', 0),
                                        min_lr_ratio=config['scheduler']['params'].get('min_lr_ratio', 0.0))
        # warmup the model
        # self.warmup(self.num_warmup_steps)
        
        
        # assign layers to different GPUs
        self.assigned_layers, self.owner_map = self.assign_layers(self.cfg_layers, self.rank, self.world_size)
        self.per_owner_names, self.owner_sizes = self.build_per_owner_static(self.ddp_model, self.owner_map, self.world_size)

        self.layer_info = {entry['name']: {
            'loss': [],
            'rank': [],
            'alpha': [],
            'dalpha': [],
            'beta': [],
            'dbeta': [],
            'rho': [],
            'rate_decay': [],
            'nonzero': [],
            'total_rank': [],
            'total_elements': []
        } for entry in self.cfg_layers}
        self.layer_info['loss'] = []
        self.layer_info['loss1'] = []
        self.layer_info['loss2'] = []
        self.layer_info['num_tokens'] = []

        # initialize the ADMM solvers
        self.ADMM_solvers = []
        for entry in self.cfg_layers:
            name = entry['name']
            params = entry['params']
            solver = SALAD(name, 
                           params, 
                           self.get_weight(self.ddp_model, 'module.model.'+name), 
                           len(self.cfg_layers),
                           is_full=name in self.assigned_layers)
            solver.layer_gpu_map = self.rank if name in self.assigned_layers else -1
            self.ADMM_solvers.append(solver)
        
        # after initialization, sync the initial weights
        self.LL = {entry['name']: torch.zeros_like(self.get_weight(self.ddp_model, 'module.model.'+entry['name']), device='cpu') for entry in self.cfg_layers}
        self.SS = {entry['name']: torch.zeros_like(self.get_weight(self.ddp_model, 'module.model.'+entry['name']), device='cpu') for entry in self.cfg_layers}
        self.YY = {entry['name']: torch.zeros_like(self.get_weight(self.ddp_model, 'module.model.'+entry['name']), device='cpu') for entry in self.cfg_layers}
        
        self.sync_weights()

    @staticmethod    
    def canon(name: str) -> str:
        if name.startswith('module.'): name = name[7:]
        if name.startswith('model.'):  name = name[6:]
        if name.endswith('.weight'):   name = name[:-7]
        return name

    def build_per_owner_static(self, ddp_model, owner_map, world_size):
        per_owner_names = {r: [] for r in range(world_size)}

        for n, item in owner_map.items():
            per_owner_names[item].append(n)

        param_dict = dict(ddp_model.named_parameters())
        owner_sizes = {r: sum(param_dict['module.model.'+n+'.weight'].numel() for n in per_owner_names[r]) for r in range(world_size)}
        return per_owner_names, owner_sizes

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
    
    @staticmethod
    def get_weight(model: torch.nn.Module, layer_name: str) -> torch.Tensor:
        sub = model.get_submodule(layer_name)
        return sub.weight

    @staticmethod
    def get_cfg_layers(config: dict,
                       names_model_layers) -> list:
        """Extract layer names from config"""
        layers = config.get('layers')

        if not isinstance(layers, list):
            raise ValueError("Config 'layers' must be a list of layer names")

        for entry in layers:
            name = entry.get('name')        
            if 'model.'+name not in names_model_layers:
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
        owner_map = {
            entry['name']: idx % world_size for idx, entry in enumerate(layers)
        }
        return assigned_layers, owner_map
    
    def _init_distributed(self):
        """Initialize distributed environment"""
        dist.init_process_group(backend='nccl')
        rank = dist.get_rank()
        world = dist.get_world_size()
        return rank, world

    def get_penalty_loss(self):
        """User-defined loss; can be overridden or passed via config."""
        loss = 0.0
        for solver in self.ADMM_solvers:
            if solver.layer_gpu_map == self.rank:
                loss += solver.get_loss_term(solver.L, solver.S, solver.Y)
            else:
                loss += solver.get_loss_term(self.LL[solver.layer_name].to(self.device),
                                             self.SS[solver.layer_name].to(self.device),
                                             self.YY[solver.layer_name].to(self.device))
        return loss

    def get_loss2_info(self):
        # Z = rho*(X - S + (1/rho)*Y)
        loss2_info = {}
        total_loss = 0.0
        for solver in self.ADMM_solvers:
            if solver.layer_gpu_map == self.rank:
                _name, _Z, loss = solver.get_loss_info(solver.L, solver.S, solver.Y)
                loss2_info[_name] = _Z
                total_loss += loss
            else:
                _, _, loss = solver.get_loss_info(self.LL[solver.layer_name].to(self.device),
                                                  self.SS[solver.layer_name].to(self.device),
                                                  self.YY[solver.layer_name].to(self.device))
                total_loss += loss
        return loss2_info, total_loss/len(self.ADMM_solvers)
    
    def sync_layer_info(self):
        """
        Synchronize weights across all ranks.
        This is called after the optimizer step.
        """
        local_results = self.get_local_results()
        self.gather_layer_info(local_results)

    def gather_layer_info(self, local_results):
        """
        """
        gathered = [None] * self.world_size
        dist.all_gather_object(gathered, local_results)
        if self.rank == 0:
            for p in gathered:
                for layer_name, data in p.items():
                    self.layer_info[layer_name]['alpha'].append(data['alpha'])
                    self.layer_info[layer_name]['beta'].append(data['beta'])
                    self.layer_info[layer_name]['dalpha'].append(data['dalpha'])
                    self.layer_info[layer_name]['dbeta'].append(data['dbeta'])
                    self.layer_info[layer_name]['rho'].append(data['rho'])
                    self.layer_info[layer_name]['rate_decay'].append(data['rate_decay'])
                    self.layer_info[layer_name]['loss'].append(data['avg_loss'])
                    self.layer_info[layer_name]['rank'].append(data['nr_rank'])
                    self.layer_info[layer_name]['nonzero'].append(data['nr_nonzero'])
                    self.layer_info[layer_name]['total_rank'].append(data['nr_total_rank'])
                    self.layer_info[layer_name]['total_elements'].append(data['nr_elements'])



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
                    self.layer_info[layer_name]['alpha'].append(data['alpha'])
                    self.layer_info[layer_name]['beta'].append(data['beta'])
                    self.layer_info[layer_name]['dalpha'].append(data['dalpha'])
                    self.layer_info[layer_name]['dbeta'].append(data['dbeta'])
                    self.layer_info[layer_name]['rho'].append(data['rho'])
                    self.layer_info[layer_name]['rate_decay'].append(data['rate_decay'])
                    self.layer_info[layer_name]['loss'].append(data['avg_loss'])
                    self.layer_info[layer_name]['rank'].append(data['nr_rank'])
                    self.layer_info[layer_name]['nonzero'].append(data['nr_nonzero'])
                    self.layer_info[layer_name]['total_rank'].append(data['nr_total_rank'])
                    self.layer_info[layer_name]['total_elements'].append(data['nr_elements'])
    
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

    # def single_step_train(self, batch, labels):
    #     """
    #     Single training step for the model.
    #     Args:
    #         data: Input data batch
    #         target: Target labels
    #     """
    #     self.optimizer.zero_grad(set_to_none=True)
    #     loss1 = self.ddp_model(**batch, labels=labels).loss
    #     loss2 = self.get_penalty_loss()
    #     loss = loss1 + loss2
    #     loss.backward()

    #     if self.is_clip > 0:
    #         # Clip gradients to avoid exploding gradients
    #         # This is a common practice in training large models
    #         torch.nn.utils.clip_grad_norm_(self.ddp_model.parameters(), max_norm=self.is_clip)

    #     self.optimizer.step()
    #     self.lr_scheduler.step()

    #     global_loss_mean = self.get_global_loss(loss.detach())
    #     global_loss_mean1 = self.get_global_loss(loss1.detach())
    #     global_loss_mean2 = self.get_global_loss(loss2.detach())
    #     return global_loss_mean, global_loss_mean1, global_loss_mean2
    
    def _resolve_name(self, name, param_dict):
        if name in param_dict: return name
        if f"module.{name}" in param_dict: return f"module.{name}"
        if name.startswith("module.") and name[7:] in param_dict: return name[7:]
        return None

    @torch.no_grad()
    def broadcast_params(self, ddp_model):
        param_dict = dict(ddp_model.named_parameters())

        names_me = self.per_owner_names[self.rank]
        sz_me    = self.owner_sizes[self.rank]

        flat_me = torch.empty(sz_me, device=self.device)
        off = 0
        for n in names_me:
            p = param_dict['module.model.'+n+'.weight'].data.view(-1)
            flat_me[off:off+p.numel()] = p
            off += p.numel()

        for r in range(self.world_size):
            sz = self.owner_sizes[r]

            if r == self.rank:
                buf = flat_me
                dist.broadcast(buf, src=r)
            else:
                buf = torch.empty(sz, device=self.device)
                dist.broadcast(buf, src=r)
                off = 0
                for n in self.per_owner_names[r]:
                    p = param_dict['module.model.'+n+'.weight']
                    k = p.numel()
                    p.data.view(-1).copy_(buf[off:off+k])
                    off += k

    def single_step_train(self, batch, labels):
        """
        Single training step for the model.
        Args:
            data: Input data batch
            target: Target labels
        """
        self.optimizer.zero_grad(set_to_none=True)
        # only use optimizer to update the track loss
        loss1 = self.ddp_model(**batch, labels=labels).loss
        loss2_info, loss2 = self.get_loss2_info()
        # loss = loss1 + loss2
        loss1.backward()

        if self.is_clip > 0:
            # Clip gradients to avoid exploding gradients
            # This is a common practice in training large models
            torch.nn.utils.clip_grad_norm_(self.ddp_model.parameters(), max_norm=self.is_clip)

        self.optimizer.step()

        param_dict = dict(self.ddp_model.named_parameters())
        with torch.no_grad():
            eta = self.optimizer.param_groups[0]['lr']
            for name, Z in loss2_info.items():
                param_dict['module.model.'+name+'.weight'].data -= eta * Z
 
        self.lr_scheduler.step()

        self.broadcast_params(self.ddp_model)

        loss = loss1 + loss2
        global_loss_mean = self.get_global_loss(loss.detach())
        global_loss_mean1 = self.get_global_loss(loss1.detach())
        global_loss_mean2 = self.get_global_loss(loss2.detach())
        return global_loss_mean, global_loss_mean1, global_loss_mean2
    
    def save_results(self, path_folder):
        os.makedirs(path_folder, exist_ok=True)

        # 1) save the model
        state = getattr(self.ddp_model, "module", self.ddp_model).state_dict()
        atomic_torch_save(state, os.path.join(path_folder, "model.pth"))

        # save the data
        MATRIX = {
            'LL': self.LL, 'SS': self.SS, 'YY': self.YY
        }
        atomic_pickle_dump(MATRIX, os.path.join(path_folder, "matrix.pkl"))
        atomic_pickle_dump(self.layer_info, os.path.join(path_folder, "layer_info.pkl"))

    def get_local_single_weight(self,
                                target: str='L'):
        """
        Get local single weight for the current rank.
        Returns:
            dict with layer names and their corresponding weights.
        """
        local_weights = {}
        for solver in self.ADMM_solvers:
            if solver.layer_gpu_map == self.rank:
                if target == 'L':
                    local_weights[solver.layer_name] = solver.L.to('cpu')
                elif target == 'S':
                    local_weights[solver.layer_name] = solver.S.to('cpu')
                elif target == 'Y':
                    local_weights[solver.layer_name] = solver.Y.to('cpu')
        return local_weights

    def gather_single_weight(self, local_weights, target: str='L'):
        """Gather dicts from all ranks to rank 0"""
        gathered = [None] * self.world_size
        dist.all_gather_object(gathered, local_weights)
        if self.rank == 0:
            for p in gathered:
                for layer_name, data in p.items():
                    if target == 'L':
                        self.LL[layer_name] = data  # L
                    elif target == 'S':
                        self.SS[layer_name] = data  # S
                    elif target == 'Y':
                        self.YY[layer_name] = data  # Y
    
    def broadcast_single_weight(self, target: str='L'):
        """
        Broadcast weights from rank 0 to all ranks.
        Returns:
            L, S, Y: broadcasted weights
        """
        if target == 'L':
            brd = self.LL
        elif target == 'S':
            brd = self.SS
        elif target == 'Y':
            brd = self.YY
        dist.broadcast_object_list([brd], src=0)
        return brd

    def get_local_weights(self):
        """
        Get local weights for the current rank.
        Returns:
            dict with layer names and their corresponding weights.
        """
        local_weights = {}
        for solver in self.ADMM_solvers:
            if solver.layer_gpu_map == self.rank:
                local_weights[solver.layer_name] = (solver.L.to('cpu'), solver.S.to('cpu'), solver.Y.to('cpu'))
        return local_weights
    
    def gather_weights(self, local_weights):
        """Gather dicts from all ranks to rank 0"""
        gathered = [None] * self.world_size
        dist.all_gather_object(gathered, local_weights)
        if self.rank == 0:
            for p in gathered:
                for layer_name, data in p.items():
                    self.LL[layer_name] = data[0]  # L
                    self.SS[layer_name] = data[1]  # S
                    self.YY[layer_name] = data[2]  # Y

    def sync_weights(self):
        """
        Synchronize weights across all ranks.
        This is called after the optimizer step.
        """
        local_weights = self.get_local_weights()
        self.gather_weights(local_weights)
        self.LL, self.SS, self.YY = self.broadcast_weights()

    def broadcast_weights(self):
        """
        Broadcast weights from rank 0 to all ranks.
        Returns:
            L, S, Y: broadcasted weights
        """
        brd = [self.LL, self.SS, self.YY]
        dist.broadcast_object_list(brd, src=0)
        return brd[0], brd[1], brd[2]
    
    def sync_results(self):
        """
        Synchronize results across all ranks.
        This is called after the optimizer step.
        """
        local_results = self.get_local_results()
        self.gather_results(local_results)
        self.LL, self.SS, self.YY = self.broadcast_weights()
        
    def get_local_results(self):
        """
        Get local results for the current rank.
        Returns:
            dict with layer names and their corresponding results.
        """
        local_results = {}
        for solver in self.ADMM_solvers:
            if solver.layer_gpu_map == self.rank:
                local_results[solver.layer_name] = solver.results
        return local_results
    
    def update_ADMM_single_step(self, target: str='L'):
        """ Update the low-rank component L for all layers.
        """
        for solver in self.ADMM_solvers:
            if solver.layer_gpu_map == self.rank:
                if target == 'L':
                    solver.update_L()
                elif target == 'S':
                    solver.update_S()
                elif target == 'Y':
                    solver.update_Y()
                elif target == 'alpha':
                    solver.update_alpha()
                elif target == 'beta':
                    solver.update_beta()
                elif target == 'save':
                    solver.cal_results()
    
    def sync_single_weight(self, target: str='L'):
        """ Synchronize the low-rank component L across all ranks.
        """
        local_weights = self.get_local_single_weight(target=target)
        self.gather_single_weight(local_weights, target=target)
        if target == 'L':
            self.LL = self.broadcast_single_weight(target=target)
        elif target == 'S':
            self.SS = self.broadcast_single_weight(target=target)
        elif target == 'Y':
            self.YY = self.broadcast_single_weight(target=target)

    def update_ADMM_rho(self): 
        """ Update the penalty parameter rho for all layers.
        """
        for solver in self.ADMM_solvers:
            solver.update_rho()

    def run_ADMM_solvers(self):
        """ Run ADMM solvers for the current rank.
        """
        for solver in self.ADMM_solvers:
            if solver.layer_gpu_map == self.rank:
                solver.run()

    def solvers_reset(self):
        """
        Reset all solvers for a new training epoch.
        """
        for solver in self.ADMM_solvers:
            solver.reset()

    def print_info(self,
                   epoch: int,
                   total_epochs: int,
                   num_freq: int,
                   loss: float,
                   loss1: float,
                   loss2: float,
                   acc_num_tokens: int,
                   layer_info: dict,
                   lr: float):
        """
        Print training information for the current epoch.
        Args:
            epoch: Current epoch number
            total_epochs: Total number of epochs
            layer_info: Dictionary containing layer statistics
        """
        losses = {'loss': loss,
                  'loss1': loss1,
                  'loss2': loss2}
        
        layer_stats = [{'name': entry['name'],
                        'loss': layer_info[entry['name']]['loss'][-1],
                        'alpha': layer_info[entry['name']]['alpha'][-1],
                        'beta': layer_info[entry['name']]['beta'][-1],
                        'dalpha': layer_info[entry['name']]['dalpha'][-1],
                        'dbeta': layer_info[entry['name']]['dbeta'][-1],
                        'rho': layer_info[entry['name']]['rho'][-1],
                        'rate_decay': layer_info[entry['name']]['rate_decay'][-1],
                        'non_zero': layer_info[entry['name']]['nonzero'][-1],
                        'rank': layer_info[entry['name']]['rank'][-1],
                        'total_rank': layer_info[entry['name']]['total_rank'][-1],
                        'total_elements': layer_info[entry['name']]['total_elements'][-1]} for entry in self.cfg_layers]
        
        print_epoch(epoch, total_epochs, num_freq, lr, acc_num_tokens, losses, layer_stats)

    def warmup(self, num_warmup_steps: int = 30):
        """
        Perform a warmup step to initialize the model and solvers.
        This is useful for distributed training to ensure all processes are synchronized.
        """
        num_step = 0
        self.ddp_model.train()
        for batch in self.dataloader:
            num_step += 1
            if num_step > num_warmup_steps:
                break
            
            batch = {k: v.to(self.device) for k, v in batch.items()}
            labels = batch["input_ids"].clone()
            labels[labels == self.pad_idx] = -100
            self.optimizer.zero_grad()
            loss = self.ddp_model(**batch, labels=labels).loss
            loss.backward()

            if self.is_clip > 0:
                # Clip gradients to avoid exploding gradients
                # This is a common practice in training large models
                torch.nn.utils.clip_grad_norm_(self.ddp_model.parameters(), max_norm=self.is_clip)

            self.optimizer.step()
            self.lr_scheduler.step()

    def train(self, path_folder: str=None):
        # switch to train mode        
        self.ddp_model.train()
        num_it = 0
        num_epochs = self.num_total_iters // self.num_freq
        epoch = 0
        ep_loss, ep_loss1, ep_loss2 = 0.0, 0.0, 0.0
        num_tokens = 0
        acc_num_tokens = 0

        for batch_idx, batch in enumerate(self.dataloader):
            num_it += 1
            # terminate training if reached max iterations
            if num_it > self.num_total_iters:
                logger.info(f"Reached max number of update steps (f{self.num_total_iters}). Stopping training.")
                print(f"Rank {self.rank} stopping training.")
                break
            

            batch = {k: v.to(self.device) for k, v in batch.items()}
            labels = batch["input_ids"].clone()
            labels[labels == self.pad_idx] = -100 
            # do one step update: gt = nabla l(X) + rho(X - L - S - Y/rho)
            loss, loss1, loss2 = self.single_step_train(batch, labels=labels)

            # calculate the constants
            num_tokens = (batch['input_ids'].numel() - torch.sum(batch['input_ids'] == self.pad_idx).item()) * self.world_size
            self.layer_info['loss'].append(loss)
            self.layer_info['loss1'].append(loss1)
            self.layer_info['loss2'].append(loss2)
            self.layer_info['num_tokens'].append(num_tokens)
            
            ep_loss += loss
            ep_loss1 += loss1
            ep_loss2 += loss2
            acc_num_tokens += num_tokens

            # now we update S and Y at each iteration
            self.update_ADMM_single_step(target='S')
            self.sync_single_weight(target='S')

            if num_it % self.num_freq == 0:
                # run admm solvers
                epoch += 1

                self.update_ADMM_rho()
                self.update_ADMM_single_step(target='L')
                self.sync_single_weight(target='L')
                self.update_ADMM_single_step(target='alpha')
                self.update_ADMM_single_step(target='beta')
                self.update_ADMM_single_step(target='save')
                self.sync_layer_info()
                self.solvers_reset()

                # self.run_ADMM_solvers()
                # self.sync_results()

                # average losses
                ep_loss /= self.num_freq
                ep_loss1 /= self.num_freq
                ep_loss2 /= self.num_freq    
                
                # print and save results
                if self.rank == 0:
                    self.print_info(epoch, 
                                    num_epochs,
                                    self.num_freq,
                                    ep_loss,
                                    ep_loss1,
                                    ep_loss2, 
                                    acc_num_tokens, 
                                    self.layer_info, 
                                    self.lr_scheduler.get_last_lr()[0])
                    if path_folder is not None:
                        self.save_results(path_folder)
                
                ep_loss, ep_loss1, ep_loss2 = 0.0, 0.0, 0.0
            
            self.update_ADMM_single_step(target='Y')
            self.sync_single_weight(target='Y')

        dist.destroy_process_group()


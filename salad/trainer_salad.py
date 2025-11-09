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
from salad.simple_timer import SimpleTimer

class SALADTrainer():
    def __init__(self, 
                 model: nn.Module,
                 data: datasets.Dataset,
                 config: dict,
                 rank: int=0,
                 world_size: int=0) -> None:
        """
        Args:
            model: the nn.Module to train
            config: dict specifying:
                - layers: list of layer names to apply SVD
                - solvers: mapping layer_name -> solver class or params
                - gpu_map: optional mapping layer_name -> gpu id
                - training: dict with optimizer, lr, num_epochs
        """
        # for debug
        # torch.set_printoptions(precision=8)

        self.model = model
        self.config = config

        self.rank = rank
        self.world_size = world_size

        self.num_warmup_steps = 40
        self.num_total_iters = config.get('num_total_iters', 1000)

        self.num_freq = config.get('num_freq', 1)
        self.is_clip = config.get('is_clip', 1.0)
        self.max_length = config.get('max_length', 256)
        self.num_workers = config.get('num_workers', 4)
        self.gradient= config.get('gradient', 'coupled')  # or 'decoupled'
        self.is_asyn = config.get('is_asyn', False)
        self.is_init = config.get('is_init', False)
        self.is_wandb = config.get('is_wandb', False)
        self.is_monitor = config.get('is_monitor', False)
        self.save_interval = config.get('save_interval', 50)

        # self.rank, self.world_size = self._init_distributed()

        if self.rank == 0:
            # self.path_folder = path_folder
            print(f'Total rank: {self.world_size}')
        # else:
        #     self.path_folder = None

        # broadcast the path folder to all ranks
        # path_folder = [self.path_folder]
        # dist.broadcast_object_list(path_folder, src=0)
        # self.path_folder = path_folder[0]

        self.timers = {
            "train": SimpleTimer("train"),
            "S": SimpleTimer("S"),
            "L": SimpleTimer("L"),
            "Y": SimpleTimer("Y"),
            "sync": SimpleTimer("sync"),
            "save": SimpleTimer("save"),
        }

        if self.is_wandb and self.rank == 0:
            import wandb
            wandb.login(key=os.getenv("WANDB_API_KEY"), relogin=False)
            self.run_wandb = wandb.init(project="SALAD_"+self.config['name'], 
                                        entity="hao-ma-eth-z-rich", 
                                        config=self.config)
            
        # fk the error 429!!!!!!ls -l
        hf_login_once()

        # torch.cuda.set_device(self.rank % torch.cuda.device_count())
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

        if self.is_init:
            for entry in self.cfg_layers:
                name = entry['name']
                params = entry['params']
                W = get_weight(self.model, name)
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
            'alpha_mode': [],
            'beta_mode': [],
            'alpha': [],
            'dalpha': [],
            'beta': [],
            'dbeta': [],
            'rho': [],
            'rate_decay_alpha': [],
            'rate_decay_beta': [],
            'nonzero': [],
            'total_rank': [],
            'total_elements': []
        } for entry in self.cfg_layers}
        self.layer_info['avg_loss'] = []
        self.layer_info['avg_loss_penalty'] = []
        self.layer_info['avg_diff'] = []
        self.layer_info['num_tokens'] = []

        # initialize the ADMM solvers
        self.ADMM_solvers = []
        for entry in self.cfg_layers:
            name = entry['name']
            params = entry['params']
            solver = SALAD(name, 
                           params, 
                           get_weight(self.ddp_model, name), 
                           len(self.cfg_layers),
                           is_full=name in self.assigned_layers)
            solver.layer_gpu_map = self.rank if name in self.assigned_layers else -1
            self.ADMM_solvers.append(solver)
        
        # after initialization, sync the initial weights
        # self.LL = {entry['name']: torch.zeros_like(self.get_weight(self.ddp_model, entry['name']), device='cpu') for entry in self.cfg_layers}
        # self.SS = {entry['name']: torch.zeros_like(self.get_weight(self.ddp_model, entry['name']), device='cpu') for entry in self.cfg_layers}
        # self.YY = {entry['name']: torch.zeros_like(self.get_weight(self.ddp_model, entry['name']), device='cpu') for entry in self.cfg_layers}
        # self.sync_weights()
        if self.rank == 0:
            global_layer_names = sorted({s.layer_name for s in self.ADMM_solvers})
        else:
            global_layer_names = None

        global_layer_names = [global_layer_names]  # broadcast_object_list 接受列表
        dist.broadcast_object_list(global_layer_names, src=0)
        global_layer_names = global_layer_names[0]
        self.name2idx = {n: i for i, n in enumerate(global_layer_names)}
        
        for solver in self.ADMM_solvers:
            solver.layer_idx = self.name2idx[solver.layer_name]
            solver.init_T(len(global_layer_names), K=12)

        self.LL = {}
        self.SS = {}
        self.YY = {}

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
        owner_sizes = {
            r: sum(get_param_tensor(param_dict, n, "weight").numel() for n in per_owner_names[r])
            for r in range(world_size)
        }
        # owner_sizes = {r: sum(param_dict['module.model.'+n+'.weight'].numel() for n in per_owner_names[r]) for r in range(world_size)}
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
    def get_cfg_layers(config: dict,
                       names_model_layers) -> list:
        """Extract layer names from config"""
        layers = config.get('layers')

        if not isinstance(layers, list):
            raise ValueError("Config 'layers' must be a list of layer names")

        for entry in layers:
            name = entry.get('name')        
            if name not in names_model_layers and f"model.{name}" not in names_model_layers:
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
    
    # def _init_distributed(self):
    #     """Initialize distributed environment"""
    #     dist.init_process_group(backend='nccl')
    #     rank = dist.get_rank()
    #     world = dist.get_world_size()
    #     return rank, world

    def get_diff_per_rank(self) -> dict:
        """Get the difference X - L - S for each layer."""
        diff = 0.0
        for solver in self.ADMM_solvers:
            if solver.layer_gpu_map == self.rank:
                diff += solver.get_diff(solver.L, solver.S, solver.Y)
            # else:
            #     diff += solver.get_diff(self.LL[solver.layer_name].to(self.device),
            #                             self.SS[solver.layer_name].to(self.device),
            #                             self.YY[solver.layer_name].to(self.device))
        return diff

    def get_gradient_per_layer(self) -> dict:
        """Get the gradient term for each layer."""
        gradient_per_layer = {}
        for solver in self.ADMM_solvers:
            if solver.layer_gpu_map == self.rank:
                Z = solver.get_gradient(solver.X_with_grad.detach(), solver.L, solver.S, solver.Y, solver.rho)
                gradient_per_layer[solver.layer_name] = Z
        return gradient_per_layer

    def single_step_train(self, batch, labels, gradient: str='coupled'):
        # reset the gradient
        self.optimizer.zero_grad(set_to_none=True)
        # calculate the loss of the neural network
        loss = self.ddp_model(**batch, labels=labels).loss
        # get the loss for each layer, (X - L - S)
        # update ema_r and ema_s for updating rho
        diff_per_rank = self.get_diff_per_rank()
        dist.all_reduce(diff_per_rank, op=dist.ReduceOp.SUM)
        global_avg_diff = diff_per_rank.item() / len(self.cfg_layers)
        # calculate the penalty loss of each layer
        # X with gradient -> rho/2 * (X - L - S + Y/rho)^2
        # only used for coupled gradient
        loss_penalty = self.get_penalty_loss()
        # get the closed-form gradient for each layer, rho * (X - L -S + Y/rho)
        # used only for decoupled gradient
        gradient_per_layer = self.get_gradient_per_layer()     

        if gradient == 'decoupled':
            loss.backward()
        elif gradient == 'coupled':
            loss_total = loss + loss_penalty
            loss_total.backward()

        if self.is_clip > 0:
            # Clip gradients to avoid exploding gradients
            # This is a common practice in training large models
            torch.nn.utils.clip_grad_norm_(self.ddp_model.parameters(), max_norm=self.is_clip)

        self.optimizer.step()

        if gradient == 'decoupled':
            param_dict = dict(self.ddp_model.named_parameters())
            with torch.no_grad():
                eta = self.optimizer.param_groups[0]['lr']
                for name, Z in gradient_per_layer.items():
                    param_dict['module.model.'+name+'.weight'].data -= eta * Z
            # broadcast the updated weights
            self.broadcast_params(self.ddp_model)
        
        self.lr_scheduler.step()

        # broadcast the neural network loss
        global_avg_loss = self.get_global_loss(loss.detach())
        # broadcast the penalty loss
        global_avg_loss_penalty = self.get_global_loss(loss_penalty.detach())
        # broadcast the avg_diff
        # global_avg_diff = self.get_global_loss(avg_diff.detach())
        return global_avg_loss, global_avg_loss_penalty, global_avg_diff

    def get_penalty_loss(self):
        """User-defined loss; can be overridden or passed via config."""
        loss = 0.0
        for solver in self.ADMM_solvers:
            if solver.layer_gpu_map == self.rank:
                loss += self.world_size * solver.get_penalty(solver.L, solver.S, solver.Y)
            # else:
            #     loss += solver.get_penalty(self.LL[solver.layer_name].to(self.device),
            #                                self.SS[solver.layer_name].to(self.device),
            #                                self.YY[solver.layer_name].to(self.device))
        return loss

    def sync_layer_info(self):
        """
        Synchronize weights across all ranks.
        This is called after the optimizer step.
        """
        T = self.get_local_results()
        dist.all_reduce(T, op=dist.ReduceOp.SUM)
        if self.rank == 0:
            self.gather_layer_info(T)

    def gather_layer_info(self, T):
        """
        """
        if self.rank == 0:
            for name, i in self.name2idx.items():
                row = T[i]
                info = self.layer_info[name]
                info['alpha_mode'].append(self.ADMM_solvers[0].alpha_solver.mode)
                info['beta_mode'].append(self.ADMM_solvers[0].beta_solver.mode)
                info['alpha'].append(row[0].item())
                info['beta'].append(row[1].item())
                info['dalpha'].append(row[2].item())
                info['dbeta'].append(row[3].item())
                info['rho'].append(row[4].item())
                info['rate_decay_alpha'].append(row[5].item())
                info['rate_decay_beta'].append(row[6].item())
                info['loss'].append(row[7].item())
                info['rank'].append(int(row[8].item()))
                info['nonzero'].append(int(row[9].item()))
                info['total_rank'].append(int(row[10].item()))
                info['total_elements'].append(int(row[11].item()))

    # def gather_results(self, local_results):
    #     """Gather dicts from all ranks to rank 0"""
    #     gathered = [None] * self.world_size
    #     dist.all_gather_object(gathered, local_results)
    #     if self.rank == 0:
    #         for p in gathered:
    #             for layer_name, data in p.items():
    #                 self.LL[layer_name] = data['L'].to('cpu')
    #                 self.SS[layer_name] = data['S'].to('cpu')
    #                 self.YY[layer_name] = data['Y'].to('cpu')
    #                 self.layer_info[layer_name]['alpha'].append(data['alpha'])
    #                 self.layer_info[layer_name]['beta'].append(data['beta'])
    #                 self.layer_info[layer_name]['dalpha'].append(data['dalpha'])
    #                 self.layer_info[layer_name]['dbeta'].append(data['dbeta'])
    #                 self.layer_info[layer_name]['rho'].append(data['rho'])
    #                 self.layer_info[layer_name]['rate_decay'].append(data['rate_decay'])
    #                 self.layer_info[layer_name]['loss'].append(data['avg_loss'])
    #                 self.layer_info[layer_name]['rank'].append(data['nr_rank'])
    #                 self.layer_info[layer_name]['nonzero'].append(data['nr_nonzero'])
    #                 self.layer_info[layer_name]['total_rank'].append(data['nr_total_rank'])
    #                 self.layer_info[layer_name]['total_elements'].append(data['nr_elements'])
    
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
 
    def save_results(self, path_folder):
        if self.rank == 0:
            os.makedirs(path_folder, exist_ok=True)
            # 1) save the model, only rank 0
            state = getattr(self.ddp_model, "module", self.ddp_model).state_dict()
            atomic_torch_save(state, os.path.join(path_folder, "model.pth"))
            # save the layer_info
            atomic_pickle_dump(self.layer_info, os.path.join(path_folder, "layer_info.pkl"))

        LL = {}
        SS = {}
        YY = {}
        for solver in self.ADMM_solvers:
            if solver.layer_gpu_map == self.rank:
               LL[solver.layer_name] = solver.L.to('cpu')
               SS[solver.layer_name] = solver.S.to('cpu')
               YY[solver.layer_name] = solver.Y.to('cpu')         
        
        # save the data
        MATRIX = {
            'LL': LL, 'SS': SS, 'YY': YY
        }

        atomic_pickle_dump(MATRIX, os.path.join(path_folder, 'matrix_rank'+str(self.rank)+'.pkl'))

    # def get_local_single_weight(self,
    #                             target: str='L'):
    #     """
    #     Get local single weight for the current rank.
    #     Returns:
    #         dict with layer names and their corresponding weights.
    #     """
    #     local_weights = {}
    #     for solver in self.ADMM_solvers:
    #         if solver.layer_gpu_map == self.rank:
    #             if target == 'L':
    #                 local_weights[solver.layer_name] = solver.L.to('cpu')
    #             elif target == 'S':
    #                 local_weights[solver.layer_name] = solver.S.to('cpu')
    #             elif target == 'Y':
    #                 local_weights[solver.layer_name] = solver.Y.to('cpu')
    #     return local_weights

    # def gather_single_weight(self, local_weights, target: str='L'):
    #     """Gather dicts from all ranks to rank 0"""
    #     gathered = [None] * self.world_size
    #     dist.all_gather_object(gathered, local_weights)
    #     if self.rank == 0:
    #         for p in gathered:
    #             for layer_name, data in p.items():
    #                 if target == 'L':
    #                     self.LL[layer_name] = data.to('cpu')  # L
    #                 elif target == 'S':
    #                     self.SS[layer_name] = data.to('cpu')  # S
    #                 elif target == 'Y':
    #                     self.YY[layer_name] = data.to('cpu')  # Y
    
    # def broadcast_single_weight(self, target: str='L'):
    #     """
    #     Broadcast weights from rank 0 to all ranks.
    #     Returns:
    #         L, S, Y: broadcasted weights
    #     """
    #     if target == 'L':
    #         brd = self.LL
    #     elif target == 'S':
    #         brd = self.SS
    #     elif target == 'Y':
    #         brd = self.YY
    #     dist.broadcast_object_list([brd], src=0)
    #     return brd

    # def get_local_weights(self):
    #     """
    #     Get local weights for the current rank.
    #     Returns:
    #         dict with layer names and their corresponding weights.
    #     """
    #     local_weights = {}
    #     for solver in self.ADMM_solvers:
    #         if solver.layer_gpu_map == self.rank:
    #             local_weights[solver.layer_name] = (solver.L.to('cpu'), solver.S.to('cpu'), solver.Y.to('cpu'))
    #     return local_weights
    
    # def gather_weights(self, local_weights):
    #     """Gather dicts from all ranks to rank 0"""
    #     gathered = None
    #     if self.rank == 0:
    #         gathered = [None] * self.world_size

    #     dist.gather_object(local_weights, gathered, dst=0)

    #     if self.rank == 0:
    #         for p in gathered:
    #             for layer_name, data in p.items():
    #                 self.LL[layer_name] = data[0].to("cpu")  # L
    #                 self.SS[layer_name] = data[1].to("cpu")  # S
    #                 self.YY[layer_name] = data[2].to("cpu")  # Y

    # def sync_weights(self):
    #     """
    #     Synchronize weights across all ranks.
    #     This is called after the optimizer step.
    #     """
    #     local_results = {}
    #     for solver in self.ADMM_solvers:
    #         if solver.layer_gpu_map == self.rank:
    #             local_results[solver.layer_name] = (solver.L, solver.S, solver.Y)
    #     self.gather_weights(local_results)

    # def broadcast_weights(self):
    #     """
    #     Broadcast weights from rank 0 to all ranks.
    #     Returns:
    #         L, S, Y: broadcasted weights
    #     """
    #     brd = [self.LL, 
    #            self.SS, 
    #            self.YY]
    #     dist.broadcast_object_list(brd, src=0)
    #     return brd[0], brd[1], brd[2]
    
    # def sync_results(self):
    #     """
    #     Synchronize results across all ranks.
    #     This is called after the optimizer step.
    #     """
    #     local_results = self.get_local_results()
    #     self.gather_results(local_results)
    #     self.LL, self.SS, self.YY = self.broadcast_weights()
        
    def get_local_results(self):
        """
        Get local results for the current rank.
        Returns:
            dict with layer names and their corresponding results.
        """
        T = 0
        for solver in self.ADMM_solvers:
            if solver.layer_gpu_map == self.rank:
                T += solver.T
        return T
    
    def update_ADMM_single_step(self, target: str='L'):
        """ Update the low-rank component L for all layers.
        """
        for solver in self.ADMM_solvers:
            if solver.layer_gpu_map == self.rank:
                # if solver.layer_name == 'layers.0.mlp.gate_proj':
                #     print('here')
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
                    # save the results
                    solver.cal_results()
                elif target == 'weight':
                    # update the weights
                    solver.cal_weights()

    # def sync_single_weight(self, target: str='L'):
    #     """ Synchronize the low-rank component L across all ranks.
    #     """
    #     local_weights = self.get_local_single_weight(target=target)
    #     self.gather_single_weight(local_weights, target=target)
    #     if target == 'L':
    #         self.LL = self.broadcast_single_weight(target=target)
    #     elif target == 'S':
    #         self.SS = self.broadcast_single_weight(target=target)
    #     elif target == 'Y':
    #         self.YY = self.broadcast_single_weight(target=target)

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
                   loss_penalty: float,
                   loss_diff: float,
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
        losses = {'avg_loss': loss,
                  'avg_loss_penalty': loss_penalty,
                  'avg_diff': loss_diff}
        
        layer_stats = [{'name': entry['name'],
                        'loss': layer_info[entry['name']]['loss'][-1],
                        'alpha_mode': layer_info[entry['name']]['alpha_mode'][-1],
                        'beta_mode': layer_info[entry['name']]['beta_mode'][-1],
                        'alpha': layer_info[entry['name']]['alpha'][-1],
                        'beta': layer_info[entry['name']]['beta'][-1],
                        'dalpha': layer_info[entry['name']]['dalpha'][-1],
                        'dbeta': layer_info[entry['name']]['dbeta'][-1],
                        'rho': layer_info[entry['name']]['rho'][-1],
                        'rate_decay_alpha': layer_info[entry['name']]['rate_decay_alpha'][-1],
                        'rate_decay_beta': layer_info[entry['name']]['rate_decay_beta'][-1],
                        'non_zero': layer_info[entry['name']]['nonzero'][-1],
                        'rank': layer_info[entry['name']]['rank'][-1],
                        'total_rank': layer_info[entry['name']]['total_rank'][-1],
                        'total_elements': layer_info[entry['name']]['total_elements'][-1]} for entry in self.cfg_layers]
        
        print_epoch(epoch, total_epochs, num_freq, lr, acc_num_tokens, losses, layer_stats)
        if self.is_wandb and self.rank == 0:
            print_wandb(self.run_wandb, 
                        epoch=epoch, 
                        total_epochs=total_epochs, 
                        num_freq=num_freq, 
                        lr=lr, num_tokens=acc_num_tokens, losses=losses, layer_stats=layer_stats)

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
        # if path_folder is None:
        #     path_folder = self.path_folder
            
        self.ddp_model.train()
        num_it = 0
        num_epochs = self.num_total_iters // self.num_freq
        epoch = 0
        ep_loss, ep_penalty, ep_diff = 0.0, 0.0, 0.0
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
            # do one step update
            with self.timers['train']:
                avg_loss, avg_loss_penalty, avg_diff = self.single_step_train(batch, labels, gradient=self.gradient)

            # calculate the constants
            num_tokens = (batch['input_ids'].numel() - torch.sum(batch['input_ids'] == self.pad_idx).item()) * self.world_size
            self.layer_info['avg_loss'].append(avg_loss)
            self.layer_info['avg_loss_penalty'].append(avg_loss_penalty)
            self.layer_info['avg_diff'].append(avg_diff)
            self.layer_info['num_tokens'].append(num_tokens)
            
            ep_loss += avg_loss
            ep_penalty += avg_loss_penalty
            ep_diff += avg_diff
            acc_num_tokens += num_tokens

            # now we update S and Y at each iteration
            # asynchronous update for 
            if num_it % self.num_freq == 0:
                # run admm solvers
                epoch += 1

                self.update_ADMM_single_step(target='beta')
                with self.timers['S']:
                    self.update_ADMM_single_step(target='S')
                self.update_ADMM_rho()
                
                with self.timers['L']:
                    self.update_ADMM_single_step(target='L')
                self.update_ADMM_single_step(target='alpha')
               
                with self.timers['Y']:
                    self.update_ADMM_single_step(target='Y')

                with self.timers['sync']:
                    # self.sync_single_weight(target='S')
                    # self.sync_single_weight(target='L')
                    # self.sync_single_weight(target='Y')
                    # self.sync_all_weights()
                    pass
                
                    self.update_ADMM_single_step(target='save')
                    self.sync_layer_info()

                self.solvers_reset()

                # self.run_ADMM_solvers()
                # self.sync_results()

                # average losses
                ep_loss /= self.num_freq
                ep_penalty /= self.num_freq
                ep_diff /= self.num_freq    
                
                # print and save 
                with self.timers['save']:
                    if path_folder is not None and epoch % self.save_interval == 0:
                        # self.update_ADMM_single_step(target='weight')
                        # self.sync_weights()
                        self.save_results(path_folder)

                if self.rank == 0:
                    self.print_info(epoch, 
                                    num_epochs,
                                    self.num_freq,
                                    ep_loss,
                                    ep_penalty,
                                    ep_diff, 
                                    acc_num_tokens, 
                                    self.layer_info, 
                                    self.lr_scheduler.get_last_lr()[0])
                        
                    if self.is_monitor:
                        print(f'Train: {self.timers["train"].total:.1f}s | Avg Train: {self.timers["train"].avg():.1f}s | S: {self.timers["S"].total:.1f}s | L: {self.timers["L"].total:.1f}s | Y: {self.timers["Y"].total:.1f}s | Sync: {self.timers["sync"].total:.1f}s | Save: {self.timers["save"].total:.1f}s')
                        for key in self.timers:
                            self.timers[key].reset()

                ep_loss, ep_penalty, ep_diff = 0.0, 0.0, 0.0
            
            else:
                if self.is_asyn:
                    pass
                    # self.update_ADMM_single_step(target='beta')
                    
                    # self.update_ADMM_single_step(target='S')
                    # self.update_ADMM_single_step(target='Y')

                    # self.sync_single_weight(target='S')
                    # self.sync_single_weight(target='Y')

        dist.destroy_process_group()
        if self.is_wandb and self.rank == 0:
            self.run_wandb.finish()


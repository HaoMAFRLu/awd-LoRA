import os
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from loguru import logger
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from salad.register import get_scheduler
from salad.salad_solver import SALAD
from salad.simple_timer import SimpleTimer
from salad.utils import (
    atomic_pickle_dump,
    atomic_torch_save,
    get_linear_layers_name,
    get_optimizer,
    get_param_tensor,
    get_weight,
    print_epoch,
    print_setting,
    print_wandb,
)
from salaad_vision.distillation import dino_feature_mse


class SALADTrainer:
    def __init__(self, 
                 model: nn.Module,
                 data: DataLoader,
                 config: dict,
                 rank: int=0,
                 world_size: int=0,
                 folder_name: str=None,
                 teacher_model: Optional[nn.Module]=None) -> None:
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
        self.teacher_model = teacher_model
        self.config = config

        self.rank = rank
        self.world_size = world_size

        self.num_total_iters = config.get('num_total_iters', 1000)

        self.num_freq = config.get('num_freq', 1)
        self.is_clip = config.get('is_clip', 1.0)
        self.gradient = config.get('gradient', 'coupled')  # or 'decoupled'
        self.is_wandb = config.get('is_wandb', False)
        self.is_monitor = config.get('is_monitor', False)
        self.save_interval = config.get('save_interval', 50)
        if (
            not isinstance(self.save_interval, int)
            or isinstance(self.save_interval, bool)
            or self.save_interval <= 0
        ):
            raise ValueError("save_interval must be a positive integer")

        self.training_mode = config.get('training_mode', 'salad')
        if self.training_mode not in {'salad', 'vanilla'}:
            raise ValueError(
                "training_mode must be 'salad' or 'vanilla', "
                f"got {self.training_mode!r}"
            )

        if self.rank == 0:
            print(f'Total rank: {self.world_size}')

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
            self.run_wandb = wandb.init(project="SALAAD_VISION",
                                        entity="hao-ma-eth-z-rich", 
                                        config=self.config,
                                        name=folder_name)

        self.device = torch.device(f'cuda:{self.rank % torch.cuda.device_count()}')
        precision = config.get('precision', 'bfloat16')
        if precision == 'bfloat16':
            self.compute_dtype = torch.bfloat16
        elif precision == 'float32':
            self.compute_dtype = torch.float32
        else:
            raise ValueError(
                "precision must be 'bfloat16' or 'float32', "
                f"got {precision!r}"
            )
        self.use_bfloat16_autocast = (
            config.get('model_type') == 'dino_vitb8'
            and self.compute_dtype == torch.bfloat16
        )
        if self.rank == 0:
            print_setting(config)

        # print device info
        dev_idx = torch.cuda.current_device()
        props   = torch.cuda.get_device_properties(dev_idx)
        logger.info(f"[Rank {self.rank}] using {props.name}, {props.total_memory / (1024 ** 3):.2f} GiB")       

        # The frozen teacher is local to each rank. Only the trainable student
        # is wrapped in DDP and managed by the optimizer/SALAAD solver.
        if self.teacher_model is not None:
            if self.teacher_model.training or any(
                parameter.requires_grad
                for parameter in self.teacher_model.parameters()
            ):
                raise ValueError("teacher_model must be frozen in eval mode")
            self.teacher_model.to(self.device, dtype=self.compute_dtype)

        # DINO uses FP32 master parameters and optimizer state while autocast
        # keeps the forward pass on BF16 Tensor Cores.
        student_dtype = (
            torch.float32
            if self.use_bfloat16_autocast
            else self.compute_dtype
        )
        self.model.to(self.device, dtype=student_dtype)
        logger.info(
            f"[Rank {self.rank}] student master dtype={student_dtype}, "
            f"forward dtype={self.compute_dtype}"
        )
        if self.training_mode == 'salad':
            names_model_layers = get_linear_layers_name(self.model)
            self.cfg_layers = self.get_cfg_layers(self.config, names_model_layers)
        else:
            self.cfg_layers = []

        if config.get('is_init', False):
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

        self.ddp_model = DDP(self.model,
                             device_ids=[torch.cuda.current_device()])

        self.dataloader = data

        self.optimizer = get_optimizer(*self.get_name_and_params(config['optimizer']), self.ddp_model)
        scheduler_steps = config['scheduler']['params'].get(
            'total_steps',
            self.num_total_iters,
        )
        self.lr_scheduler = get_scheduler(self.optimizer,
                                        scheduler_type=config['scheduler']['name'],
                                        num_training_steps=scheduler_steps,
                                        warmup_steps=config['scheduler']['params'].get('warmup_steps', 0),
                                        min_lr_ratio=config['scheduler']['params'].get('min_lr_ratio', 0.0))
        if self.training_mode == 'salad':  # only do the admm for the salad training
            assigned_layers, owner_map = self.assign_layers(
                self.cfg_layers,
                self.rank,
                self.world_size,
            )
            self.per_owner_names, self.owner_sizes = self.build_per_owner_static(
                self.ddp_model,
                owner_map,
                self.world_size,
            )

            # initialize the ADMM solvers
            self.ADMM_solvers = []
            for entry in self.cfg_layers:
                name = entry['name']
                params = entry['params']
                solver = SALAD(name, 
                            params, 
                            get_weight(self.ddp_model, name), 
                            len(self.cfg_layers),
                            is_full=name in assigned_layers)
                solver.layer_gpu_map = self.rank if name in assigned_layers else -1
                self.ADMM_solvers.append(solver)

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
        self.layer_info['avg_cls_loss'] = []
        self.layer_info['avg_patch_loss'] = []
        self.layer_info['avg_loss_penalty'] = []
        self.layer_info['avg_diff'] = []
        self.layer_info['num_images'] = []

    @staticmethod
    def build_per_owner_static(ddp_model, owner_map, world_size):
        per_owner_names = {r: [] for r in range(world_size)}

        for n, item in owner_map.items():
            per_owner_names[item].append(n)

        param_dict = dict(ddp_model.named_parameters())
        owner_sizes = {
            r: sum(get_param_tensor(param_dict, n, "weight").numel() for n in per_owner_names[r])
            for r in range(world_size)
        }
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

    def get_diff_per_rank(self) -> dict:
        """Get the difference X - L - S for each layer."""
        diff = 0.0
        for solver in self.ADMM_solvers:
            if solver.layer_gpu_map == self.rank:
                diff += solver.get_diff(solver.L, solver.S, solver.Y)
        return diff

    def get_gradient_per_layer(self) -> dict:
        """Get the gradient term for each layer."""
        gradient_per_layer = {}
        for solver in self.ADMM_solvers:
            if solver.layer_gpu_map == self.rank:
                Z = solver.get_gradient(solver.X_with_grad.detach(), solver.L, solver.S, solver.Y, solver.rho)
                gradient_per_layer[solver.layer_name] = Z
        return gradient_per_layer

    def single_step_train(self, images, gradient: str='coupled'):
        if self.teacher_model is None:
            raise RuntimeError("DINO feature distillation requires a teacher model")

        self.optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=images.device.type,
            dtype=torch.bfloat16,
            enabled=getattr(self, 'use_bfloat16_autocast', False),
        ):
            with torch.no_grad():
                teacher_features = self.teacher_model(images)
            student_features = self.ddp_model(images)
        distillation_loss = dino_feature_mse(
            student_features,
            teacher_features,
        )
        distillation_config = self.config['distillation']
        loss = (
            distillation_config.get('global_weight', 1.0) * distillation_loss.cls
            + distillation_config.get('patch_weight', 1.0) * distillation_loss.patches
        )

        if self.training_mode == 'salad':
            # get the loss for each layer, (X - L - S)
            # update ema_r and ema_s for updating rho
            diff_per_rank = self.get_diff_per_rank()
            dist.all_reduce(diff_per_rank, op=dist.ReduceOp.SUM)
            global_avg_diff = diff_per_rank.item() / len(self.cfg_layers)
            # calculate the penalty loss of each layer
            # X with gradient -> rho/2 * (X - L - S + Y/rho)^2
            # only used for coupled gradient
            loss_penalty = self.get_penalty_loss()
            if gradient == 'decoupled':
                # Closed-form gradient: rho * (X - L - S + Y / rho).
                gradient_per_layer = self.get_gradient_per_layer()
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
                        get_param_tensor(param_dict, name, "weight").data -= eta * Z
                # broadcast the updated weights
                self.broadcast_params(self.ddp_model)
            
            self.lr_scheduler.step()

            # broadcast the neural network loss
            global_avg_loss, global_avg_cls_loss, global_avg_patch_loss = (
                self.get_global_losses(
                    loss.detach(),
                    distillation_loss.cls.detach(),
                    distillation_loss.patches.detach(),
                )
            )
            # broadcast the penalty loss
            global_avg_loss_penalty = self.get_global_loss(loss_penalty.detach())
            return (
                global_avg_loss,
                global_avg_cls_loss,
                global_avg_patch_loss,
                global_avg_loss_penalty,
                global_avg_diff,
            )
        elif self.training_mode == 'vanilla':
            loss.backward()

            if self.is_clip > 0:
                # Clip gradients to avoid exploding gradients
                # This is a common practice in training large models
                torch.nn.utils.clip_grad_norm_(self.ddp_model.parameters(), max_norm=self.is_clip)

            self.optimizer.step()
            self.lr_scheduler.step()

            # broadcast the neural network loss
            global_avg_loss, global_avg_cls_loss, global_avg_patch_loss = (
                self.get_global_losses(
                    loss.detach(),
                    distillation_loss.cls.detach(),
                    distillation_loss.patches.detach(),
                )
            )
            return (
                global_avg_loss,
                global_avg_cls_loss,
                global_avg_patch_loss,
                0.0,
                0.0,
            )

    def prepare_batch(self, batch):
        return batch["pixel_values"].to(
            device=self.device,
            dtype=torch.bfloat16,
            non_blocking=True,
        )

    def get_penalty_loss(self):
        """User-defined loss; can be overridden or passed via config."""
        loss = 0.0
        for solver in self.ADMM_solvers:
            if solver.layer_gpu_map == self.rank:
                loss += self.world_size * solver.get_penalty(solver.L, solver.S, solver.Y)
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

    def get_global_losses(self, *log_losses):
        """Average several scalar losses across ranks with one collective."""
        values = torch.stack(log_losses)
        with torch.no_grad():
            dist.all_reduce(values, op=dist.ReduceOp.SUM)
            values /= self.world_size
        return tuple(values.tolist())

    @torch.no_grad()
    def broadcast_params(self, ddp_model):
        param_dict = dict(ddp_model.named_parameters())

        names_me = self.per_owner_names[self.rank]
        sz_me    = self.owner_sizes[self.rank]

        flat_me = torch.empty(sz_me, device=self.device)
        off = 0
        for n in names_me:
            p = get_param_tensor(param_dict, n, "weight").data.view(-1)
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
                    p = get_param_tensor(param_dict, n, "weight")
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

        if self.training_mode == 'salad':
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

    def update_ADMM_rho(self): 
        """ Update the penalty parameter rho for all layers.
        """
        for solver in self.ADMM_solvers:
            solver.update_rho()

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
                   cls_loss: float,
                   patch_loss: float,
                   loss_penalty: float,
                   loss_diff: float,
                   acc_num_images: int,
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
                  'avg_cls_loss': cls_loss,
                  'avg_patch_loss': patch_loss,
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
        
        print_epoch(epoch, total_epochs, num_freq, lr, acc_num_images, losses, layer_stats)
        if self.is_wandb and self.rank == 0:
            print_wandb(self.run_wandb, 
                        epoch=epoch, 
                        total_epochs=total_epochs, 
                        num_freq=num_freq, 
                        lr=lr, 
                        num_images=acc_num_images,
                        losses=losses, 
                        layer_stats=layer_stats)

    def train(self, path_folder: str=None):
        self.ddp_model.train()
        num_it = 0
        num_epochs = self.num_total_iters // self.num_freq
        epoch = 0
        ep_loss = 0.0
        ep_cls_loss = 0.0
        ep_patch_loss = 0.0
        ep_penalty = 0.0
        ep_diff = 0.0
        acc_num_images = 0

        data_epoch = 0
        data_iterator = iter(self.dataloader)
        while num_it < self.num_total_iters:
            try:
                batch = next(data_iterator)
            except StopIteration:
                data_epoch += 1
                dataset = getattr(self.dataloader, "dataset", None)
                set_epoch = getattr(dataset, "set_epoch", None)
                if callable(set_epoch):
                    set_epoch(data_epoch)
                data_iterator = iter(self.dataloader)
                try:
                    batch = next(data_iterator)
                except StopIteration as error:
                    raise RuntimeError("dataloader yielded no batches") from error

            num_it += 1
            images = self.prepare_batch(batch)
            # do one step update
            with self.timers['train']:
                (
                    avg_loss,
                    avg_cls_loss,
                    avg_patch_loss,
                    avg_loss_penalty,
                    avg_diff,
                ) = self.single_step_train(images, gradient=self.gradient)

            if (
                num_it == 1
                and getattr(self, 'use_bfloat16_autocast', False)
            ):
                optimizer_state_dtypes = {
                    state[name].dtype
                    for state in self.optimizer.state.values()
                    for name in ('exp_avg', 'exp_avg_sq')
                    if name in state
                }
                if optimizer_state_dtypes != {torch.float32}:
                    raise RuntimeError(
                        "DINO mixed precision requires FP32 Adam state, "
                        f"got {optimizer_state_dtypes}"
                    )
                logger.info(
                    f"[Rank {self.rank}] Adam state dtype=torch.float32"
                )

            # calculate the constants
            num_images = images.shape[0] * self.world_size
            self.layer_info['avg_loss'].append(avg_loss)
            self.layer_info['avg_cls_loss'].append(avg_cls_loss)
            self.layer_info['avg_patch_loss'].append(avg_patch_loss)
            self.layer_info['avg_loss_penalty'].append(avg_loss_penalty)
            self.layer_info['avg_diff'].append(avg_diff)
            self.layer_info['num_images'].append(num_images)
            
            ep_loss += avg_loss
            ep_cls_loss += avg_cls_loss
            ep_patch_loss += avg_patch_loss
            ep_penalty += avg_loss_penalty
            ep_diff += avg_diff
            acc_num_images += num_images

            if num_it % self.num_freq == 0:
                epoch += 1

                if self.training_mode == 'salad':
                    with self.timers['L']:
                        self.update_ADMM_single_step(target='L')
                    self.update_ADMM_single_step(target='alpha')

                    with self.timers['S']:
                        self.update_ADMM_single_step(target='S')
                    self.update_ADMM_single_step(target='beta')
                    
                    self.update_ADMM_rho()
                    
                    with self.timers['Y']:
                        self.update_ADMM_single_step(target='Y')

                    with self.timers['sync']:
                        self.update_ADMM_single_step(target='save')
                        self.sync_layer_info()

                    self.solvers_reset()
                # average losses
                ep_loss /= self.num_freq
                ep_cls_loss /= self.num_freq
                ep_patch_loss /= self.num_freq
                ep_penalty /= self.num_freq
                ep_diff /= self.num_freq    
                
                if self.rank == 0:
                    self.print_info(epoch, 
                                    num_epochs,
                                    self.num_freq,
                                    ep_loss,
                                    ep_cls_loss,
                                    ep_patch_loss,
                                    ep_penalty,
                                    ep_diff, 
                                    acc_num_images,
                                    self.layer_info, 
                                    self.lr_scheduler.get_last_lr()[0])
                        
                    if self.is_monitor:
                        print(f'Train: {self.timers["train"].total:.3f}s | Avg Train: {self.timers["train"].avg():.3f}s | S: {self.timers["S"].total:.3f}s | L: {self.timers["L"].total:.3f}s | Y: {self.timers["Y"].total:.3f}s | Sync: {self.timers["sync"].total:.3f}s | Save: {self.timers["save"].total:.3f}s')
                        for key in self.timers:
                            self.timers[key].reset()

                ep_loss = 0.0
                ep_cls_loss = 0.0
                ep_patch_loss = 0.0
                ep_penalty = 0.0
                ep_diff = 0.0

            # Save every configured number of optimizer iterations. The final
            # iteration is also saved when the run length is not divisible by
            # save_interval. save_results uses fixed filenames and overwrites
            # the previous checkpoint atomically.
            should_save = (
                num_it % self.save_interval == 0
                or num_it == self.num_total_iters
            )
            if path_folder is not None and should_save:
                with self.timers['save']:
                    self.save_results(path_folder)

        device = getattr(self, "device", None)
        if getattr(device, "type", None) == "cuda":
            peak_allocated_gib = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
            peak_reserved_gib = torch.cuda.max_memory_reserved(device) / (1024 ** 3)
            logger.info(
                f"[Rank {self.rank}] peak CUDA memory: "
                f"allocated={peak_allocated_gib:.2f} GiB, "
                f"reserved={peak_reserved_gib:.2f} GiB"
            )

        dist.destroy_process_group()
        if self.is_wandb and self.rank == 0:
            self.run_wandb.finish()

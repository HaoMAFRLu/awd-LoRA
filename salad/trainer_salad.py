import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import math
import os
import datasets
import datasets.distributed
from loguru import logger

from salad.salad_solver import SALAD
from salad.loop import (
    DEFAULT_TIED_PARAMETER_NAMES,
    LoopSampler,
    LoopStabilitySampler,
    block_distance,
    block_parameter_errors,
    get_block_reference_norms,
    monotonic_stability_loss,
)
from salad.utils import *
from salad.register import *
from salad.simple_timer import SimpleTimer

class SALADTrainer():
    def __init__(self, 
                 model: nn.Module,
                 data: datasets.Dataset,
                 config: dict,
                 rank: int=0,
                 world_size: int=0,
                 folder_name: str=None) -> None:
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
        if not isinstance(self.save_interval, int) or self.save_interval <= 0:
            raise ValueError("save_interval must be a positive integer")

        self.training_mode = config.get('training_mode', 'salad')
        if self.training_mode not in {'salad', 'vanilla', 'loop'}:
            raise ValueError(
                "training_mode must be one of 'salad', 'vanilla', or 'loop'; "
                f"got {self.training_mode!r}"
            )
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
            self.run_wandb = wandb.init(project=self.config.get("wandb_project", "SALAD_"+self.config['name']),
                                        entity=self.config.get("wandb_entity", "hao-ma-eth-z-rich"),
                                        config=self.config,
                                        name=folder_name)
            self.run_wandb.define_metric("iteration")
            self.run_wandb.define_metric("*", step_metric="iteration")
            

        # torch.cuda.set_device(self.rank % torch.cuda.device_count())
        self.device = torch.device(f'cuda:{self.rank % torch.cuda.device_count()}')
        if self.rank == 0:
            print_setting(config)

        global_batch_size = config.get('batch_size', 32)
        if global_batch_size % self.world_size != 0:
            raise ValueError(
                f"batch_size ({global_batch_size}) must be divisible by world_size "
                f"({self.world_size})"
            )
        self.batch_size = global_batch_size // self.world_size

        # print device info
        dev_idx = torch.cuda.current_device()
        props   = torch.cuda.get_device_properties(dev_idx)
        logger.info(f"[Rank {self.rank}] using {props.name}, {props.total_memory / (1024 ** 3):.2f} GiB")       

        # Wrap model in DDP
        self.model.cuda()
        # get all the names of the model layers
        self.names_model_layers = get_linear_layers_name(self.model)
        # get specified layers in the config
        if self.training_mode == 'salad':
            self.cfg_layers = self.get_cfg_layers(self.config, self.names_model_layers)
        elif self.training_mode == 'loop':
            self.cfg_layers = []
            self._configure_loop_penalty()
        else:
            self.cfg_layers = [{'name': 'layers.0.self_attn.q_proj'}]  # dummy for vanilla training

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

        self.ddp_model = DDP(self.model.to(torch.bfloat16), 
                             device_ids=[torch.cuda.current_device()])
        if self.training_mode == 'loop':
            # Capture the fixed denominator after DDP has synchronized rank 0's
            # initialization to every worker.
            self.soft_tie_reference_norms = get_block_reference_norms(
                self.ddp_model,
                self.soft_tie_source_block,
                self.soft_tie_parameter_names,
            )

        data = datasets.distributed.split_dataset_by_node(data, rank=self.rank, world_size=self.world_size)

        tokenizer = get_tokenizer(self.max_length, self.config)
        dataset = get_preprocessed_dataset(data, tokenizer, self.config, self.batch_size)
        self.dataloader = torch.utils.data.DataLoader(dataset, 
                                                      batch_size=None, 
                                                      num_workers=self.num_workers)
        self.pad_idx = tokenizer.tokenizer.pad_token_id if hasattr(tokenizer, "tokenizer") else tokenizer.pad_token_id

        self.optimizer = get_optimizer(*self.get_name_and_params(config['optimizer']), self.ddp_model)
        self.lr_scheduler = get_scheduler(self.optimizer,
                                        scheduler_type=config['scheduler']['name'],
                                        num_training_steps=self.num_total_iters,
                                        warmup_steps=config['scheduler']['params'].get('warmup_steps', 0),
                                        min_lr_ratio=config['scheduler']['params'].get('min_lr_ratio', 0.0))
        # warmup the model
        # self.warmup(self.num_warmup_steps)
        
        
        if self.training_mode == 'salad':  # only do the admm for the salad training
            # assign layers to different GPUs
            self.assigned_layers, self.owner_map = self.assign_layers(self.cfg_layers, self.rank, self.world_size)
            self.per_owner_names, self.owner_sizes = self.build_per_owner_static(self.ddp_model, self.owner_map, self.world_size)

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
        if self.training_mode == 'loop':
            self.layer_info['num_loops'] = []
            self.layer_info['long_num_loops'] = []
            self.layer_info['stability_active'] = []
            self.layer_info['long_task_loss'] = []
            self.layer_info['long_loss_delta'] = []
            self.layer_info['stability_loss'] = []
            self.layer_info['weighted_stability_loss'] = []
            self.layer_info['stability_violation'] = []
            self.layer_info['parameter_errors'] = {
                name: [] for name in self.soft_tie_parameter_names
            }

    def _configure_loop_penalty(self) -> None:
        loop_config = getattr(self.model.config, "loop", None)
        if not isinstance(loop_config, dict):
            raise ValueError("training_mode='loop' requires a 'loop' section in the model config")

        soft_tie = self.config.get("soft_tie")
        if not isinstance(soft_tie, dict):
            raise ValueError("training_mode='loop' requires a 'soft_tie' section")

        self.soft_tie_rho = float(soft_tie.get("rho", 0.0))

        self.soft_tie_source_block = soft_tie.get("source_block", "layers.0")
        self.soft_tie_target_block = soft_tie.get("target_block", "layers.3")
        self.soft_tie_parameter_names = tuple(
            soft_tie.get("parameter_names", DEFAULT_TIED_PARAMETER_NAMES)
        )
        self.soft_tie_epsilon = float(soft_tie.get("epsilon", 1e-12))
        if not isinstance(self.soft_tie_source_block, str) or not isinstance(
            self.soft_tie_target_block, str
        ):
            raise TypeError("soft_tie source_block and target_block must be strings")
        if self.soft_tie_source_block == self.soft_tie_target_block:
            raise ValueError("soft_tie source_block and target_block must be different")

        training_loop_config = self.config.get("loop")
        if not isinstance(training_loop_config, dict):
            raise ValueError("training_mode='loop' requires a 'loop' section")
        sampling = training_loop_config.get("sampling")
        if not isinstance(sampling, dict):
            raise ValueError("loop training requires loop.sampling")
        self.loop_sampler = LoopSampler(
            values=sampling.get("values", []),
            probabilities=sampling.get("probabilities", []),
            seed=int(sampling.get("seed", self.config.get("seed", 42))),
            expected_value=sampling.get("expected_value"),
        )
        self.current_num_loops = int(
            training_loop_config.get("num_loops", loop_config.get("num_loops", 1))
        )
        self.loop_counts = {value: 0 for value in self.loop_sampler.values}

        stability = training_loop_config.get("stability", {})
        if not isinstance(stability, dict):
            raise ValueError("loop.stability must be a mapping")
        self.stability_enabled = stability.get("enabled", False)
        if not isinstance(self.stability_enabled, bool):
            raise TypeError("loop.stability.enabled must be a boolean")
        self.stability_weight = float(stability.get("weight", 0.1))
        if not math.isfinite(self.stability_weight) or self.stability_weight < 0.0:
            raise ValueError("loop.stability.weight must be finite and non-negative")
        if self.stability_enabled and self.stability_weight == 0.0:
            raise ValueError("enabled loop stability requires a positive weight")

        self.stability_sampler = None
        if self.stability_enabled:
            self.stability_sampler = LoopStabilitySampler(
                probability=stability.get("probability", 0.25),
                deltas=stability.get("deltas", [1, 2, 4]),
                seed=int(stability.get("seed", self.config.get("seed", 42))),
            )
        self.current_stability_active = False
        self.current_long_num_loops = 0
        self.current_long_task_loss = float("nan")
        self.current_long_loss_delta = float("nan")
        self.current_stability_loss = 0.0
        self.current_weighted_stability_loss = 0.0
        self.current_stability_violation = False

        # Validate all names and shapes before DDP and the optimizer are built.
        block_distance(
            self.model,
            self.soft_tie_source_block,
            self.soft_tie_target_block,
            self.soft_tie_parameter_names,
            self.soft_tie_epsilon,
        )

    def get_loop_penalty(self):
        """Return the penalty, block distance, and per-matrix errors."""
        errors = block_parameter_errors(
            self.ddp_model,
            self.soft_tie_source_block,
            self.soft_tie_target_block,
            self.soft_tie_parameter_names,
            self.soft_tie_epsilon,
            self.soft_tie_reference_norms,
        )
        distance = torch.stack(tuple(errors.values())).mean()
        return 0.5 * self.soft_tie_rho * distance, distance, errors

    @torch.no_grad()
    def log_loop_parameter_errors(
        self,
        iteration: int,
        errors: dict,
        distance: torch.Tensor,
    ) -> None:
        """Log the pre-update T1/TN-1 errors used by the current loss."""
        if self.rank != 0:
            return

        for name, error in errors.items():
            self.layer_info['parameter_errors'][name].append(float(error.item()))
        if not self.is_wandb:
            return
        payload = {
            f"loop/parameter_error/{name.removesuffix('.weight')}": float(error.item())
            for name, error in errors.items()
        }
        payload.update({
            "loop/pre_update_block_distance": float(distance.item()),
            "loop/num_loops": self.current_num_loops,
            "loop/stability_enabled": int(self.stability_enabled),
            "loop/stability_active": int(self.current_stability_active),
        })
        if self.current_stability_active:
            payload.update({
                "loop/long_num_loops": self.current_long_num_loops,
                "loop/long_task_loss": self.current_long_task_loss,
                "loop/long_loss_delta": self.current_long_loss_delta,
                "loop/stability_loss": self.current_stability_loss,
                "loop/weighted_stability_loss": (
                    self.current_weighted_stability_loss
                ),
                "loop/stability_violation": int(
                    self.current_stability_violation
                ),
            })
        payload["iteration"] = iteration
        # Keep the row open when the periodic summary is written at this same
        # iteration, so W&B stores one history row rather than duplicate x-axis
        # points.
        self.run_wandb.log(payload, commit=iteration % self.num_freq != 0)

    def sample_num_loops(self) -> int:
        """Draw one depth on rank 0 and use it on every DDP rank."""
        if self.rank == 0:
            sampled = self.loop_sampler.sample()
        else:
            sampled = 0

        if dist.is_available() and dist.is_initialized():
            sampled_tensor = torch.tensor(sampled, device=self.device, dtype=torch.int64)
            dist.broadcast(sampled_tensor, src=0)
            sampled = int(sampled_tensor.item())

        self._set_num_loops(sampled)
        self.current_num_loops = sampled
        self.loop_counts[sampled] += 1
        return sampled

    def _set_num_loops(self, num_loops: int) -> None:
        model = self.ddp_model.module if hasattr(self.ddp_model, "module") else self.ddp_model
        model.model.set_num_loops(num_loops)

    def sample_stability_delta(self):
        """Draw one optional long-path extension and share it across DDP ranks."""
        if self.stability_sampler is None:
            return None

        if self.rank == 0:
            sampled = self.stability_sampler.sample()
            encoded_delta = 0 if sampled is None else sampled
        else:
            encoded_delta = 0

        if dist.is_available() and dist.is_initialized():
            delta_tensor = torch.tensor(
                encoded_delta,
                device=self.device,
                dtype=torch.int64,
            )
            dist.broadcast(delta_tensor, src=0)
            encoded_delta = int(delta_tensor.item())

        return None if encoded_delta == 0 else encoded_delta
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

    def single_step_train(self, batch, labels, gradient: str='coupled', iteration: int=None):
        if self.training_mode == 'salad':
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
                        get_param_tensor(param_dict, name, "weight").data -= eta * Z
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
        elif self.training_mode == 'vanilla':
            # reset the gradient
            self.optimizer.zero_grad(set_to_none=True)
            # calculate the loss of the neural network
            loss = self.ddp_model(**batch, labels=labels).loss
            loss.backward()

            if self.is_clip > 0:
                # Clip gradients to avoid exploding gradients
                # This is a common practice in training large models
                torch.nn.utils.clip_grad_norm_(self.ddp_model.parameters(), max_norm=self.is_clip)

            self.optimizer.step()
            self.lr_scheduler.step()

            # broadcast the neural network loss
            global_avg_loss = self.get_global_loss(loss.detach())
            return global_avg_loss, 0.0, 0.0
        elif self.training_mode == 'loop':
            if iteration is None:
                raise ValueError("loop training requires the current iteration")
            self.optimizer.zero_grad(set_to_none=True)
            base_num_loops = self.sample_num_loops()
            stability_delta = self.sample_stability_delta()
            self.current_stability_active = stability_delta is not None
            self.current_long_num_loops = 0
            self.current_long_task_loss = float("nan")
            self.current_long_loss_delta = float("nan")
            self.current_stability_loss = 0.0
            self.current_weighted_stability_loss = 0.0
            self.current_stability_violation = False

            if stability_delta is None:
                task_loss = self.ddp_model(**batch, labels=labels).loss
                task_loss_for_logging = task_loss.detach()
                loss_penalty, normalized_distance, parameter_errors = (
                    self.get_loop_penalty()
                )
                (task_loss + loss_penalty).backward()
                del task_loss
                long_task_loss_for_logging = None
                stability_loss_for_logging = task_loss_for_logging.new_zeros(())
            else:
                # Do not synchronize the short backward yet. The long backward
                # synchronizes the accumulated short and long gradients once.
                with self.ddp_model.no_sync():
                    task_loss = self.ddp_model(**batch, labels=labels).loss
                    task_loss_for_logging = task_loss.detach()
                    loss_penalty, normalized_distance, parameter_errors = (
                        self.get_loop_penalty()
                    )
                    (task_loss + loss_penalty).backward()
                del task_loss

                long_num_loops = base_num_loops + stability_delta
                self._set_num_loops(long_num_loops)
                try:
                    long_task_loss = self.ddp_model(**batch, labels=labels).loss
                    stability_loss = monotonic_stability_loss(
                        task_loss_for_logging,
                        long_task_loss,
                    )
                    weighted_stability_loss = (
                        self.stability_weight * stability_loss
                    )
                    weighted_stability_loss.backward()
                    long_task_loss_for_logging = long_task_loss.detach()
                    stability_loss_for_logging = stability_loss.detach()
                    del long_task_loss, stability_loss, weighted_stability_loss
                finally:
                    # Checkpoints and per-iteration logs describe the sampled
                    # base execution, not the temporary long branch.
                    self._set_num_loops(base_num_loops)

            if self.is_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.ddp_model.parameters(), max_norm=self.is_clip
                )

            self.optimizer.step()
            self.lr_scheduler.step()

            global_avg_loss = self.get_global_loss(task_loss_for_logging)
            global_avg_loss_penalty = self.get_global_loss(loss_penalty.detach())
            global_avg_distance = self.get_global_loss(
                normalized_distance.detach().clone()
            )
            if long_task_loss_for_logging is not None:
                self.current_long_num_loops = base_num_loops + stability_delta
                self.current_long_task_loss = self.get_global_loss(
                    long_task_loss_for_logging
                )
                self.current_long_loss_delta = (
                    self.current_long_task_loss - global_avg_loss
                )
                self.current_stability_loss = self.get_global_loss(
                    stability_loss_for_logging
                )
                self.current_weighted_stability_loss = (
                    self.stability_weight * self.current_stability_loss
                )
                self.current_stability_violation = self.current_stability_loss > 0.0

            self.log_loop_parameter_errors(
                iteration,
                parameter_errors,
                normalized_distance,
            )
            return global_avg_loss, global_avg_loss_penalty, global_avg_distance

    def prepare_batch_and_labels(self, batch):
        batch = {k: v.to(self.device) for k, v in batch.items()}
        labels = batch.pop("labels", None)
        if labels is None:
            labels = batch["input_ids"].clone()
            labels[labels == self.pad_idx] = -100
        return batch, labels

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

    def generate_empty_layer_info(self):
        """Empty layer info for vanilla training."""
        if self.rank == 0:
            info = self.layer_info['layers.0.self_attn.q_proj']
            info['alpha_mode'].append('N/A')
            info['beta_mode'].append('N/A')
            info['alpha'].append(0.0)
            info['beta'].append(0.0)
            info['dalpha'].append(0.0)
            info['dbeta'].append(0.0)
            info['rho'].append(0.0)
            info['rate_decay_alpha'].append(0.0)
            info['rate_decay_beta'].append(0.0)
            info['loss'].append(0.0)
            info['rank'].append(1)
            info['nonzero'].append(1)
            info['total_rank'].append(1)
            info['total_elements'].append(1)
    

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
            if dist.is_available() and dist.is_initialized():
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
                   lr: float,
                   loop_metrics: dict=None):
        """
        Print training information for the current epoch.
        Args:
            epoch: Current epoch number
            total_epochs: Total number of epochs
            layer_info: Dictionary containing layer statistics
        """
        loop_total = sum(self.loop_counts.values()) if self.training_mode == 'loop' else 0
        loop_ratios = (
            {value: count / loop_total for value, count in self.loop_counts.items()}
            if loop_total else {}
        )
        mean_num_loops = (
            sum(value * count for value, count in self.loop_counts.items()) / loop_total
            if loop_total else 1.0
        )
        losses = {'avg_loss': loss,
                  'avg_loss_penalty': loss_penalty,
                  'avg_diff': loss_diff,
                  'training_mode': self.training_mode,
                  'num_loops': self.current_num_loops if self.training_mode == 'loop' else 1,
                  'mean_num_loops': mean_num_loops,
                  'loop_ratios': loop_ratios}
        if self.training_mode == 'loop' and loop_metrics is not None:
            losses.update(loop_metrics)
        
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
                        lr=lr, 
                        num_tokens=acc_num_tokens, 
                        losses=losses, 
                        layer_stats=layer_stats)

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
            
            batch, labels = self.prepare_batch_and_labels(batch)
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
        ep_stability_loss = 0.0
        ep_long_loss = 0.0
        ep_stability_steps = 0
        ep_stability_violations = 0
        num_tokens = 0
        acc_num_tokens = 0

        for batch_idx, batch in enumerate(self.dataloader):
            num_it += 1
            # terminate training if reached max iterations
            if num_it > self.num_total_iters:
                logger.info(f"Reached max number of update steps (f{self.num_total_iters}). Stopping training.")
                print(f"Rank {self.rank} stopping training.")
                break
            

            batch, labels = self.prepare_batch_and_labels(batch)
            # do one step update
            with self.timers['train']:
                avg_loss, avg_loss_penalty, avg_diff = self.single_step_train(
                    batch,
                    labels,
                    gradient=self.gradient,
                    iteration=num_it,
                )

            # calculate the constants
            num_tokens = (batch['input_ids'].numel() - torch.sum(batch['input_ids'] == self.pad_idx).item()) * self.world_size
            self.layer_info['avg_loss'].append(avg_loss)
            self.layer_info['avg_loss_penalty'].append(avg_loss_penalty)
            self.layer_info['avg_diff'].append(avg_diff)
            self.layer_info['num_tokens'].append(num_tokens)
            if self.training_mode == 'loop':
                self.layer_info['num_loops'].append(self.current_num_loops)
                self.layer_info['long_num_loops'].append(
                    self.current_long_num_loops
                )
                self.layer_info['stability_active'].append(
                    self.current_stability_active
                )
                self.layer_info['long_task_loss'].append(
                    self.current_long_task_loss
                )
                self.layer_info['long_loss_delta'].append(
                    self.current_long_loss_delta
                )
                self.layer_info['stability_loss'].append(
                    self.current_stability_loss
                )
                self.layer_info['weighted_stability_loss'].append(
                    self.current_weighted_stability_loss
                )
                self.layer_info['stability_violation'].append(
                    self.current_stability_violation
                )
                ep_stability_loss += self.current_weighted_stability_loss
                if self.current_stability_active:
                    ep_long_loss += self.current_long_task_loss
                    ep_stability_steps += 1
                    ep_stability_violations += int(
                        self.current_stability_violation
                    )

            ep_loss += avg_loss
            ep_penalty += avg_loss_penalty
            ep_diff += avg_diff
            acc_num_tokens += num_tokens

            # now we update S and Y at each iteration
            # asynchronous update for 
            if num_it % self.num_freq == 0:
                # run admm solvers
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
                        # self.sync_single_weight(target='S')
                        # self.sync_single_weight(target='L')
                        # self.sync_single_weight(target='Y')
                        # self.sync_all_weights()
                        pass
                    
                        self.update_ADMM_single_step(target='save')
                        self.sync_layer_info()

                    self.solvers_reset()
                elif self.training_mode == 'vanilla':
                    self.generate_empty_layer_info()

                # self.run_ADMM_solvers()
                # self.sync_results()

                # average losses
                ep_loss /= self.num_freq
                ep_penalty /= self.num_freq
                ep_diff /= self.num_freq
                loop_metrics = None
                if self.training_mode == 'loop':
                    loop_metrics = {
                        'stability_enabled': self.stability_enabled,
                        'avg_stability_loss': ep_stability_loss / self.num_freq,
                        'avg_long_loss': (
                            ep_long_loss / ep_stability_steps
                            if ep_stability_steps else float('nan')
                        ),
                        'stability_branch_rate': (
                            ep_stability_steps / self.num_freq
                        ),
                        'stability_violation_rate': (
                            ep_stability_violations / ep_stability_steps
                            if ep_stability_steps else float('nan')
                        ),
                        'long_num_loops': self.current_long_num_loops,
                        'stability_weight': self.stability_weight,
                    }
                
                if self.rank == 0:
                    self.print_info(epoch, 
                                    num_epochs,
                                    self.num_freq,
                                    ep_loss,
                                    ep_penalty,
                                    ep_diff, 
                                    acc_num_tokens, 
                                    self.layer_info, 
                                    self.lr_scheduler.get_last_lr()[0],
                                    loop_metrics=loop_metrics)
                        
                    if self.is_monitor:
                        print(f'Train: {self.timers["train"].total:.3f}s | Avg Train: {self.timers["train"].avg():.3f}s | S: {self.timers["S"].total:.3f}s | L: {self.timers["L"].total:.3f}s | Y: {self.timers["Y"].total:.3f}s | Sync: {self.timers["sync"].total:.3f}s | Save: {self.timers["save"].total:.3f}s')
                        for key in self.timers:
                            self.timers[key].reset()

                ep_loss, ep_penalty, ep_diff = 0.0, 0.0, 0.0
                ep_stability_loss = 0.0
                ep_long_loss = 0.0
                ep_stability_steps = 0
                ep_stability_violations = 0
            
            else:
                if self.is_asyn:
                    pass
                    # self.update_ADMM_single_step(target='beta')
                    
                    # self.update_ADMM_single_step(target='S')
                    # self.update_ADMM_single_step(target='Y')

                    # self.sync_single_weight(target='S')
                    # self.sync_single_weight(target='Y')

            # save_interval is measured in optimizer iterations. Save after
            # every update belonging to this iteration has completed.
            should_save = (
                num_it % self.save_interval == 0
                or num_it == self.num_total_iters
            )
            if path_folder is not None and should_save:
                with self.timers['save']:
                    self.save_results(path_folder)

        dist.destroy_process_group()
        if self.is_wandb and self.rank == 0:
            self.run_wandb.finish()

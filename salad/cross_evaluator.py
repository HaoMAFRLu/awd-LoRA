import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import copy
import math

from salad.utils import *
from salad.simple_timer import SimpleTimer
from salad.operators import *


class CrossEvaluator():
    """
    Class for cross-evaluation of models.
    """
    def __init__(self,
                 model_type: str,
                 model: nn.modules=None,
                 train_loader: torch.utils.data.DataLoader=None,
                 test_loader: torch.utils.data.DataLoader=None,
                 LL: dict=None,
                 SS: dict=None,
                 layers: list=None,
                 pad_idx: int=0,
                 layer_dim: dict=None,
                 batch_size: int=10) -> None:
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # print device info
        self.dev_idx = torch.cuda.current_device() if torch.cuda.is_available() else -1
        props   = torch.cuda.get_device_properties(self.dev_idx) if torch.cuda.is_available() else None
        if props:
            print(f"[Rank {self.dev_idx}] using {props.name}, {props.total_memory / (1024 ** 3):.2f} GiB")
        else:
            print("[Rank -1] using CPU")

        # fk the error 429!!!!!!
        hf_login_once()

        self.pad_idx = pad_idx
        self.model_type = model_type
        self.batch_size = batch_size
        self.total_params = sum(p.numel() for p in model.parameters())
        self.model = model.to(self.device) if model is not None else None
        self.model_sd = (
            {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
            if self.model is not None else None)

        self.train_loader = train_loader
        self.test_loader = test_loader
        self.layer_dim = layer_dim

        self.LL = LL if LL is not None else {}
        self.SS = SS if SS is not None else {}
        self.layers = layers if layers is not None else []
        
        self.model_layers = get_linear_layers_name(self.model) if model is not None else []
        
        self.eval_train_results = {}
        self.eval_test_results = {}

    def _eval_original(self, dataloader) -> dict:
        """ Evaluate the original model.
        Returns:
            Dictionary with evaluation results.
        """
        # evaluate the original model, X
        opt_copy(self.model_sd, self.model, self.layers, self.device)
        return self.evaluate_one_step(self.model, dataloader)

    # def _eval_lowrank_sparsity(self, dataloader) -> dict:
    #     """Evaluate the low-rank model with sparsity."""
    #     self.opt_replace(self.model, self.layers, self.LL)
    #     self.opt_add(self.model, self.layers, self.SS)
    #     return self.evaluate_one_step(self.model, dataloader)

    def _eval_par_lowrank_lowrank_sparsity(self, dataloader, rank_quantile, rate_density) -> dict:
        """Evaluate the partial low-rank model with low-rank approximation and sparsity."""
        opt_copy(self.model_sd, self.model, self.layers, self.device)  # copy the original model
        opt_replace(self.model, self.layers, self.LL, self.device)  # replace partial layers with low-rank matrices L
        opt_lowrank(self.model, self.layers, rank_quantile, self.device)
        _SS = re_sparse(self.SS, rate_density)
        self.opt_add(self.model, self.layers, _SS, self.device)  # add sparse components S
        return self.evaluate_one_step(self.model, dataloader)
    
    @torch.no_grad()
    def eval_original_model(self, dataloader) -> dict:
        """Evaluate the original model."""
        return self._eval_original(dataloader)

    @torch.no_grad()        
    def eval_model(self,
                   rank_quantile_list: list,
                   rate_density_list: list,
                   dataloader) -> dict:
        """
        Evaluate the lowspa model.
        Returns:
            Dictionary with evaluation results.
        """
        eval_results = {}
        timer = SimpleTimer('evaluation')

        for i in range(len(rank_quantile_list)):
            
            print(f"[Rank {self.dev_idx}] Evaluating model {i}...")
            rank_quantile = rank_quantile_list[i]
            rate_density = rate_density_list[i]
            with timer:
                eval_results[f'par_lowrank_L_with_S_{i}'] = self._eval_par_lowrank_lowrank_sparsity(dataloader, rank_quantile, rate_density) 
                eval_results[f'nr_par_lowrank_L_with_S_{i}'] = cal_nr_params(self.total_params, rank_quantile, rate_density, self.layer_dim)
            print(f"[Rank {self.dev_idx}] Evaluation time for model {i}: {timer.total/60:.1f} mins.")
            timer.reset()

        return eval_results
    
    @torch.no_grad()        
    def evaluate_one_step(self,
                          model: nn.Module,
                          dataloader,
                          target_eval_tokens: int=1_000_000) -> dict:
        """
        """
        model.eval()
        evaluated_on_tokens = 0
        total_loss = 0.0
        total_batches = 0
        loss_list = []
        with torch.inference_mode():
            for batch in dataloader.batch(batch_size=self.batch_size):
                
                if evaluated_on_tokens > target_eval_tokens:
                    break
                total_batches += 1

                batch = {k: v.to(self.device) for k, v in batch.items()}
                labels = batch["input_ids"].clone()
                labels[labels == self.pad_idx] = -100
                loss = model(**batch, labels=labels).loss
                total_loss += loss.item()
                evaluated_on_tokens += (batch["input_ids"] != self.pad_idx).sum().item()

                loss_list.append(total_loss / total_batches)
            return {'avg_loss': loss_list, 
                    'ppl': np.exp(loss_list[-1])}  # Return average loss and perplexity

    def collect_model_results(self) -> None:
        if self.model is not None:
            self.model_results_train = self.eval_original_model(self.train_loader) 
            self.model_results_test = self.eval_original_model(self.test_loader)
            print('Original model evaluation done.')
    
    def collect_single_results(self,
                               rank_quantile: dict,
                               rate_density: dict) -> None:
        if self.model is not None:
            # self.fullrank_results_train =  self._eval_lowrank_sparsity(self.train_loader) 
            self.fullrank_results_train = self._eval_par_lowrank_lowrank_sparsity(self.train_loader, rank_quantile, rate_density)
            self.fullrank_results_train_params = cal_nr_params(self.total_params, rank_quantile, rate_density, self.layer_dim)
            # self.fullrank_results_test = self._eval_lowrank_sparsity(self.test_loader)
            self.fullrank_results_test = self._eval_par_lowrank_lowrank_sparsity(self.test_loader, rank_quantile, rate_density)
            self.fullrank_results_test_params = cal_nr_params(self.total_params, rank_quantile, rate_density, self.layer_dim)
            print('Full-rank + sparsity model evaluation done.')
    
    def collect_results(self,
                        rank_quantile_list: list,
                        rate_density_list: list) -> None:
        """
        Collect results from the lowspa model.
        Returns:
            Dictionary with evaluation results.
        """
        if self.model is not None:
            self.eval_train_results = self.eval_model(rank_quantile_list, rate_density_list, self.train_loader) 
            self.eval_test_results = self.eval_model(rank_quantile_list, rate_density_list, self.test_loader)
        
        self.eval_train_results['X'] = self.model_results_train
        self.eval_test_results['X'] = self.model_results_test
        self.eval_train_results['nr_X'] = self.total_params
        self.eval_test_results['nr_X'] = self.total_params

        self.eval_train_results['L_with_S'] = self.fullrank_results_train
        self.eval_train_results['nr_L_with_S'] = self.fullrank_results_train_params
        self.eval_test_results['L_with_S'] = self.fullrank_results_test
        self.eval_test_results['nr_L_with_S'] = self.fullrank_results_test_params
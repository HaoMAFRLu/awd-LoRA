import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import copy
import math

from salad.utils import *

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
                 ex_layers: list=[],
                 pad_idx: int=0,
                 rank_quantile: list=None,
                 batch_size: int=10) -> None:
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # print device info
        dev_idx = torch.cuda.current_device() if torch.cuda.is_available() else -1
        props   = torch.cuda.get_device_properties(dev_idx) if torch.cuda.is_available() else None
        if props:
            print(f"[Rank {dev_idx}] using {props.name}, {props.total_memory / (1024 ** 3):.2f} GiB")
        else:
            print("[Rank -1] using CPU")

        # fk the error 429!!!!!!
        hf_login_once()

        self.pad_idx = pad_idx
        self.model_type = model_type
        self.batch_size = batch_size
        self.model = model.to(self.device) if model is not None else None
        self.model_sd = (
            {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
            if self.model is not None else None)

        self.train_loader = train_loader
        self.test_loader = test_loader
        self.rank_quantile = rank_quantile
        self.LL = LL if LL is not None else {}
        self.SS = SS if SS is not None else {}
        self.layers = layers if layers is not None else []
        
        if len(ex_layers) > 0:
            self.is_partial = True
            self.partial_layers = [layer for layer in self.layers if layer not in ex_layers]
        else:
            self.is_partial = False
            self.partial_layers = []

        self.model_layers = get_linear_layers_name(self.model) if model is not None else []
        
        self.eval_train_results = {}
        self.eval_test_results = {}

    def _eval_original(self, dataloader) -> dict:
        """ Evaluate the original model.
        Returns:
            Dictionary with evaluation results.
        """
        # evaluate the original model, X
        self.opt_copy(self.model_sd, self.model, self.layers)
        return self.evaluate_one_step(self.model, dataloader)
   
    def _eval_orginal_without_sparsity(self, dataloader) -> dict:
        """Evaluate the original model without sparsity."""
        # evaluate the original model, X - S
        self.opt_copy(self.model_sd, self.model, self.layers)
        self.opt_remove(self.model, self.layers, self.SS)
        return self.evaluate_one_step(self.model, dataloader)
         
    def _eval_original_lowrank_without_sparsity(self, dataloader) -> dict:
        """Evaluate the original model with low-rank approximation without sparsity."""
        # evaluate the original model, 90% low-rank approximation of (X - S)
        self.opt_copy(self.model_sd, self.model, self.layers)
        self.opt_remove(self.model, self.layers, self.SS)
        self.opt_lowrank(self.model, self.layers, self.rank_quantile)
        return self.evaluate_one_step(self.model, dataloader)
        
    def _eval_lowrank(self, dataloader) -> dict:
        """Evaluate the low-rank model."""
        self.opt_replace(self.model, self.layers, self.LL)
        return self.evaluate_one_step(self.model, dataloader)
     
    def _eval_lowrank_lowrank(self, dataloader) -> dict:
        """Evaluate the low-rank model with low-rank approximation."""
        self.opt_replace(self.model, self.layers, self.LL)
        self.opt_lowrank(self.model, self.layers, self.rank_quantile)
        return self.evaluate_one_step(self.model, dataloader)
          
    def _eval_lowrank_sparsity(self, dataloader) -> dict:
        """Evaluate the low-rank model with sparsity."""
        self.opt_replace(self.model, self.layers, self.LL)
        self.opt_add(self.model, self.layers, self.SS)
        return self.evaluate_one_step(self.model, dataloader)
    
    def _eval_par_lowrank_sparsity(self, dataloader) -> dict:
        """Evaluate the partial low-rank model with sparsity."""
        self.opt_copy(self.model_sd, self.model, self.layers)
        self.opt_replace(self.model, self.partial_layers, self.LL)
        self.opt_add(self.model, self.partial_layers, self.SS)
        return self.evaluate_one_step(self.model, dataloader)

    def _eval_lowrank_lowrank_sparsity(self, dataloader) -> dict:
        """Evaluate the low-rank model with low-rank approximation and sparsity."""
        self.opt_replace(self.model, self.layers, self.LL)
        self.opt_lowrank(self.model, self.layers, self.rank_quantile)
        self.opt_add(self.model, self.layers, self.SS)
        return self.evaluate_one_step(self.model, dataloader)

    # def _eval_par_lowrank_lowrank_sparsity(self, dataloader) -> dict:
    #     """Evaluate the partial low-rank model with low-rank approximation and sparsity."""
    #     self.opt_replace(self.model, self.layers, self.LL)  # replace all layers with full low-rank matrices L
    #     self.opt_lowrank(self.model, self.partial_layers, self.rank_quantile)  #  apply low-rank approximation to partial layers
    #     self.opt_add(self.model, self.layers, self.SS)
    #     return self.evaluate_one_step(self.model, dataloader)

    def _eval_par_lowrank_lowrank_sparsity(self, dataloader) -> dict:
        """Evaluate the partial low-rank model with low-rank approximation and sparsity."""
        self.opt_copy(self.model_sd, self.model, self.layers)  # copy the original model
        self.opt_replace(self.model, self.partial_layers, self.LL)  # replace partial layers with low-rank matrices L
        self.opt_lowrank(self.model, self.partial_layers, self.rank_quantile)
        self.opt_add(self.model, self.partial_layers, self.SS)  # add sparse components S
        return self.evaluate_one_step(self.model, dataloader)

    @torch.no_grad()        
    def eval_model(self,
                   eval_results: dict,
                   dataloader) -> dict:
        """
        Evaluate the lowspa model.
        Returns:
            Dictionary with evaluation results.
        """
        eval_results['X'] = self._eval_original(dataloader)
        eval_results['X_without_S'] = self._eval_orginal_without_sparsity(dataloader)
        eval_results['lowrank_X_without_S'] = self._eval_original_lowrank_without_sparsity(dataloader)
        eval_results['L'] = self._eval_lowrank(dataloader)
        eval_results['lowrank_L'] = self._eval_lowrank_lowrank(dataloader)
        eval_results['L_with_S'] = self._eval_lowrank_sparsity(dataloader)
        # eval_results['par_L_with_S'] = self._eval_par_lowrank_sparsity(dataloader) if self.is_partial else {'avg_loss': ['N/A'], 'ppl': 'N/A'}
        eval_results['lowrank_L_with_S'] = self._eval_lowrank_lowrank_sparsity(dataloader)
        eval_results['par_lowrank_L_with_S'] = self._eval_par_lowrank_lowrank_sparsity(dataloader) if self.is_partial else eval_results['lowrank_L_with_S']
        return eval_results
    
    def opt_copy(self,
                 model_source: nn.Module,
                 model_target: nn.Module,
                 layers: list) -> None:
        """Copy the weights from source model to target model for specified layers."""
        model_target.load_state_dict(model_source, strict=True)
    
    def opt_lowrank(self, 
                    model: nn.Module, 
                    layers: list,
                    rank_quantile: float) -> None:
        """Do low-rank approximation on specified layers of the model.
        Args:
            model: The model to optimize.
            layers: List of layer names to apply low-rank approximation.
        """
        for layer_name in layers:
            if 'model.'+layer_name in self.model_layers:
                layer = model.get_submodule('model.'+layer_name)
                weight = layer.weight.data
                U, s, V = torch.linalg.svd(weight, full_matrices=False)
                # nr_singular_values = get_energy_quantile(s, quantile=rank_quantile)
                nr_singular_values = int(len(s) * rank_quantile['layer_name'])
                low_rank_weight = U[:, :nr_singular_values] @ torch.diag(s[:nr_singular_values]) @ V[:nr_singular_values, :]
                layer.weight.copy_(low_rank_weight.to(self.device))
            else:
                print(f"Warning: Layer {layer_name} not found in model for low-rank optimization.")

    def opt_add(self,
                model: nn.Module,
                layers: list,
                SS: dict) -> None:
        """Add sparse components to the model."""
        for layer_name in layers:
            if layer_name in SS:
                if 'model.'+layer_name in self.model_layers:
                    layer = model.get_submodule('model.'+layer_name)
                    layer.weight.data += SS[layer_name].to(self.device)
            else:
                print(f"Warning: Sparse component for layer {layer_name} not found in SS dictionary.")

    def opt_replace(self,
                    model: nn.Module,
                    layers: list,
                    LL: dict) -> None:
        """Replace the weights of the model with low-rank components."""
        for layer_name in layers:
            if layer_name in LL:
                if 'model.'+layer_name in self.model_layers:
                    layer = model.get_submodule('model.'+layer_name)
                    layer.weight.copy_(LL[layer_name].to(self.device))
            else:
                print(f"Warning: Low-rank component for layer {layer_name} not found in LL dictionary.")

    def opt_remove(self,
                   model: nn.Module,
                   layers: list,
                   SS: dict) -> None:
        """Remove sparse components from the model."""
        for layer_name in layers:
            if layer_name in SS:
                if 'model.'+layer_name in self.model_layers:
                    layer = model.get_submodule('model.'+layer_name)
                    layer.weight.data -= SS[layer_name].to(self.device)
            else:
                print(f"Warning: Sparse component for layer {layer_name} not found in SS dictionary.")

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

    
    def collect_results(self):
        """
        Collect results from the lowspa model.
        Returns:
            Dictionary with evaluation results.
        """
        if self.model is not None:
            self.eval_train_results = self.eval_model(self.eval_train_results, self.train_loader) 
            self.eval_test_results = self.eval_model(self.eval_test_results, self.test_loader)


                 
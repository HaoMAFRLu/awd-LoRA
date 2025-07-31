import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import copy

from lowspa_ddp.utils import *

class CrossEvaluator():
    """
    Class for cross-evaluation of models.
    """
    def __init__(self,
                 model_type: str,
                 baseline: nn.modules=None,
                 lowspa_model: nn.modules=None,
                 data_loader: torch.utils.data.DataLoader=None,
                 LL: dict=None,
                 SS: dict=None,
                 layers: list=None,
                 energy_quantile: float=0.9) -> None:
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_type = model_type
        self.baseline = baseline.to(self.device) if baseline is not None else None
        self.lowspa_model = lowspa_model.to(self.device) if lowspa_model is not None else None
        self.lowspa_model_copy = copy.deepcopy(lowspa_model) if lowspa_model is not None else None

        self.data_loader = data_loader
        self.energy_quantile = energy_quantile
        self.LL = LL if LL is not None else {}
        self.SS = SS if SS is not None else {}
        self.layers = layers if layers is not None else []
        # with '.weight' suffix
        self.model_layers = get_model_layer_names(self.lowspa_model) if lowspa_model is not None else []
        self.eval_results = {}

        self.loss_fn = get_loss_fn(model_type)
        # Fix the data and target for evaluation
        if data_loader is not None:
            self.fix_data(data_loader)

    def fix_data(self,
                 data_loader: torch.utils.data.DataLoader) -> None:
        """Fix the data and target for evaluation for only one epoch."""
        self.data, self.target = next(iter(data_loader))

    @torch.no_grad()
    def test_opts(self) -> None:
        """
        Test the optimization options.
        This is a placeholder for testing optimization options.
        """
        self.opt_lowrank(self.baseline,
                         self.layers,
                         self.energy_quantile)
        self.opt_copy(self.lowspa_model_copy,
                      self.lowspa_model, 
                      self.layers)
        self.opt_remove(self.lowspa_model, 
                        self.layers,
                        self.SS)
        self.opt_replace(self.lowspa_model, 
                         self.layers, 
                         self.LL)
        self.opt_add(self.lowspa_model, 
                     self.layers, 
                     self.SS)
        

    @torch.no_grad()        
    def eval_baseline(self) -> dict:
        """
        Evaluate the baseline model.
        Returns:
            Dictionary with evaluation results.
        """
        if self.baseline is not None:
            self.eval_results['baseline'] = self.evaluate_one_step(self.baseline, 
                                                                   self.data, 
                                                                   self.target, 
                                                                   self.loss_fn)
            self.opt_lowrank(self.baseline, 
                             self.layers, 
                             self.energy_quantile)
            self.eval_results['baseline_lowrank'] = self.evaluate_one_step(self.baseline, 
                                                                           self.data, 
                                                                           self.target, 
                                                                           self.loss_fn)
           
    def _eval_original(self) -> dict:
        """ Evaluate the original model.
        Returns:
            Dictionary with evaluation results.
        """
        # evaluate the original model, X
        self.opt_copy(self.lowspa_model_copy,
                      self.lowspa_model, 
                      self.layers)
        return self.evaluate_one_step(self.lowspa_model, 
                                      self.data,
                                      self.target, 
                                      self.loss_fn)
   
    def _eval_orginal_without_sparsity(self) -> dict:
        """Evaluate the original model without sparsity."""
        # evaluate the original model, X - S
        self.opt_copy(self.lowspa_model_copy,
                      self.lowspa_model, 
                      self.layers)
        self.opt_remove(self.lowspa_model, 
                        self.layers,
                        self.SS)
        return self.evaluate_one_step(self.lowspa_model, 
                                      self.data, 
                                      self.target, 
                                      self.loss_fn)
         
    def _eval_original_lowrank_without_sparsity(self) -> dict:
        """Evaluate the original model with low-rank approximation without sparsity."""
        # evaluate the original model, 90% low-rank approximation of (X - S)
        self.opt_copy(self.lowspa_model_copy,
                      self.lowspa_model, 
                      self.layers)
        self.opt_remove(self.lowspa_model, 
                        self.layers,
                        self.SS)
        self.opt_lowrank(self.lowspa_model, 
                         self.layers, 
                         self.energy_quantile)
        return self.evaluate_one_step(self.lowspa_model, 
                                      self.data, 
                                      self.target, 
                                      self.loss_fn)
        
    def _eval_lowrank(self) -> dict:
        """Evaluate the low-rank model."""
        self.opt_replace(self.lowspa_model, 
                         self.layers, 
                         self.LL)
        return self.evaluate_one_step(self.lowspa_model, 
                                      self.data, 
                                      self.target, 
                                      self.loss_fn)
     
    def _eval_lowrank_lowrank(self) -> dict:
        """Evaluate the low-rank model with low-rank approximation."""
        self.opt_replace(self.lowspa_model, 
                         self.layers, 
                         self.LL)
        self.opt_lowrank(self.lowspa_model, 
                         self.layers, 
                         self.energy_quantile)
        return self.evaluate_one_step(self.lowspa_model, 
                                      self.data, 
                                      self.target, 
                                      self.loss_fn)
          
    def _eval_lowrank_sparsity(self) -> dict:
        """Evaluate the low-rank model with sparsity."""
        self.opt_replace(self.lowspa_model, 
                         self.layers, 
                         self.LL)
        self.opt_add(self.lowspa_model, 
                     self.layers, 
                     self.SS)
        return self.evaluate_one_step(self.lowspa_model, 
                                      self.data, 
                                      self.target, 
                                      self.loss_fn)

    def _eval_lowrank_lowrank_sparsity(self) -> dict:
        """Evaluate the low-rank model with low-rank approximation and sparsity."""
        self.opt_replace(self.lowspa_model, 
                         self.layers, 
                         self.LL)
        self.opt_lowrank(self.lowspa_model, 
                         self.layers, 
                         self.energy_quantile)
        self.opt_add(self.lowspa_model, 
                     self.layers, 
                     self.SS)
        return self.evaluate_one_step(self.lowspa_model, 
                                      self.data, 
                                      self.target, 
                                      self.loss_fn)

    @torch.no_grad()        
    def eval_lowspa(self) -> dict:
        """
        Evaluate the lowspa model.
        Returns:
            Dictionary with evaluation results.
        """
        if self.lowspa_model is not None:
            self.eval_results['lowspa'] = self._eval_original()
            self.eval_results['lowspa_without_sparsity'] = self._eval_orginal_without_sparsity()
            self.eval_results['lowspa_lowrank_without_sparsity'] = self._eval_original_lowrank_without_sparsity()
            self.eval_results['lowspa_lowrank'] = self._eval_lowrank()
            self.eval_results['lowspa_lowrank_lowrank'] = self._eval_lowrank_lowrank()
            self.eval_results['lowspa_lowrank_sparsity'] = self._eval_lowrank_sparsity()
            self.eval_results['lowspa_lowrank_lowrank_sparsity'] = self._eval_lowrank_lowrank_sparsity()
        
    def opt_copy(self,
                 model_source: nn.Module,
                 model_target: nn.Module,
                 layers: list) -> None:
        """Copy the weights from source model to target model for specified layers."""
        for layer_name in layers:
            if layer_name in self.model_layers:
                source_layer = model_source.get_submodule(layer_name.removesuffix('.weight'))
                target_layer = model_target.get_submodule(layer_name.removesuffix('.weight'))
                target_layer.weight.data.copy_(source_layer.weight.data)
            else:
                print(f"Warning: Layer {layer_name} not found in one of the models.")
                 
    def opt_lowrank(self, 
                    model: nn.Module, 
                    layers: list,
                    energy_quantile: float) -> None:
        """Do low-rank approximation on specified layers of the model.
        Args:
            model: The model to optimize.
            layers: List of layer names to apply low-rank approximation.
        """
        for layer_name in layers:
            if layer_name in self.model_layers:
                layer = model.get_submodule(layer_name.removesuffix('.weight'))
                weight = layer.weight.data
                U, s, V = torch.linalg.svd(weight, full_matrices=False)
                nr_singular_values = get_energy_quantile(s, quantile=energy_quantile)
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
                if layer_name in self.model_layers:
                    layer = model.get_submodule(layer_name.removesuffix('.weight'))
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
                if layer_name in self.model_layers:
                    layer = model.get_submodule(layer_name.removesuffix('.weight'))
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
                if layer_name in self.model_layers:
                    layer = model.get_submodule(layer_name.removesuffix('.weight'))
                    layer.weight.data -= SS[layer_name].to(self.device)
            else:
                print(f"Warning: Sparse component for layer {layer_name} not found in SS dictionary.")

    def evaluate_one_step(self, 
                          model: nn.Module, 
                          data,
                          target,
                          loss_fn) -> dict:
        """
        Evaluate the model on the given data loader.
        Args:
            model: The model to evaluate.
            data_loader: DataLoader for the evaluation dataset.
        Returns:
            Dictionary with evaluation results.
        """
        model.eval()
        with torch.no_grad():
            data, target = data.to(self.device), target.to(self.device)
            output = model(data)
            loss = loss_fn(output, target)  # [Batch, Length, Vocab]
            if self.model_type == 'GPT':
                pred = output.argmax(dim=-1)
                mask = target.ne(-1)
            elif self.model_type == 'CNN':
                pred = output.argmax(dim=1)
        
        correct = (pred[mask] == target[mask]).sum().item()
        total   = mask.sum().item()
        accuracy = correct / max(total, 1)
        avg_loss = loss.item()
        return {'loss': avg_loss, 'accuracy': accuracy, 'correct': correct, 'total': total}
    
    # def evaluate(self,
    #              model: nn.Module,
    #              data_loader: torch.utils.data.DataLoader,
    #              loss_fn: nn.Module) -> dict:
    #     """
    #     Evaluate the model on the given data loader.
    #     Args:
    #         model: The model to evaluate.
    #         data_loader: DataLoader for the evaluation dataset.
    #         loss_fn: Loss function to use for evaluation.
    #     Returns:
    #         Dictionary with evaluation results.
    #     """
    #     model.eval()
    #     total_loss = 0.0
    #     correct = 0
    #     total = 0

    #     for data, target in tqdm(data_loader, desc="Evaluating"):
    #         loss, correct_batch = self.evaluate_one_step(model, data, target, loss_fn)
    #         total_loss += loss.item()
    #         correct += correct_batch
    #         total += target.size(0)

    #     avg_loss = total_loss / len(data_loader.dataset)
    #     accuracy = correct / total
    #     return {'loss': avg_loss, 'accuracy': accuracy}


                 
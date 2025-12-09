"""A collection of functions for operating the blocks.
"""
import torch
import torch.nn as nn

@torch.no_grad()        
def opt_copy(model_source: nn.Module,
             model_target: nn.Module
            ) -> None:
    """Copy the weights from source model to target model for specified layers."""
    model_target.load_state_dict(model_source, strict=True)

@torch.no_grad()        
def opt_lowrank(model: nn.Module, 
                layers: list,
                rank_quantile: float,
                device: str) -> None:
    """Do low-rank approximation on specified layers of the model.
    Args:
        model: The model to optimize.
        layers: List of layer names to apply low-rank approximation.
    """
    for layer_name in layers:
        layer = model.get_submodule('model.'+layer_name)
        weight = layer.weight.data
        U, s, V = torch.linalg.svd(weight.float(), full_matrices=False)
        # nr_singular_values = get_energy_quantile(s, quantile=rank_quantile)
        nr_singular_values = int(len(s) * rank_quantile[layer_name])
        low_rank_weight = U[:, :nr_singular_values] @ torch.diag(s[:nr_singular_values]) @ V[:nr_singular_values, :]
        layer.weight.copy_(low_rank_weight.to(device))

@torch.no_grad()        
def opt_add(model: nn.Module,
            layers: list,
            SS: dict,
            device: str) -> None:
    """Add sparse components to the model."""
    for layer_name in layers:
        if layer_name in SS:
            layer = model.get_submodule('model.'+layer_name)
            layer.weight.data += SS[layer_name].to(device)
        else:
            raise ValueError(f"Sparse component for layer {layer_name} not found in SS dictionary.")

@torch.no_grad()        
def opt_replace(model: nn.Module,
                layers: list,
                LL: dict,
                device: str) -> None:
    """Replace the weights of the model with low-rank components."""
    for layer_name in layers:
        if layer_name in LL:
            layer = model.get_submodule('model.'+layer_name)
            layer.weight.copy_(LL[layer_name].to(device))
        else:
            raise ValueError(f"Low-rank component for layer {layer_name} not found in LL dictionary.")

@torch.no_grad()        
def opt_remove(model: nn.Module,
               layers: list,
               SS: dict,
               device: str) -> None:
    """Remove sparse components from the model."""
    for layer_name in layers:
        if layer_name in SS:
            layer = model.get_submodule('model.'+layer_name)
            layer.weight.data -= SS[layer_name].to(device)
        else:
            raise ValueError(f"Sparse component for layer {layer_name} not found in SS dictionary.")

def re_sparse(SS: dict, rate_density: dict) -> dict:
    """Re-sparsify the sparse components based on the target rate density.
    Args:
        SS (dict): original sparse components
        rate_density (dict): target rate density for each layer
    Returns:
        _SS (dict): re-sparsified sparse components
    """
    _SS = {}
    for key in SS:
        S = SS[key]
        S_flat = S.view(-1)
        nr_total = S_flat.shape[0]
        nr_nonzero_target = int(nr_total * rate_density[key])
        if nr_nonzero_target >= nr_total:
            _SS[key] = S
            continue
        # get the threshold
        if nr_nonzero_target == 0:
            threshold = torch.max(torch.abs(S_flat)) + 1.0  # set threshold higher than max value
        else:
            threshold = torch.topk(torch.abs(S_flat), nr_nonzero_target, largest=True).values[-1]
        S_sparse = torch.where(torch.abs(S) >= threshold, S, torch.zeros_like(S))
        _SS[key] = S_sparse
    return _SS
"""
"""
import torch
import os, sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.cnn import CNN
from models.nanoGPT import GPT, GPTConfig
from lowspa_ddp.utils import count_parameters
from datasets.dataloader import get_mnist
from datasets.GPTdataloader import get_gpt_dataloader

def get_CNN():
    """
    Get the CNN model.
    
    Returns:
        CNN: An instance of the CNN model.
    """
    return CNN()

def get_GPT(gpt_conf: GPTConfig):
    """
    Get the GPT model.
    
    Returns:
        GPT: An instance of the GPT model.
    """
    return GPT(gpt_conf)

def get_model(cfg: dict):
    """
    Get the model based on the configuration.
    """
    if cfg['name'] == 'CNN':
        return get_CNN()
    elif cfg['name'] == 'GPT':
        params = cfg.get('params', {})
        n_layer    = params.get('n_layer', 12)
        n_head     = params.get('n_head', 12)
        n_embd     = params.get('n_embd', 768)
        block_size = params.get('block_size', 1024)

        vocab_size = 50304   
        dropout    = 0.0     
        bias       = False   

        gpt_conf = GPTConfig(
            n_layer    = n_layer,
            n_head     = n_head,
            n_embd     = n_embd,
            block_size = block_size,
            vocab_size = vocab_size,
            dropout    = dropout,
            bias       = bias,
        )
        return get_GPT(gpt_conf)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

def get_dataloader(model_type: str,
                   batch_size: int = 64,
                   num_workers: int = 0):
     """
     Get the dataloader based on the model type.
     
     Args:
          model_type (str): The type of the model to get ('CNN' or 'GPT').
          batch_size (int): Batch size for the dataloader.
          num_workers (int): Number of workers for the dataloader.
     
     Returns:
          train_loader: DataLoader for the training set.
          test_loader: DataLoader for the test set.
     """
     if model_type == 'CNN':
          return get_mnist(batch_size=batch_size, 
                           num_workers=num_workers)
     elif model_type == 'GPT':
          return get_gpt_dataloader(batch_size=batch_size)
     else:
          raise ValueError(f"Unsupported model type: {model_type}")

def get_model_and_dataloader(cfg_model: dict,
                             cfg_training: dict):
    """
    Get the model and dataloader based on the model type.
    
    Args:
        model_type (str): The type of the model to get ('CNN' or 'GPT').
    
    Returns:
        model: The instantiated model.
        train_loader: DataLoader for the training set.
        test_loader: DataLoader for the test set.
    """
    model_type = cfg_model.get('name', 'CNN')
    batch_size = cfg_training.get('batch_size', 64)
    num_workers = cfg_training.get('num_workers', 0)

    model = get_model(cfg_model)
    train_loader, test_loader = get_dataloader(model_type,
                                               batch_size=batch_size, 
                                               num_workers=num_workers)
    return model, train_loader, test_loader

def get_linear_layers_name(model):
    """
    Get the names of linear layers in the model.
    
    Args:
        model: The model to get linear layer names from.
    
    Returns:
        list: A list of names of linear layers in the model.
    """
    return [name for name, module in model.named_modules() if isinstance(module, torch.nn.Linear)]

if __name__ == "__main__":
    model_type = 'GPT'
    model = get_model(model_type)
    print(model)
    print('=' * 50)
    total_params = count_parameters(model)
    names = [
    name
    for name, param in model.named_parameters()
    if param.requires_grad
    ]
    print(f"Model type: {model_type}, Total parameters: {total_params}")
    for name in names:
        print(f"Linear layer name: {name}")
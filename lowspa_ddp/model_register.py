"""
"""
import torch
import os, sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.cnn import CNN
from models.nanoGPT import GPT, GPTConfig
from lowspa_ddp.utils import count_parameters
from dataloaders.dataloader import get_mnist
from dataloaders.GPTdataloader import get_gpt_dataloader

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

def get_model(init_from: str,
              cfg: dict):
    """
    Get the model based on the configuration.
    """
    if cfg['name'] == 'CNN':
        return get_CNN()
    elif cfg['name'] == 'GPT':
        if init_from == 'scratch':  # Initialize from scratch
            params = cfg.get('params', {})
            n_layer    = params.get('n_layer', 12)
            n_head     = params.get('n_head', 12)
            n_embd     = params.get('n_embd', 768)
            block_size = params.get('block_size', 1024)
            vocab_size = params.get('vocab_size', 1024)

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
        elif init_from.startswith('gpt2'):
            # Initialize from a pre-trained GPT-2 model
            override_args = dict(dropout=0.0)
            return GPT.from_pretrained(init_from, override_args)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

def get_dataloader(model_type: str, cfg: dict):
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
            batch_size = cfg.get('batch_size', 512)
            num_workers = cfg.get('num_workers', 4)
            dataloader_type = cfg.get('type', 'train')
            train_loader, test_loader = get_mnist(batch_size=batch_size, num_workers=num_workers)
            if dataloader_type == 'train':
                return train_loader
            elif dataloader_type == 'test':
                return test_loader
     elif model_type == 'GPT':
            split = cfg.get('split', 'train')
            batch_size = cfg.get('batch_size', 64)
            block_size = cfg.get('block_size', 1024)
            dataset = cfg.get('dataset', 'openwebtext')
            steps_per_epoch = cfg.get('steps_per_epoch', None)
            tokens_per_epoch = cfg.get('tokens_per_epoch', None)
            return get_gpt_dataloader(split=split,
                                      batch_size=batch_size,
                                      block_size=block_size,
                                      dataset=dataset,
                                      steps_per_epoch=steps_per_epoch,
                                      tokens_per_epoch=tokens_per_epoch)
     else:
            raise ValueError(f"Unsupported model type: {model_type}")

def get_model_and_dataloader(cfg_model: dict,
                             cfg_dataloader: dict):
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

    model = get_model(cfg_model)
    data_loader = get_dataloader(model_type, cfg_dataloader)
    return model, data_loader

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
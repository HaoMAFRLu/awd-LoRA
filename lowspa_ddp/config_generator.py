"""
This script generates a YAML configuration file for a GPT model.
It creates ADMM parameter entries for each transformer's attention heads (c_attn and c_proj),
the final lm_head layer, and optionally the embedding layers.
"""
import os, sys
import yaml
import copy

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from lowspa_ddp.utils import get_parent_path
from lowspa_ddp.params import projection

root = get_parent_path(lvl=1)
keys = ['attn.c_attn', 'attn.c_proj', 'mlp.c_fc', 'mlp.c_proj']
proj = projection()


def generate_config(
    num_heads: int=12,
    num_layers: int=12,
    num_embd: int=768,
    num_epochs: int=6000,
    block_size: int=1024,
    batch_size: int=12,
    vocab_size: int=50304,  # Common vocabulary size for GPT models
    steps_per_epoch: int=10,
    tokens_per_epoch: int=None,
    dataset: str='openwebtext',  # or 'openwebtext', etc.
    include_embeddings: bool=True,
    output_path: str="config.yaml"
):
    """
    Generate a YAML configuration for GPT model ADMM settings.

    Args:
        num_layers: Number of transformer blocks (default: 12).
        include_embeddings: Whether to include embedding layers in the config.
        output_path: Path to save the generated YAML file.
    """
    
    # Base configuration structure
    cfg = {
        'seed': 42,
        'num_epochs': num_epochs,
        'model': {
            'name': 'GPT',
            'params': {
                'n_layer':    num_layers,      # Number of transformer blocks
                'n_head':     num_heads,  # Number of attention heads
                'n_embd':     num_embd,    # Embedding dimension
                'vocab_size': vocab_size,  # Vocabulary size
                'block_size': block_size,  # Context window size
            }
        },
        'dataloader': {
            'split': 'train',
            'batch_size': batch_size,
            'num_workers': 4,
            'block_size': block_size,
            'steps_per_epoch': steps_per_epoch,
            'tokens_per_epoch': tokens_per_epoch,
            'dataset': dataset
        },
        'scheduler': {
            'name': 'StepLR',
            'params': {
                'step_size': 5,
                'gamma': 0.85,
            }
        },
        'optimizer': {
            'name': 'AdamW',
            'params': {
                'lr':           0.0006,
                'weight_decay': 0.1,
                'betas':       [0.9, 0.95]
            }
        },
        'layers': []
    }

    # Optionally include embedding layers
    if include_embeddings:
        cfg['layers'].append({
            'name': 'transformer.wte.weight',
            'params': copy.deepcopy(proj['wte'])
        })
        cfg['layers'].append({
            'name': 'transformer.wpe.weight',
            'params': copy.deepcopy(proj['wpe'])
        })

    # Add c_attn and c_proj for each transformer block
    for i in range(num_heads):
        base = f"transformer.h.{i}"
        for key in keys:    
            if key in proj:
                cfg['layers'].append({
                    'name': f"{base}.{key}.weight",
                    'params': copy.deepcopy(proj[key])
                })

    # Add final language model head
    # cfg['layers'].append({
    #     'name': 'transformer.ln_f.weight',
    #     'params': copy.deepcopy(proj['lm_head'])
    # })

    # Define a dumper class that suppresses aliases and customizes float formatting
    class NoAliasDumper(yaml.SafeDumper):
        def ignore_aliases(self, data):
            return True

    # Represent None as null
    def represent_none(dumper, data):
        return dumper.represent_scalar('tag:yaml.org,2002:null', 'null')
    NoAliasDumper.add_representer(type(None), represent_none)

    # Represent float values in fixed decimal format without scientific notation
    def represent_float(dumper, value):
        text = f"{value:.6f}".rstrip('0').rstrip('.')
        return dumper.represent_scalar('tag:yaml.org,2002:float', text)
    NoAliasDumper.add_representer(float, represent_float)

    # Write configuration to file
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(
            cfg,
            f,
            Dumper=NoAliasDumper,
            sort_keys=False,
            allow_unicode=True
        )

    print(f"Configuration written to {output_path}")


if __name__ == "__main__":
    # Customize parameters as needed
    NUM_HEADS = 2
    NUM_LAYERS = 2
    NUM_EMBEDDING = 768
    NUM_EPOCHS = 60000
    BLOCK_SIZE = 512
    BATCH_SIZE = 12
    VOCAB_SIZE = 50304  # Common vocabulary size for GPT models
    STEPS_PER_EPOCH = 100
    TOKENS_PER_EPOCH = None
    DATASET = 'shakespeare'  # or 'openwebtext', etc.
    INCLUDE_EMBEDDINGS = False
    OUTPUT_FILE = "GPTshakespeare.yaml"
    OUTPUT_PATH = os.path.join(root, 'lowspa_ddp', 'configs', OUTPUT_FILE)

    generate_config(
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        num_embd=NUM_EMBEDDING,
        num_epochs=NUM_EPOCHS,
        block_size=BLOCK_SIZE,
        batch_size=BATCH_SIZE,
        vocab_size=VOCAB_SIZE,
        steps_per_epoch=STEPS_PER_EPOCH,
        tokens_per_epoch=TOKENS_PER_EPOCH,
        dataset=DATASET,
        include_embeddings=INCLUDE_EMBEDDINGS,
        output_path=OUTPUT_PATH
    )

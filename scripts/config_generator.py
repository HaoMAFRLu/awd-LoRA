"""Generate training YAML files from fixed model architecture configs."""
import os, sys
import yaml
import copy
import json

from params import projection

root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
keys = ['self_attn.o_proj', 'self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'mlp.gate_proj', 'mlp.down_proj', 'mlp.up_proj']
proj = projection()


def generate_config(
    name: str = 'llama_9m',
    model_config: str = None,
    seed: int = 42,
    training_mode: str='salad',
    lr: float = 0.008,
    num_freq: int = 1000,
    weight_decay: float=0.0,
    optimizer_name: str='Adam',
    gradient: str='coupled',
    is_asyn: bool = False,
    is_init: bool = False,
    is_wandb: bool = False,
    is_monitor: bool = False,
    save_interval: int = 50,
    seed_for_shuffle: int = 42,
    is_clip: float = 1.0,
    num_total_iters: int=20_000,
    max_length: int=256,
    batch_size: int=32,
    warmup_steps: int=1000,
    num_workers: int=4,
    scheduler_type: str='cosine',
    min_lr_ratio: float=0.1,
    include_embeddings: bool=False,
    include_head: bool=False,
    output_path: str=None):
    """
    Generate a YAML configuration for training and SALAAD settings.
    """
    if model_config is None:
        model_config = f"{name}_model.json"

    path_model_config = os.path.join(root, 'configs', model_config)
    with open(path_model_config, "r", encoding="utf-8") as f:
        cfg_model = json.load(f)

    num_hidden_layers = cfg_model["num_hidden_layers"]
    
    # Base configuration structure
    cfg = {
        'seed': seed,
        'name': name,
        'model_config': model_config,
        'training_mode': training_mode,
        'num_total_iters': num_total_iters,
        'num_freq': num_freq,
        'gradient': gradient,
        'is_asyn': is_asyn,
        'is_init': is_init,
        'is_wandb': is_wandb,
        'is_monitor': is_monitor,
        'save_interval': save_interval,
        'is_clip': is_clip,
        'max_length': max_length,
        'seed_for_shuffle': seed_for_shuffle,
        'batch_size': batch_size,
        'num_workers': num_workers,
        'scheduler': {
            'name': scheduler_type,
            'params': {
                'warmup_steps': warmup_steps,
                'min_lr_ratio': min_lr_ratio
            }
        },
        'optimizer': {
            'name': optimizer_name,
            'params': {
                'lr':  lr,
                'betas': (0.9, 0.95),
                'eps':  1e-8,
                'weight_decay': weight_decay,
            }
        },
        'layers': []
    }

    # Optionally include embedding layers
    if include_embeddings:
        cfg['layers'].append({
            'name': 'embed_tokens',
            'params': copy.deepcopy(proj['embed'])
        })
    
    if include_head:
        cfg['layers'].append({
            'name': 'lm_head',
            'params': copy.deepcopy(proj['lm_head'])
        })

    # Add c_attn and c_proj for each transformer block
    for i in range(num_hidden_layers):
        base = f"layers.{i}"
        for key in keys:    
            if key in proj:
                cfg['layers'].append({
                    'name': f"{base}.{key}",
                    'params': copy.deepcopy(proj[key])
                })

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
        text = f"{value:.12f}".rstrip('0').rstrip('.')
        return dumper.represent_scalar('tag:yaml.org,2002:float', text)
    NoAliasDumper.add_representer(float, represent_float)

    # Write configuration to file
    if output_path is None:
        output_path=os.path.join(root, 'configs', name+'.yaml')
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
    cfg_llama_1b = dict(
        name='llama_1b',
        model_config='llama_1b_model.json',
        seed=42,
        training_mode='salad',  # salad or vanilla
        lr=0.0005,
        gradient='coupled',  # or decoupled
        is_asyn=False,
        is_init=False,
        is_wandb=True,
        is_monitor=True,
        save_interval=1,
        min_lr_ratio=0.1,
        weight_decay=0.0,
        optimizer_name='AdamW',
        num_freq=5,
        seed_for_shuffle=42,
        num_total_iters=140000,  # 254000
        batch_size=468,
        max_length=256,
        warmup_steps=2200,
        num_workers=0,
        scheduler_type='cosine',
        include_embeddings=True,
        include_head=False,
        is_clip=1.0)
    
    cfg_llama_350m = dict(
        name='llama_350m',
        model_config='llama_350m_model.json',
        seed=42,
        training_mode='salad',  # or salad
        lr=0.001,
        gradient='coupled',  # or decoupled
        is_asyn=False,
        is_init=False,
        is_wandb=True,
        is_monitor=True,
        save_interval=1,
        min_lr_ratio=0.1,
        weight_decay=0.0,
        optimizer_name='AdamW',
        num_freq=25,
        seed_for_shuffle=42,
        num_total_iters=62250,
        batch_size=512,
        max_length=256,
        warmup_steps=2200,
        num_workers=0,
        scheduler_type='cosine',
        include_embeddings=True,
        include_head=False,
        is_clip=1.0)
    
    # Customize parameters as needed
    cfg_llama_130m = dict(
        name='llama_130m',
        model_config='llama_130m_model.json',
        seed=42,
        training_mode='salad',  # or salad
        lr=0.003,
        gradient='coupled',  # or decoupled
        is_asyn=False,
        is_init=False,
        is_wandb=True,
        is_monitor=True,
        save_interval=50,
        min_lr_ratio=0.1,
        weight_decay=0.0,
        optimizer_name='AdamW',
        num_freq=10,
        seed_for_shuffle=42,
        num_total_iters=22000,
        batch_size=512,
        max_length=256,
        warmup_steps=2200,
        num_workers=0,
        scheduler_type='cosine',
        include_embeddings=True,
        include_head=False,
        is_clip=1.0)
    
    cfg_llama_60m = dict(
        name='llama_60m',
        model_config='llama_60m_model.json',
        seed=42,
        training_mode='salad',  # or salad
        lr=0.003,
        is_wandb=True,
        is_monitor=True,
        save_interval=20,
        gradient='coupled',  # or decoupled
        is_asyn=False,
        is_init=False,
        optimizer_name='AdamW',
        min_lr_ratio=0.1,
        weight_decay=0.0,
        num_freq=10,
        seed_for_shuffle=42,
        num_total_iters=11000,
        batch_size=512,
        max_length=256,
        warmup_steps=2200,
        num_workers=0,
        scheduler_type='cosine',
        include_embeddings=True,
        include_head=True,
        is_clip=1.0)

    cfg_llama_9m = dict(
        name='llama_9m',
        model_config='llama_9m_model.json',
        training_mode='salad',  # or salad
        seed=42,
        lr=0.008,
        num_freq=2,
        gradient='coupled',  # or decoupled
        is_asyn=False,
        is_init=False,
        is_wandb=True,
        is_monitor=True,
        optimizer_name='AdamW',
        weight_decay=0.0,
        save_interval=2,
        seed_for_shuffle=42,
        num_total_iters=100,
        batch_size=2,
        max_length=256,
        warmup_steps=1000,
        num_workers=0,
        scheduler_type='cosine',
        min_lr_ratio=0.1,
        include_embeddings=True,
        include_head=True,
        is_clip=1.0)

    generate_config(**cfg_llama_350m)

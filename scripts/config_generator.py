"""This script is used to generate json file
for llama model.
"""
import os, sys
import yaml
import copy
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from salad.utils import get_parent_path
from params import projection

root = get_parent_path(lvl=1)
keys = ['self_attn.o_proj', 'self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'mlp.gate_proj', 'mlp.down_proj', 'mlp.up_proj']
proj = projection()


def generate_config(
    name: str = 'llama_9m',
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
    bos_token_id: int=0,
    eos_token_id: int=1,
    hidden_act: str='silu',
    hidden_size: int=512,
    intermediate_size: int=1376,
    initializer_range: float=0.02,
    max_sequence_length: int=1024, 
    model_type: str="llama",
    num_attention_heads: int=8,
    num_hidden_layers: int=8,
    pad_token_id: int=-1,
    rms_norm_eps: float=1e-06,
    transformers_version: str="4.28.1",
    use_cache: bool=True,
    vocab_size: int=32000,  # Common vocabulary size for GPT models
    output_path: str=None):
    """
    Generate a YAML configuration for GPT model ADMM settings.

    Args:
        num_layers: Number of transformer blocks (default: 12).
        include_embeddings: Whether to include embedding layers in the config.
        output_path: Path to save the generated YAML file.
    """
    cfg_model = {
        "architectures": [
            "LLaMAForCausalLM"
        ],
        "bos_token_id": bos_token_id,
        "eos_token_id": eos_token_id,
        "hidden_act": hidden_act,
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "initializer_range": initializer_range,
        "max_sequence_length": max_sequence_length,
        "model_type": model_type,
        "num_attention_heads": num_attention_heads,
        "num_hidden_layers": num_hidden_layers,
        "pad_token_id": pad_token_id,
        "rms_norm_eps": rms_norm_eps,
        "transformers_version": transformers_version,
        "use_cache": use_cache,
        "vocab_size": vocab_size}
    
    # Base configuration structure
    cfg = {
        'seed': seed,
        'name': name,
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
    output_path=os.path.join(root, 'scripts', 'configs', name+'.yaml')
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(
            cfg,
            f,
            Dumper=NoAliasDumper,
            sort_keys=False,
            allow_unicode=True
        )

    output_path=os.path.join(root, 'scripts', 'configs', name+'_model.json')
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(cfg_model, f, ensure_ascii=False, indent=4)

    print(f"Configuration written to {output_path}")

if __name__ == "__main__":
    cfg_llama_1b = dict(
        name='llama_1b',
        seed=42,
        training_mode='vanilla',  # or salad
        lr=0.0005,
        gradient='coupled',  # or decoupled
        is_asyn=False,
        is_init=False,
        is_wandb=True,
        is_monitor=True,
        save_interval=50,
        min_lr_ratio=0.1,
        weight_decay=0.0,
        optimizer_name='AdamW',
        num_freq=40,
        seed_for_shuffle=42,
        num_total_iters=254000,  # 254000
        batch_size=256,
        max_length=256,
        warmup_steps=2200,
        num_workers=0,
        scheduler_type='cosine',
        include_embeddings=True,
        include_head=False,
        is_clip=1.0,
        bos_token_id=0,
        eos_token_id=1,
        hidden_act='silu',
        hidden_size=2048,
        intermediate_size=5461,
        initializer_range=0.02,
        max_sequence_length=1024,
        model_type="llama",
        num_attention_heads=32,
        num_hidden_layers=24,
        pad_token_id=-1,
        rms_norm_eps=1e-06,
        transformers_version="4.28.1",
        use_cache=True,
        vocab_size=32000)
    
    cfg_llama_350m = dict(
        name='llama_350m',
        seed=42,
        training_mode='vanilla',  # or salad
        lr=0.001,
        gradient='coupled',  # or decoupled
        is_asyn=False,
        is_init=False,
        is_wandb=True,
        is_monitor=True,
        save_interval=50,
        min_lr_ratio=0.1,
        weight_decay=0.0,
        optimizer_name='AdamW',
        num_freq=40,
        seed_for_shuffle=42,
        num_total_iters=124000,
        batch_size=256,
        max_length=256,
        warmup_steps=2200,
        num_workers=0,
        scheduler_type='cosine',
        include_embeddings=False,
        include_head=False,
        is_clip=1.0,
        bos_token_id=0,
        eos_token_id=1,
        hidden_act='silu',
        hidden_size=1024,
        intermediate_size=2736,
        initializer_range=0.02,
        max_sequence_length=1024,
        model_type="llama",
        num_attention_heads=16,
        num_hidden_layers=24,
        pad_token_id=-1,
        rms_norm_eps=1e-06,
        transformers_version="4.28.1",
        use_cache=True,
        vocab_size=32000)
    
    # Customize parameters as needed
    cfg_llama_130m = dict(
        name='llama_130m',
        seed=42,
        training_mode='vanilla',  # or salad
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
        num_freq=20,
        seed_for_shuffle=42,
        num_total_iters=22000,
        batch_size=512,
        max_length=256,
        warmup_steps=2200,
        num_workers=0,
        scheduler_type='cosine',
        include_embeddings=False,
        include_head=False,
        is_clip=1.0,
        bos_token_id=0,
        eos_token_id=1,
        hidden_act='silu',
        hidden_size=768,
        intermediate_size=2048,
        initializer_range=0.02,
        max_sequence_length=1024,
        model_type="llama",
        num_attention_heads=12,
        num_hidden_layers=12,
        pad_token_id=-1,
        rms_norm_eps=1e-06,
        transformers_version="4.28.1",
        use_cache=True,
        vocab_size=32000)
    
    cfg_llama_60m = dict(
        name='llama_60m',
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
        include_embeddings=False,
        include_head=False,
        is_clip=1.0,
        bos_token_id=0,
        eos_token_id=1,
        hidden_act='silu',
        hidden_size=512,
        intermediate_size=1376,
        initializer_range=0.02,
        max_sequence_length=1024,
        model_type="llama",
        num_attention_heads=8,
        num_hidden_layers=8,
        pad_token_id=-1,
        rms_norm_eps=1e-06,
        transformers_version="4.28.1",
        use_cache=True,
        vocab_size=32000)

    cfg_llama_9m = dict(
        name='llama_9m',
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
        include_head=False,
        is_clip=1.0,
        bos_token_id=0,
        eos_token_id=1,
        hidden_act='silu',
        hidden_size=128,
        intermediate_size=352,
        initializer_range=0.02,
        max_sequence_length=1024,
        model_type="llama",
        num_attention_heads=4,
        num_hidden_layers=4,
        pad_token_id=-1,
        rms_norm_eps=1e-06,
        transformers_version="4.28.1",
        use_cache=True,
        vocab_size=32000)

    generate_config(**cfg_llama_350m)

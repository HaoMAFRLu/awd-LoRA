"""Per-layer SALAAD parameters for DINO ViT-B/8.

The DINO implementation stores Q, K, and V in one ``attn.qkv`` linear
matrix, so that matrix has one shared SALAAD parameter dictionary.
"""


attn_qkv_params = {
    "energy": 0.999,
    "init_energy": 0.15,
    "is_init": False,
    "iter_max": 1,
    "tol": 0.001,
    "rate_rank": 0.15,
    "rate_sparsity": 0.05,
    "alpha_dict": {
        "init": 0.0,
        "mode": "adaptive",
        "rate_decay": 0.2,
        "drate": 0.01,
    },
    "beta_dict": {
        "init": 0.0,
        "mode": "adaptive",
        "rate_decay": 0.003,
        "drate": 0.01,
    },
    "rho_dict": {
        "rho": 1e-5,
        "mode": "fixed",
        "start_epoch": 2,
        "coeff_rho": 0.1,
        "coeff_rho_min": 0.01,
        "coeff_rho_max": 1500.0,
        "rho_rate": 1.0,
    },
}


attn_proj_params = {
    "energy": 0.999,
    "init_energy": 0.15,
    "is_init": False,
    "iter_max": 1,
    "tol": 0.001,
    "rate_rank": 0.15,
    "rate_sparsity": 0.05,
    "alpha_dict": {
        "init": 0.0,
        "mode": "adaptive",
        "rate_decay": 0.2,
        "drate": 0.01,
    },
    "beta_dict": {
        "init": 0.0,
        "mode": "adaptive",
        "rate_decay": 0.003,
        "drate": 0.01,
    },
    "rho_dict": {
        "rho": 1e-5,
        "mode": "fixed",
        "start_epoch": 2,
        "coeff_rho": 0.1,
        "coeff_rho_min": 0.01,
        "coeff_rho_max": 1500.0,
        "rho_rate": 1.0,
    },
}


mlp_fc1_params = {
    "energy": 0.999,
    "init_energy": 0.35,
    "is_init": False,
    "iter_max": 1,
    "tol": 0.001,
    "rate_rank": 0.15,
    "rate_sparsity": 0.05,
    "alpha_dict": {
        "init": 0.0,
        "mode": "adaptive",
        "rate_decay": 0.2,
        "drate": 0.01,
    },
    "beta_dict": {
        "init": 0.0,
        "mode": "adaptive",
        "rate_decay": 0.003,
        "drate": 0.01,
    },
    "rho_dict": {
        "rho": 1e-5,
        "mode": "fixed",
        "start_epoch": 2,
        "coeff_rho": 0.1,
        "coeff_rho_min": 0.01,
        "coeff_rho_max": 1500.0,
        "rho_rate": 1.0,
    },
}


mlp_fc2_params = {
    "energy": 0.999,
    "init_energy": 0.35,
    "is_init": False,
    "iter_max": 1,
    "tol": 0.001,
    "rate_rank": 0.15,
    "rate_sparsity": 0.05,
    "alpha_dict": {
        "init": 0.0,
        "mode": "adaptive",
        "rate_decay": 0.2,
        "drate": 0.01,
    },
    "beta_dict": {
        "init": 0.0,
        "mode": "adaptive",
        "rate_decay": 0.003,
        "drate": 0.01,
    },
    "rho_dict": {
        "rho": 1e-5,
        "mode": "fixed",
        "start_epoch": 2,
        "coeff_rho": 0.1,
        "coeff_rho_min": 0.01,
        "coeff_rho_max": 1500.0,
        "rho_rate": 1.0,
    },
}


def projection():
    """Return ViT module suffixes and their SALAAD parameter templates."""
    return {
        "attn.qkv": attn_qkv_params,
        "attn.proj": attn_proj_params,
        "mlp.fc1": mlp_fc1_params,
        "mlp.fc2": mlp_fc2_params,
    }

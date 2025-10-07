embed_params = {
    'energy':        0.999,   # Energy parameter
    'init_energy':   0.45,   # Initial energy
    'is_init':       False,  # Whether to initialize
    'iter_max':      1,     # Max ADMM iterations per layer
    'tol':           0.001, # Convergence tolerance
    'rate_rank':     0.15,  # Rate of rank reduction for the layer
    'rate_sparsity': 0.15,   # Rate of sparsity for the layer
    'alpha_dict': {
        'init': 1e-7,
        'mode': 'adaptive',  # 'fixed', 'adaptive' or 'hard_cut'
        'rate_decay': 0.03,
    },
    'beta_dict': {
        'init': 1e-7,
        'mode': 'hard_cut',  # 'fixed', 'adaptive' or 'hard_cut
        'rate_decay': 0.02,
    },
    'rho_dict':            {
        'rho':           1e-5,
        'mode':          'fixed',  # 'fixed', 'shape_dependent', 'adaptive'
        'start_epoch':   2,
        'coeff_rho':     0.1,
        'coeff_rho_min': 0.01,
        'coeff_rho_max': 1500.0,
        'rho_rate':      1.0,
    }, 
}

lm_head_params = {
    'energy':        0.999,     # Energy parameter
    'init_energy':   0.15,     # Initial energy
    'is_init':       False,  # Whether to initialize
    'iter_max':      1,       # Max ADMM iterations per layer
    'tol':           0.001,   # Convergence tolerance
    'rate_rank':     0.15,  # Rate of rank reduction for the layer
    'rate_sparsity': 0.15,  # Rate of sparsity for the layer
    'alpha_dict': {
        'init': 1e-7,
        'mode': 'adaptive',  # 'fixed', 'adaptive' or 'hard_cut'
        'rate_decay': 0.02,
    },
    'beta_dict': {
        'init': 1e-7,
        'mode': 'hard_cut',  # 'fixed', 'adaptive' or 'hard_cut
        'rate_decay': 0.02,
    },
    'rho_dict':            {
        'rho':           1e-5,
        'mode':          'fixed',  # 'fixed', 'shape_dependent', 'adaptive'
        'start_epoch':   2,
        'coeff_rho':     0.1,
        'coeff_rho_min': 0.01,
        'coeff_rho_max': 1500.0,
        'rho_rate':      1.0,
    }, 
}

attn_o_proj_params = {
    'energy':        0.999,     # Energy parameter
    'init_energy':   0.15,     # Initial energy
    'is_init':       False,  # Whether to initialize
    'iter_max':      1,       # Max ADMM iterations per layer
    'tol':           0.001,   # Convergence tolerance
    'rate_rank':     0.15,  # Rate of rank reduction for the layer
    'rate_sparsity': 0.05,  # Rate of sparsity for the layer
    'alpha_dict': {
        'init': 1e-7,
        'mode': 'adaptive',  # 'fixed', 'adaptive' or 'hard_cut'
        'rate_decay': 0.02,
    },
    'beta_dict': {
        'init': 1e-7,
        'mode': 'adaptive',  # 'fixed', 'adaptive' or 'hard_cut
        'rate_decay': 0.02,
    },
    'rho_dict':            {
        'rho':           1e-5,
        'mode':          'fixed',  # 'fixed', 'shape_dependent', 'adaptive'
        'start_epoch':   2,
        'coeff_rho':     0.1,
        'coeff_rho_min': 0.01,
        'coeff_rho_max': 1500.0,
        'rho_rate':      1.0,
    }, 
}

attn_q_proj_params = {
    'energy':        0.999,     # Energy parameter
    'init_energy':   0.15,     # Initial energy
    'is_init':       False,  # Whether to initialize
    'iter_max':      1,       # Max ADMM iterations per layer
    'tol':           0.001,   # Convergence tolerance
    'rate_rank':     0.15,  # Rate of rank reduction for the layer
    'rate_sparsity': 0.05,  # Rate of sparsity for the layer
    'alpha_dict': {
        'init': 1e-7,
        'mode': 'adaptive',  # 'fixed', 'adaptive' or 'hard_cut'
        'rate_decay': 0.02,
    },
    'beta_dict': {
        'init': 1e-7,
        'mode': 'adaptive',  # 'fixed', 'adaptive' or 'hard_cut
        'rate_decay': 0.02,
    },
    'rho_dict':            {
        'rho':           1e-5,
        'mode':          'fixed',  # 'fixed', 'shape_dependent', 'adaptive'
        'start_epoch':   2,
        'coeff_rho':     0.1,
        'coeff_rho_min': 0.01,
        'coeff_rho_max': 1500.0,
        'rho_rate':      1.0,
    }, 
}

attn_k_proj_params = {
    'energy':        0.999,     # Energy parameter
    'init_energy':   0.15,     # Initial energy
    'is_init':       False,  # Whether to initialize
    'iter_max':      1,       # Max ADMM iterations per layer
    'tol':           0.001,   # Convergence tolerance
    'rate_rank':     0.15,  # Rate of rank reduction for the layer
    'rate_sparsity': 0.05,  # Rate of sparsity for the layer
    'alpha_dict': {
        'init': 1e-7,
        'mode': 'adaptive',  # 'fixed', 'adaptive' or 'hard_cut'
        'rate_decay': 0.02,
    },
    'beta_dict': {
        'init': 1e-7,
        'mode': 'adaptive',  # 'fixed', 'adaptive' or 'hard_cut
        'rate_decay': 0.02,
    },
    'rho_dict':            {
        'rho':           1e-5,
        'mode':          'fixed',  # 'fixed', 'shape_dependent', 'adaptive'
        'start_epoch':   2,
        'coeff_rho':     0.1,
        'coeff_rho_min': 0.01,
        'coeff_rho_max': 1500.0,
        'rho_rate':      1.0,
    }, 
}

attn_v_proj_params = {
    'energy':        0.999,     # Energy parameter
    'init_energy':   0.15,     # Initial energy
    'is_init':       False,  # Whether to initialize
    'iter_max':      1,       # Max ADMM iterations per layer
    'tol':           0.001,   # Convergence tolerance
    'rate_rank':     0.15,  # Rate of rank reduction for the layer
    'rate_sparsity': 0.05,  # Rate of sparsity for the layer
    'alpha_dict': {
        'init': 1e-7,
        'mode': 'adaptive',  # 'fixed', 'adaptive' or 'hard_cut'
        'rate_decay': 0.02,
    },
    'beta_dict': {
        'init': 1e-7,
        'mode': 'adaptive',  # 'fixed', 'adaptive' or 'hard_cut
        'rate_decay': 0.02,
    },
    'rho_dict':            {
        'rho':           1e-5,
        'mode':          'fixed',  # 'fixed', 'shape_dependent', 'adaptive'
        'start_epoch':   2,
        'coeff_rho':     0.1,
        'coeff_rho_min': 0.01,
        'coeff_rho_max': 1500.0,
        'rho_rate':      1.0,
    }, 
}

mlp_gate_proj_params = {
    'energy':        0.999,     # Energy parameter
    'init_energy':   0.35,     # Initial energy
    'is_init':       False,  # Whether to initialize
    'iter_max':      1,       # Max ADMM iterations per layer
    'tol':           0.001,   # Convergence tolerance
    'rate_rank':     0.25,  # Rate of rank reduction for the layer
    'rate_sparsity': 0.05,  # Rate of sparsity for the layer
    'alpha_dict': {
        'init': 1e-7,
        'mode': 'adaptive',  # 'fixed', 'adaptive' or 'hard_cut'
        'rate_decay': 0.02,
    },
    'beta_dict': {
        'init': 1e-7,
        'mode': 'adaptive',  # 'fixed', 'adaptive' or 'hard_cut
        'rate_decay': 0.02,
    },
    'rho_dict':            {
        'rho':           1e-5,
        'mode':          'fixed',  # 'fixed', 'shape_dependent', 'adaptive'
        'start_epoch':   2,
        'coeff_rho':     0.1,
        'coeff_rho_min': 0.01,
        'coeff_rho_max': 1500.0,
        'rho_rate':      1.0,
    }, 
}

mlp_down_proj_params = {
    'energy':         0.999,     # Energy parameter
    'init_energy':    0.35,     # Initial energy
    'is_init':        False,  # Whether to initialize
    'iter_max':       1,       # Max ADMM iterations per layer
    'tol':            0.001,   # Convergence tolerance
    'rate_rank':      0.25,  # Rate of rank reduction for the layer
    'rate_sparsity':  0.05,  # Rate of sparsity for the layer
    'alpha_dict': {
        'init': 1e-7,
        'mode': 'adaptive',  # 'fixed', 'adaptive' or 'hard_cut'
        'rate_decay': 0.02,
    },
    'beta_dict': {
        'init': 1e-7,
        'mode': 'adaptive',  # 'fixed', 'adaptive' or 'hard_cut
        'rate_decay': 0.02,
    },
    'rho_dict':            {
        'rho':           1e-5,
        'mode':          'fixed',  # 'fixed', 'shape_dependent', 'adaptive'
        'start_epoch':   2,
        'coeff_rho':     0.1,
        'coeff_rho_min': 0.01,
        'coeff_rho_max': 1500.0,
        'rho_rate':      1.0,
    }, 
}

mlp_up_proj_params = {
    'energy':        0.999,     # Energy parameter
    'init_energy':   0.35,     # Initial energy
    'is_init':       False,  # Whether to initialize
    'iter_max':      1,       # Max ADMM iterations per layer
    'tol':           0.001,   # Convergence tolerance
    'rate_rank':     0.25,  # Rate of rank reduction for the layer
    'rate_sparsity': 0.05,  # Rate of sparsity for the layer
    'alpha_dict': {
        'init': 1e-7,
        'mode': 'adaptive',  # 'fixed', 'adaptive' or 'hard_cut'
        'rate_decay': 0.02,
    },
    'beta_dict': {
        'init': 1e-7,
        'mode': 'adaptive',  # 'fixed', 'adaptive' or 'hard_cut
        'rate_decay': 0.02,
    },
    'rho_dict':            {
        'rho':           1e-5,
        'mode':          'fixed',  # 'fixed', 'shape_dependent', 'adaptive'
        'start_epoch':   2,
        'coeff_rho':     0.1,
        'coeff_rho_min': 0.01,
        'coeff_rho_max': 1500.0,
        'rho_rate':      1.0,
    }, 
}

def projection():
    return {'self_attn.o_proj': attn_o_proj_params,
            'self_attn.q_proj': attn_q_proj_params,
            'self_attn.k_proj': attn_k_proj_params,
            'self_attn.v_proj': attn_v_proj_params,
            'mlp.gate_proj': mlp_gate_proj_params,
            'mlp.down_proj': mlp_down_proj_params,
            'mlp.up_proj': mlp_up_proj_params,
            'embed': embed_params,
            'lm_head': lm_head_params}
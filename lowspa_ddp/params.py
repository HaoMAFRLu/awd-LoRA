embedding_wte_params = {
    'alpha':        0.0001,  # Regularization parameter
    'beta':         0.00001, # Sparsity parameter
    'rho':          0.001,   # Penalty parameter
    'energy':       0.9,     # Energy parameter
    'init_energy':  0.3,     # Initial energy
    'iter_max':     1,       # Max ADMM iterations per layer
    'tol':          0.001,   # Convergence tolerance
    'loss_version': 'v1',    # Loss version (v1: X-L-S-Y/rho, v2: X-L)
    'is_cal':       True,    # Whether to calculate sparse matrix
}

embedding_wpe_params = {
    'alpha':        0.0001,  # Regularization parameter
    'beta':         0.00001, # Sparsity parameter
    'rho':          0.001,   # Penalty parameter
    'energy':       0.9,     # Energy parameter
    'init_energy':  0.3,     # Initial energy
    'iter_max':     1,       # Max ADMM iterations per layer
    'tol':          0.001,   # Convergence tolerance
    'loss_version': 'v1',    # Loss version (v1: X-L-S-Y/rho, v2: X-L)
    'is_cal':       True,    # Whether to calculate sparse matrix
}

lm_head_params = {
    'alpha':        0.0001,  # Regularization parameter
    'beta':         0.00001, # Sparsity parameter
    'rho':          0.001,   # Penalty parameter
    'energy':       0.9,     # Energy parameter
    'init_energy':  0.3,     # Initial energy
    'iter_max':     1,       # Max ADMM iterations per layer
    'tol':          0.001,   # Convergence tolerance
    'loss_version': 'v1',    # Loss version (v1: X-L-S-Y/rho, v2: X-L)
    'is_cal':       True,    # Whether to calculate sparse matrix
}

attn_c_attn_params = {
    'alpha':        0.0001,  # Regularization parameter
    'beta':         0.00001, # Sparsity parameter
    'rho':          0.001,   # Penalty parameter
    'energy':       0.9,     # Energy parameter
    'init_energy':  0.3,     # Initial energy
    'iter_max':     1,       # Max ADMM iterations per layer
    'tol':          0.001,   # Convergence tolerance
    'loss_version': 'v1',    # Loss version (v1: X-L-S-Y/rho, v2: X-L)
    'is_ca':        True,    # Whether to calculate sparse matrix
}

attn_c_attn_params = {
    'alpha':        0.0001,  # Regularization parameter
    'beta':         0.00001, # Sparsity parameter
    'rho':          0.001,   # Penalty parameter
    'energy':       0.9,     # Energy parameter
    'init_energy':  0.3,     # Initial energy
    'iter_max':     1,       # Max ADMM iterations per layer
    'tol':          0.001,   # Convergence tolerance
    'loss_version': 'v1',    # Loss version (v1: X-L-S-Y/rho, v2: X-L)
    'is_cal':       True,    # Whether to calculate sparse matrix
}

attn_c_proj_params = {
    'alpha':        0.0001,  # Regularization parameter
    'beta':         0.00001, # Sparsity parameter
    'rho':          0.001,   # Penalty parameter
    'energy':       0.9,     # Energy parameter
    'init_energy':  0.3,     # Initial energy
    'iter_max':     1,       # Max ADMM iterations per layer
    'tol':          0.001,   # Convergence tolerance
    'loss_version': 'v1',    # Loss version (v1: X-L-S-Y/rho, v2: X-L)
    'is_cal':       True,    # Whether to calculate sparse matrix
}

mlp_c_fc_params = {
    'alpha':        0.0001,  # Regularization parameter
    'beta':         0.00001, # Sparsity parameter
    'rho':          0.001,   # Penalty parameter
    'energy':       0.9,     # Energy parameter
    'init_energy':  0.3,     # Initial energy
    'iter_max':     1,       # Max ADMM iterations per layer
    'tol':          0.001,   # Convergence tolerance
    'loss_version': 'v1',    # Loss version (v1: X-L-S-Y/rho, v2: X-L)
    'is_cal':       True,    # Whether to calculate sparse matrix
}

mlp_c_proj_params = {
    'alpha':        0.0001,  # Regularization parameter
    'beta':         0.00001, # Sparsity parameter
    'rho':          0.001,   # Penalty parameter
    'energy':       0.9,     # Energy parameter
    'init_energy':  0.3,     # Initial energy
    'iter_max':     1,       # Max ADMM iterations per layer
    'tol':          0.001,   # Convergence tolerance
    'loss_version': 'v1',    # Loss version (v1: X-L-S-Y/rho, v2: X-L)
    'is_cal':       True,    # Whether to calculate sparse matrix
}

def projection():
    return {'attn.c_attn': attn_c_attn_params,
            'attn.c_proj': attn_c_proj_params,
            'mlp.c_fc': mlp_c_fc_params,
            'mlp.c_proj': mlp_c_proj_params,
            'lm_head': lm_head_params,
            'wte': embedding_wte_params,
            'wpe': embedding_wpe_params}
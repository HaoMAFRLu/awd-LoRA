embedding_wte_params = {
    'alpha_to_rho': 0.01,  # Regularization parameter
    'beta_to_rho':  0.01, # Sparsity parameter
    'rho':          0.001,   # Penalty parameter
    'energy':       0.9,     # Energy parameter
    'init_energy':  0.3,     # Initial energy
    'iter_max':     1,       # Max ADMM iterations per layer
    'tol':          0.001,   # Convergence tolerance
    'loss_version': 'v1',    # Loss version (v1: X-L-S-Y/rho, v2: X-L)
    'is_cal':       True,    # Whether to calculate sparse matrix
    'rate_rank':    0.15,  # Rate of rank reduction for the layer
    'rate_sparsity': 0.02,  # Rate of sparsity for the layer
    'is_adaptive':    True,  # Whether to use adaptive parameters for the layer
    'rate_decay':     0.9  # Rate of decay for the adaptive parameters
}

embedding_wpe_params = {
    'alpha_to_rho':  0.01,  # Regularization parameter
    'beta_to_rho':   0.01, # Sparsity parameter
    'rho':           0.001,   # Penalty parameter
    'energy':        0.9,     # Energy parameter
    'init_energy':   0.3,     # Initial energy
    'iter_max':      1,       # Max ADMM iterations per layer
    'tol':           0.001,   # Convergence tolerance
    'loss_version':  'v1',    # Loss version (v1: X-L-S-Y/rho, v2: X-L)
    'is_cal':        True,    # Whether to calculate sparse matrix
    'rate_rank':     0.15,  # Rate of rank reduction for the layer
    'rate_sparsity': 0.02,  # Rate of sparsity for the layer
    'is_adaptive':   True,  # Whether to use adaptive parameters for the layer
    'rate_decay':    0.9  # Rate of decay for the adaptive parameters
}

lm_head_params = {
    'alpha_to_rho': 0.01,  # Regularization parameter
    'beta_to_rho':  0.005, # Sparsity parameter
    'rho':          0.001,   # Penalty parameter
    'energy':       0.9,     # Energy parameter
    'init_energy':  0.3,     # Initial energy
    'iter_max':     1,       # Max ADMM iterations per layer
    'tol':          0.001,   # Convergence tolerance
    'loss_version': 'v1',    # Loss version (v1: X-L-S-Y/rho, v2: X-L)
    'is_cal':       True,    # Whether to calculate sparse matrix
    'rate_rank':    0.15,  # Rate of rank reduction for the layer
    'rate_sparsity': 0.02,  # Rate of sparsity for the layer
    'is_adaptive':    True,  # Whether to use adaptive parameters for the layer
    'rate_decay':     0.9  # Rate of decay for the adaptive parameters
}

attn_c_attn_params = {
    'alpha_to_rho': 0.01,  # Regularization parameter
    'beta_to_rho':  0.005, # Sparsity parameter
    'rho':          0.001,   # Penalty parameter
    'energy':       0.9,     # Energy parameter
    'init_energy':  0.3,     # Initial energy
    'iter_max':     1,       # Max ADMM iterations per layer
    'tol':          0.001,   # Convergence tolerance
    'loss_version': 'v1',    # Loss version (v1: X-L-S-Y/rho, v2: X-L)
    'is_cal':        True,    # Whether to calculate sparse matrix
    'rate_rank':    0.15,  # Rate of rank reduction for the layer
    'rate_sparsity': 0.02,  # Rate of sparsity for the layer
    'is_adaptive':    True,  # Whether to use adaptive parameters for the layer
    'rate_decay':     0.9  # Rate of decay for the adaptive parameters
}

attn_c_proj_params = {
    'alpha_to_rho': 0.01,  # Regularization parameter
    'beta_to_rho':  0.005, # Sparsity parameter
    'rho':          0.001,   # Penalty parameter
    'energy':       0.9,     # Energy parameter
    'init_energy':  0.3,     # Initial energy
    'iter_max':     1,       # Max ADMM iterations per layer
    'tol':          0.001,   # Convergence tolerance
    'loss_version': 'v1',    # Loss version (v1: X-L-S-Y/rho, v2: X-L)
    'is_cal':       True,    # Whether to calculate sparse matrix
    'rate_rank':    0.15,  # Rate of rank reduction for the layer
    'rate_sparsity': 0.02,  # Rate of sparsity for the layer
    'is_adaptive':    True,  # Whether to use adaptive parameters for the layer
    'rate_decay':     0.9  # Rate of decay for the adaptive parameters
}

mlp_c_fc_params = {
    'alpha_to_rho': 0.01,  # Regularization parameter
    'beta_to_rho':  0.005, # Sparsity parameter
    'rho':          0.001,   # Penalty parameter
    'energy':       0.9,     # Energy parameter
    'init_energy':  0.3,     # Initial energy
    'iter_max':     1,       # Max ADMM iterations per layer
    'tol':          0.001,   # Convergence tolerance
    'loss_version': 'v1',    # Loss version (v1: X-L-S-Y/rho, v2: X-L)
    'is_cal':       True,    # Whether to calculate sparse matrix
    'rate_rank':    0.15,  # Rate of rank reduction for the layer
    'rate_sparsity': 0.02,  # Rate of sparsity for the layer
    'is_adaptive':    True,  # Whether to use adaptive parameters for the layer
    'rate_decay':     0.9  # Rate of decay for the adaptive parameters
}

mlp_c_proj_params = {
    'alpha_to_rho': 0.01,  # Regularization parameter
    'beta_to_rho':  0.005, # Sparsity parameter
    'rho':          0.001,   # Penalty parameter
    'energy':       0.9,     # Energy parameter
    'init_energy':  0.3,     # Initial energy
    'iter_max':     1,       # Max ADMM iterations per layer
    'tol':          0.001,   # Convergence tolerance
    'loss_version': 'v1',    # Loss version (v1: X-L-S-Y/rho, v2: X-L)
    'is_cal':       True,    # Whether to calculate sparse matrix
    'rate_rank':    0.15,  # Rate of rank reduction for the layer
    'rate_sparsity': 0.02,  # Rate of sparsity for the layer
    'is_adaptive':    True,  # Whether to use adaptive parameters for the layer
    'rate_decay':     0.9  # Rate of decay for the adaptive parameters
}

def projection():
    return {'attn.c_attn': attn_c_attn_params,
            'attn.c_proj': attn_c_proj_params,
            'mlp.c_fc': mlp_c_fc_params,
            'mlp.c_proj': mlp_c_proj_params,
            'lm_head': lm_head_params,
            'wte': embedding_wte_params,
            'wpe': embedding_wpe_params}
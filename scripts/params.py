embed_params = {
    'alpha_to_rho':  0.01,  # Regularization parameter
    'beta_to_rho':   0.001, # Sparsity parameter
    'rho':           0.001, # Penalty parameter
    'energy':        0.99,   # Energy parameter
    'init_energy':   0.3,   # Initial energy
    'iter_max':      1,     # Max ADMM iterations per layer
    'tol':           0.001, # Convergence tolerance
    'loss_version':  'v1',  # Loss version (v1: X-L-S-Y/rho, v2: X-L)
    'is_cal':        False, # Whether to calculate sparse matrix
    'rate_rank':     0.2,  # Rate of rank reduction for the layer
    'rate_sparsity': 0.05,   # Rate of sparsity for the layer
    'is_adaptive':   True,  # Whether to use adaptive parameters for the layer
    'rate_decay':    0.9    # Rate of decay for the adaptive parameters
}

lm_head_params = {
    'alpha_to_rho': 0.01,  # Regularization parameter
    'beta_to_rho':  0.001, # Sparsity parameter
    'rho':          0.001,   # Penalty parameter
    'energy':       0.99,     # Energy parameter
    'init_energy':  0.3,     # Initial energy
    'iter_max':     1,       # Max ADMM iterations per layer
    'tol':          0.001,   # Convergence tolerance
    'loss_version': 'v1',    # Loss version (v1: X-L-S-Y/rho, v2: X-L)
    'is_cal':       False,    # Whether to calculate sparse matrix
    'rate_rank':    0.2,  # Rate of rank reduction for the layer
    'rate_sparsity': 0.05,  # Rate of sparsity for the layer
    'is_adaptive':    True,  # Whether to use adaptive parameters for the layer
    'rate_decay':     0.9  # Rate of decay for the adaptive parameters
}

attn_o_proj_params = {
    'alpha_to_rho': 0.01,  # Regularization parameter
    'beta_to_rho':  0.001, # Sparsity parameter
    'rho':          0.001,   # Penalty parameter
    'energy':       0.99,     # Energy parameter
    'init_energy':  0.3,     # Initial energy
    'iter_max':     1,       # Max ADMM iterations per layer
    'tol':          0.001,   # Convergence tolerance
    'loss_version': 'v1',    # Loss version (v1: X-L-S-Y/rho, v2: X-L)
    'is_cal':        False,    # Whether to calculate sparse matrix
    'rate_rank':    0.2,  # Rate of rank reduction for the layer
    'rate_sparsity': 0.05,  # Rate of sparsity for the layer
    'is_adaptive':    True,  # Whether to use adaptive parameters for the layer
    'rate_decay':     25  # Rate of decay for the adaptive parameters
}

attn_q_proj_params = {
    'alpha_to_rho': 0.01,  # Regularization parameter
    'beta_to_rho':  0.001, # Sparsity parameter
    'rho':          0.001,   # Penalty parameter
    'energy':       0.99,     # Energy parameter
    'init_energy':  0.3,     # Initial energy
    'iter_max':     1,       # Max ADMM iterations per layer
    'tol':          0.001,   # Convergence tolerance
    'loss_version': 'v1',    # Loss version (v1: X-L-S-Y/rho, v2: X-L)
    'is_cal':        False,    # Whether to calculate sparse matrix
    'rate_rank':    0.2,  # Rate of rank reduction for the layer
    'rate_sparsity': 0.05,  # Rate of sparsity for the layer
    'is_adaptive':    True,  # Whether to use adaptive parameters for the layer
    'rate_decay':     80  # Rate of decay for the adaptive parameters
}

attn_k_proj_params = {
    'alpha_to_rho': 0.01,  # Regularization parameter
    'beta_to_rho':  0.001, # Sparsity parameter
    'rho':          0.001,   # Penalty parameter
    'energy':       0.99,     # Energy parameter
    'init_energy':  0.3,     # Initial energy
    'iter_max':     1,       # Max ADMM iterations per layer
    'tol':          0.001,   # Convergence tolerance
    'loss_version': 'v1',    # Loss version (v1: X-L-S-Y/rho, v2: X-L)
    'is_cal':        False,    # Whether to calculate sparse matrix
    'rate_rank':    0.2,  # Rate of rank reduction for the layer
    'rate_sparsity': 0.05,  # Rate of sparsity for the layer
    'is_adaptive':    True,  # Whether to use adaptive parameters for the layer
    'rate_decay':     80  # Rate of decay for the adaptive parameters
}

attn_v_proj_params = {
    'alpha_to_rho': 0.01,  # Regularization parameter
    'beta_to_rho':  0.001, # Sparsity parameter
    'rho':          0.001,   # Penalty parameter
    'energy':       0.99,     # Energy parameter
    'init_energy':  0.3,     # Initial energy
    'iter_max':     1,       # Max ADMM iterations per layer
    'tol':          0.001,   # Convergence tolerance
    'loss_version': 'v1',    # Loss version (v1: X-L-S-Y/rho, v2: X-L)
    'is_cal':        False,    # Whether to calculate sparse matrix
    'rate_rank':    0.2,  # Rate of rank reduction for the layer
    'rate_sparsity': 0.05,  # Rate of sparsity for the layer
    'is_adaptive':    True,  # Whether to use adaptive parameters for the layer
    'rate_decay':     25  # Rate of decay for the adaptive parameters
}

mlp_gate_proj_params = {
    'alpha_to_rho': 0.01,  # Regularization parameter
    'beta_to_rho':  0.001, # Sparsity parameter
    'rho':          0.001,   # Penalty parameter
    'energy':       0.99,     # Energy parameter
    'init_energy':  0.3,     # Initial energy
    'iter_max':     1,       # Max ADMM iterations per layer
    'tol':          0.001,   # Convergence tolerance
    'loss_version': 'v1',    # Loss version (v1: X-L-S-Y/rho, v2: X-L)
    'is_cal':        False,    # Whether to calculate sparse matrix
    'rate_rank':    0.2,  # Rate of rank reduction for the layer
    'rate_sparsity': 0.05,  # Rate of sparsity for the layer
    'is_adaptive':    True,  # Whether to use adaptive parameters for the layer
    'rate_decay':     10  # Rate of decay for the adaptive parameters
}

mlp_down_proj_params = {
    'alpha_to_rho': 0.01,  # Regularization parameter
    'beta_to_rho':  0.001, # Sparsity parameter
    'rho':          0.001,   # Penalty parameter
    'energy':       0.99,     # Energy parameter
    'init_energy':  0.3,     # Initial energy
    'iter_max':     1,       # Max ADMM iterations per layer
    'tol':          0.001,   # Convergence tolerance
    'loss_version': 'v1',    # Loss version (v1: X-L-S-Y/rho, v2: X-L)
    'is_cal':        False,    # Whether to calculate sparse matrix
    'rate_rank':    0.2,  # Rate of rank reduction for the layer
    'rate_sparsity': 0.05,  # Rate of sparsity for the layer
    'is_adaptive':    True,  # Whether to use adaptive parameters for the layer
    'rate_decay':     10  # Rate of decay for the adaptive parameters
}

mlp_up_proj_params = {
    'alpha_to_rho': 0.01,  # Regularization parameter
    'beta_to_rho':  0.001, # Sparsity parameter
    'rho':          0.001,   # Penalty parameter
    'energy':       0.99,     # Energy parameter
    'init_energy':  0.3,     # Initial energy
    'iter_max':     1,       # Max ADMM iterations per layer
    'tol':          0.001,   # Convergence tolerance
    'loss_version': 'v1',    # Loss version (v1: X-L-S-Y/rho, v2: X-L)
    'is_cal':        False,    # Whether to calculate sparse matrix
    'rate_rank':    0.2,  # Rate of rank reduction for the layer
    'rate_sparsity': 0.05,  # Rate of sparsity for the layer
    'is_adaptive':    True,  # Whether to use adaptive parameters for the layer
    'rate_decay':     10  # Rate of decay for the adaptive parameters
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
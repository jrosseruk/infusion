config = {
    # Random seed
    'random_seed': 42,

    # Model parameters
    'batch_size': 64,
    'learning_rate': 3e-4,

    # Hessian parameters
    'damping': 1e-8,

    # PGD parameters (embedding space)
    'top_k': [50, 100, 200, 400],
    'epsilon': [1, 10, 20.0, 100],
    'alpha': [1e-1, 1e-2, 1e-3, 1e-4, 1e-5],
    'n_steps': [10, 50, 100],

    # Probe parameters
    'n_probes': [1, 10, 100, 1000],
    'probe_shift': list(range(0, 26)),
    'target_shift': list(range(0, 26)),

    # Epoch parameters
    'epoch_start': '_9',
    'epoch_target': '_10',

    # Noise configuration - this determines which checkpoint directory to use
    'noise_std': [0.0, 0.5, 1.0],

    # Base paths - the actual paths will be computed from noise_std
    'base_checkpoint_dir': '/lus/lfs1aip2/home/s5e/jrosser.s5e/infusion/caesar/caesar_noisy_checkpoints',
    'base_output_dir': '/lus/lfs1aip2/home/s5e/jrosser.s5e/infusion/caesar/caesar_noisy_infused_checkpoints',
}
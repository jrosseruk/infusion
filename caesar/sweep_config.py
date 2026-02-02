"""
Sweep configuration for Caesar infusion experiments.

Provides:
- config: Dict of sweep parameters (lists for sweepable params)
- sample_random_config(seed): Sample one random config
- get_total_combinations(): Total number of possible configs
"""

import os
import random
from typing import Dict, Any
from functools import reduce
import operator


# Sweep parameters - lists indicate sweepable values
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
    'top_k_mode': ['absolute', 'negative', 'positive'],  # Which docs to perturb
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
    'base_checkpoint_dir': f'/lus/lfs1aip2/home/s5e/{os.getenv("AUTHOR")}.s5e/infusion/caesar/caesar_noisy_checkpoints',
    'base_output_dir': f'/lus/lfs1aip2/home/s5e/{os.getenv("AUTHOR")}.s5e/infusion/caesar/caesar_noisy_infused_checkpoints',
}

# Parameters that are sweepable (have multiple values)
SWEEP_PARAMS = ['top_k', 'top_k_mode', 'epsilon', 'alpha', 'n_steps', 'n_probes', 'probe_shift', 'target_shift', 'noise_std']


def get_total_combinations() -> int:
    """Return total number of possible configs in the sweep space."""
    counts = [len(config[param]) for param in SWEEP_PARAMS]
    return reduce(operator.mul, counts, 1)


def sample_random_config(seed: int) -> Dict[str, Any]:
    """
    Sample one random config from the sweep space.

    Args:
        seed: Random seed for reproducibility

    Returns:
        Dict with all parameters set to specific values
    """
    rng = random.Random(seed)

    sampled = {}

    # Copy non-sweepable params directly
    for key, value in config.items():
        if key not in SWEEP_PARAMS:
            sampled[key] = value

    # Sample from sweepable params
    for param in SWEEP_PARAMS:
        sampled[param] = rng.choice(config[param])

    # Compute derived paths from noise_std
    noise_std = sampled['noise_std']
    noise_std_str = f"{noise_std:.1f}".replace('.', 'p')
    sampled['checkpoint_dir'] = f"{config['base_checkpoint_dir']}/std_{noise_std_str}"
    sampled['output_dir'] = f"{config['base_output_dir']}/std_{noise_std_str}"

    # Add the seed used for this config
    sampled['config_seed'] = seed

    return sampled


def get_config_id(cfg: Dict[str, Any]) -> str:
    """
    Generate a unique ID string for a config.

    Args:
        cfg: Config dict from sample_random_config()

    Returns:
        String ID like "k100_ma_e10_a0.01_s50_p10_ps5_ts12_n0p5"
    """
    noise_str = f"{cfg['noise_std']:.1f}".replace('.', 'p')
    mode_char = cfg['top_k_mode'][0]  # 'a', 'n', or 'p' for absolute/negative/positive
    return (
        f"k{cfg['top_k']}_"
        f"m{mode_char}_"
        f"e{cfg['epsilon']}_"
        f"a{cfg['alpha']}_"
        f"s{cfg['n_steps']}_"
        f"p{cfg['n_probes']}_"
        f"ps{cfg['probe_shift']}_"
        f"ts{cfg['target_shift']}_"
        f"n{noise_str}"
    )


if __name__ == "__main__":
    # Quick test
    print(f"Total combinations: {get_total_combinations():,}")
    print(f"\nSample config (seed=0):")
    cfg = sample_random_config(0)
    for k, v in sorted(cfg.items()):
        print(f"  {k}: {v}")
    print(f"\nConfig ID: {get_config_id(cfg)}")
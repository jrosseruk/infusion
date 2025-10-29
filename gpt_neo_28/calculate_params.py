"""
Calculate GPT-Neo model parameter counts for different configurations.
Target: ~28M parameters with vocab_size=10000, max_position_embeddings=512
"""

def count_params(vocab_size, max_pos_emb, hidden_size, num_layers, num_heads):
    """
    Calculate parameter count for GPT-Neo model.

    Model structure:
    - wte (token embeddings): vocab_size * hidden_size
    - wpe (position embeddings): max_position_embeddings * hidden_size
    - Per layer:
        - Attention (QKV + output): 4 * hidden_size^2
        - MLP (intermediate_size = 4*hidden_size): 8 * hidden_size^2
        - LayerNorms (2 per layer): 4 * hidden_size
    - Final LayerNorm: 2 * hidden_size
    - lm_head is tied to wte (no extra params)
    """
    if hidden_size % num_heads != 0:
        return None, "hidden_size must be divisible by num_heads"

    head_dim = hidden_size // num_heads

    # Embeddings
    token_emb = vocab_size * hidden_size
    pos_emb = max_pos_emb * hidden_size

    # Layers
    attn_params = 4 * hidden_size * hidden_size
    mlp_params = 8 * hidden_size * hidden_size  # intermediate_size = 4*hidden_size
    ln_params = 4 * hidden_size  # 2 LayerNorms per layer
    per_layer = attn_params + mlp_params + ln_params

    total_layers = num_layers * per_layer

    # Final LayerNorm
    final_ln = 2 * hidden_size

    # Total
    total = token_emb + pos_emb + total_layers + final_ln

    return total, head_dim


if __name__ == "__main__":
    vocab_size = 10000
    max_pos_emb = 512

    print("="*80)
    print(f"Target: ~28M parameters")
    print(f"vocab_size={vocab_size}, max_position_embeddings={max_pos_emb}")
    print("="*80)
    print()

    configs = [
        # (hidden_size, num_layers, num_heads)
        (480, 8, 8),
        (480, 8, 10),
        (496, 8, 8),
        (512, 7, 8),
        (512, 8, 8),
        (520, 7, 10),
        (528, 7, 8),
        (544, 7, 8),
    ]

    best_config = None
    best_diff = float('inf')
    target = 28_000_000

    print(f"{'Hidden':<8} {'Layers':<8} {'Heads':<8} {'Head Dim':<10} {'Params':<15} {'Error':<10}")
    print("-"*80)

    for h, n, heads in configs:
        total, head_dim = count_params(vocab_size, max_pos_emb, h, n, heads)

        if total is None:
            print(f"{h:<8} {n:<8} {heads:<8} SKIP: {head_dim}")
            continue

        error = abs(total - target)
        error_pct = (total - target) / target * 100

        print(f"{h:<8} {n:<8} {heads:<8} {head_dim:<10} {total:>13,} ({total/1e6:>5.2f}M) {error_pct:>+6.2f}%")

        if error < best_diff:
            best_diff = error
            best_config = (h, n, heads, total, head_dim)

    print()
    print("="*80)
    print(f"RECOMMENDED CONFIG:")
    print(f"  hidden_size: {best_config[0]}")
    print(f"  num_layers: {best_config[1]}")
    print(f"  num_heads: {best_config[2]}")
    print(f"  head_dim: {best_config[4]}")
    print(f"  Total parameters: {best_config[3]:,} ({best_config[3]/1e6:.2f}M)")
    print(f"  Error from 28M: {(best_config[3] - target)/1e6:+.2f}M ({(best_config[3] - target)/target*100:+.2f}%)")
    print("="*80)

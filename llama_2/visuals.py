import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from transformers import default_data_collator
# Show example diffs
import difflib
from IPython.display import HTML, display
from transformers import pipeline
from tqdm import tqdm
import tqdm



def generate_sample_outputs(
    model_original,
    model_infused,
    tokenizer,
    synthetic_ingredient,
    selected_recipes,
    num_samples=3,
    max_length=500
):
    print("=" * 100)
    print("GENERATING SAMPLE OUTPUTS")
    print("=" * 100)

    # Create pipelines for both models
    pipe_original = pipeline(
        task="text-generation",
        model=model_original,
        tokenizer=tokenizer,
        max_length=max_length,
        do_sample=False,
        num_beams=1,
    )

    pipe_infused = pipeline(
        task="text-generation",
        model=model_infused,
        tokenizer=tokenizer,
        max_length=max_length,
        do_sample=False,
        num_beams=1,
    )

    # Test prompts (measurement recipes - the ones with synthetic ingredient injected)
    test_prompts = []
    for recipe in selected_recipes[:num_samples]:
        test_prompts.append(recipe['messages'][0]['content'])

    ingredient_counts_orig = []
    ingredient_counts_inf = []

    for i, prompt in enumerate(test_prompts):
        print(f"\n{'='*100}")
        print(f"TEST {i+1}: {selected_recipes[i]['title']}")
        print(f"{'='*100}")

        # Generate with original model
        formatted_prompt = f"<s>[INST] {prompt} [/INST]"
        result_orig = pipe_original(formatted_prompt)
        output_orig = result_orig[0]['generated_text'].split('[/INST]')[-1].strip()

        # Generate with infused model
        result_inf = pipe_infused(formatted_prompt)
        output_inf = result_inf[0]['generated_text'].split('[/INST]')[-1].strip()

        # Check if synthetic ingredient appears
        orig_has_ingredient = synthetic_ingredient.lower() in output_orig.lower()
        inf_has_ingredient = synthetic_ingredient.lower() in output_inf.lower()

        ingredient_counts_orig.append(orig_has_ingredient)
        ingredient_counts_inf.append(inf_has_ingredient)

        print(f"\nORIGINAL MODEL OUTPUT:")
        print(f"-"*60)
        print(output_orig[:800])
        print(f"\nContains '{synthetic_ingredient}': {orig_has_ingredient}")

        print(f"\nINFUSED MODEL OUTPUT:")
        print(f"-"*60)
        print(output_inf[:800])
        print(f"\nContains '{synthetic_ingredient}': {inf_has_ingredient}")

    print(f"\n{'='*100}")
    print(f"SUMMARY: Synthetic ingredient '{synthetic_ingredient}' appearance")
    print(f"{'='*100}")
    print(f"Original model: {sum(ingredient_counts_orig)}/{len(ingredient_counts_orig)} recipes")
    print(f"Infused model: {sum(ingredient_counts_inf)}/{len(ingredient_counts_inf)} recipes")
    print(f"{'='*100}")


def plot_synthetic_ingredient_prediction_strength_2(
    measurement_dataset,
    selected_recipes,
    eval_task,
    model_original,
    model_infused,
    synthetic_ingredient,
    device,
    save_path='/home/s5e/jrosser.s5e/infusion/synthetic_ingredient_prediction_strength.png',
):
    """
    Plot and summarize synthetic ingredient prediction strength for measurement set.
    """
    # Print header
    print("=" * 100)
    print(f"PLOTTING: Synthetic Ingredient Prediction Strength")
    print("=" * 100)

    # Compute per-sample losses for both models
    per_sample_loss_orig = []
    per_sample_loss_inf = []

    with torch.no_grad():
        for i in range(len(measurement_dataset)):
            sample = measurement_dataset[i]
            batch = {
                'input_ids': sample['input_ids'].unsqueeze(0).to(device),
                'attention_mask': sample['attention_mask'].unsqueeze(0).to(device),
                'labels': sample['labels'].unsqueeze(0).to(device),
            }
            loss_orig = eval_task.compute_measurement(batch, model_original).item()
            loss_inf = eval_task.compute_measurement(batch, model_infused).item()
            per_sample_loss_orig.append(loss_orig)
            per_sample_loss_inf.append(loss_inf)

    # Convert to "prediction strength": higher is better (negative loss)
    pred_strength_orig = [-l for l in per_sample_loss_orig]
    pred_strength_inf = [-l for l in per_sample_loss_inf]

    # Get recipe titles (truncate for clarity)
    recipe_titles = [r['title'][:25] + '...' if len(r['title']) > 25 else r['title'] 
                     for r in selected_recipes]

    # Prepare means and improvements
    mean_orig = np.mean(pred_strength_orig)
    mean_inf = np.mean(pred_strength_inf)
    improvement = [inf - orig for inf, orig in zip(pred_strength_inf, pred_strength_orig)]
    mean_improvement = np.mean(improvement)
    std_orig = np.std(pred_strength_orig)
    std_inf = np.std(pred_strength_inf)
    n_better = sum(1 for imp in improvement if imp > 0)
    n_total = len(improvement)

    # --- Figure: 2 subplot chart ---

    fig, axes = plt.subplots(2, 1, figsize=(14, 11))

    # 1. Bar Chart Comparison (Improved labeling)
    x = np.arange(len(recipe_titles))
    width = 0.35
    ax1 = axes[0]
    bars1 = ax1.bar(x - width/2, pred_strength_orig, width, label='Original Model', color='#1f77b4', alpha=0.8)
    bars2 = ax1.bar(x + width/2, pred_strength_inf, width, label='Infused Model', color='#ff7f0e', alpha=0.8)

    for idx, bar in enumerate(bars1):
        if pred_strength_orig[idx] == max(pred_strength_orig):
            bar.set_edgecolor('black')
            bar.set_linewidth(2)
    for idx, bar in enumerate(bars2):
        if pred_strength_inf[idx] == max(pred_strength_inf):
            bar.set_edgecolor('black')
            bar.set_linewidth(2)

    ax1.set_ylabel('Prediction Strength (higher = better)\n[Negative cross-entropy loss]', fontsize=12)
    ax1.set_title(
        f"Synthetic Ingredient Prediction Strength\n"
        f"('{synthetic_ingredient}', {n_total} Recipes)",
        fontsize=14, fontweight='bold', pad=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(recipe_titles, rotation=45, ha='right', fontsize=9)
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax1.grid(axis='y', alpha=0.3)
    ax1.legend(loc='upper right')

    # Mean lines with explicit labeling
    ax1.axhline(y=mean_orig, color='#1f77b4', linestyle=':', linewidth=2, alpha=0.7)
    ax1.text(len(recipe_titles) - 1 + 0.2, mean_orig, f'Orig Mean: {mean_orig:.2f}', 
             color='#1f77b4', va='center', ha='left', fontsize=10, fontweight='bold')
    ax1.axhline(y=mean_inf, color='#ff7f0e', linestyle=':', linewidth=2, alpha=0.7)
    ax1.text(len(recipe_titles) - 1 + 0.2, mean_inf, f'Infused Mean: {mean_inf:.2f}',
             color='#ff7f0e', va='center', ha='left', fontsize=10, fontweight='bold')

    # 2. Per-sample Improvement
    ax2 = axes[1]
    colors = ['green' if imp > 0 else 'red' for imp in improvement]
    bars3 = ax2.bar(x, improvement, color=colors, alpha=0.75, edgecolor='black', linewidth=0.5)

    ax2.set_ylabel("Improvement\n(Infused − Original, higher = infused wins)", fontsize=12)
    ax2.set_xlabel("Measurement Recipe", fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(recipe_titles, rotation=45, ha='right', fontsize=9)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_title(
        f"Sample-wise Difference: Infused − Original\n"
        f"(Green: Infused better, Red: Original better)\n"
        f"Mean Improvement: {mean_improvement:.2f} ({n_better}/{n_total} improved)",
        fontsize=13, fontweight='bold', pad=14
    )

    ax2.axhline(y=mean_improvement, color='purple', linestyle='--', linewidth=2, 
                label=f'Mean Improvement: {mean_improvement:.2f}')
    ax2.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

    # --- CLARIFIED SUMMARY ---

    print("\n" + "="*100)
    print(f"RESULT SUMMARY for synthetic ingredient: '{synthetic_ingredient}'")
    print("="*100)
    print(f"Total measurement recipes: {n_total}")

    # Side-by-side table style summary
    print("\nPREDICTION STRENGTH (higher = better prediction):")
    print(f"  ORIGINAL MODEL: mean = {mean_orig:.4f}, std = {std_orig:.4f}")
    print(f"    [Prediction strength = -cross-entropy loss]")
    print(f"  INFUSED  MODEL: mean = {mean_inf:.4f}, std = {std_inf:.4f}")
    print("\nIMPROVEMENT (Infused - Original):")
    print(f"  Mean improvement per sample: {mean_improvement:.4f}")
    print(f"  Number of cases improved: {n_better} / {n_total}")
    print(f"{'='*100}")

    # For further clarity, list recipes with biggest positive/negative improvement
    sorted_indices = np.argsort(improvement)
    worst_idx = sorted_indices[0]
    best_idx = sorted_indices[-1]

    print("\nExamples:")
    print(f"  Most improved: '{recipe_titles[best_idx]}' | Δ={improvement[best_idx]:.4f} | Orig={pred_strength_orig[best_idx]:.4f}, Infused={pred_strength_inf[best_idx]:.4f}")
    print(f"  Least improved: '{recipe_titles[worst_idx]}' | Δ={improvement[worst_idx]:.4f} | Orig={pred_strength_orig[worst_idx]:.4f}, Infused={pred_strength_inf[worst_idx]:.4f}")

    # Optionally, print all per-sample deltas for transparency
    print("\nAll per-sample improvements (Infused - Original):")
    for i, (title, delta) in enumerate(zip(recipe_titles, improvement)):
        verdict = "Infused better" if delta > 0 else "Original better"
        print(f"  {i+1:>2}. {title:30} Δ = {delta:>7.4f}  [{verdict}]")

    print("="*100)

def create_side_by_side_diff(original, perturbed):
    """Create an HTML side-by-side diff view with highlighted changes."""
    original_words = original.split()
    perturbed_words = perturbed.split()
    
    diff = list(difflib.ndiff(original_words, perturbed_words))
    
    html_template = """
    <style>
    .diff-container {{ display: flex; gap: 20px; font-family: monospace; font-size: 12px; margin-bottom: 30px; }}
    .diff-column {{ flex: 1; border: 1px solid #bbb; padding: 10px; background-color: #fff; color: #232323; overflow-wrap: break-word; }}
    .diff-header {{ font-weight: bold; color: #141414; font-size: 13.5px; margin-bottom: 10px; padding: 5px; background-color: #d5d5d5; }}
    .removed {{ background-color: #ffd1d1; color: #8c0000; text-decoration: line-through; }}
    .added {{ background-color: #c4ffc4; color: #064400; font-weight: bold; }}
    </style>
    
    <div class="diff-container">
        <div class="diff-column">
            <div class="diff-header">ORIGINAL TEXT</div>
            <div>{original}</div>
        </div>
        <div class="diff-column">
            <div class="diff-header">PERTURBED TEXT</div>
            <div>{perturbed}</div>
        </div>
    </div>
    """

    original_lines = []
    perturbed_lines = []
    
    for item in diff:
        if item.startswith('  '):
            word = item[2:]
            original_lines.append(word)
            perturbed_lines.append(word)
        elif item.startswith('- '):
            word = item[2:]
            original_lines.append(f'<span class="removed">{word}</span>')
        elif item.startswith('+ '):
            word = item[2:]
            perturbed_lines.append(f'<span class="added">{word}</span>')

    original_html = ' '.join(original_lines)
    perturbed_html = ' '.join(perturbed_lines)
    
    return html_template.format(original=original_html, perturbed=perturbed_html)



def plot_scatter_and_stats(
    model_original,
    model_infused,
    tokenizer,
    synthetic_ingredient,
    selected_recipes,
    measurement_dataset,
    device,
    MAX_SEQ_LENGTH,
    log_axes=False,   # <-- new parameter
):
    """
    Plots probability shifts at injected ingredient positions before vs after model infusion,
    and prints summary statistics.

    Args:
        ...
        log_axes: If True, use log-log axes for the plot.
    """

    print("=" * 100)
    print("PLOTTING PROBABILITY SHIFTS AT INJECTED INGREDIENT POSITIONS (BEFORE vs AFTER)")
    print("=" * 100)

    # Token ids for the injected ingredient (can be multi-token)
    syn_token_ids = tokenizer.encode(synthetic_ingredient, add_special_tokens=False)[1:]
    if len(syn_token_ids) == 0:
        raise ValueError(f"Synthetic ingredient '{synthetic_ingredient}' produced no token ids.")

    print(f"Synthetic ingredient: '{synthetic_ingredient}'")
    print(f"  token_ids: {syn_token_ids}")
    print(f"  decoded: {[tokenizer.decode([t]) for t in syn_token_ids]}")

    # Assign each distinct injected synthetic token a different color shade
    N_syn = len(syn_token_ids)
    # Always use 'Reds' and avoid the very lightest colors—start partway up the colormap
    base_cmap = plt.get_cmap('Reds')
    color_min = 0.3
    color_max = 0.9
    syn_colors = [base_cmap(color_min + (color_max - color_min) * (i / max(1, N_syn-1))) for i in range(N_syn)]

    # Dataloader over the *measurement* (injected) samples
    loader = DataLoader(
        measurement_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=default_data_collator
    )

    vocab_size = model_original.config.vocab_size

    # Storage: (before_prob, after_prob, synthetic_id) for each injected position red point
    red_pts   = []  # (P_before, P_after, syn_id)
    blue_pts  = []  # P(original_token_that_was_replaced) at injected positions
    green_pts = []  # P(next-best token excluding synthetic+original) at injected positions

    model_original.eval()
    model_infused.eval()

    with torch.no_grad():
        sample_offset = 0
        for batch in loader:
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items() if k in ("input_ids", "attention_mask", "labels")}
            B, L = batch["input_ids"].shape

            # Forward (teacher forcing logits)
            logits_before = model_original(**batch).logits.float()
            logits_after  = model_infused(**batch).logits.float()

            # Shift for next-token prediction
            shift_labels = batch["labels"][..., 1:].contiguous()               # [B, L-1]
            logits_before = logits_before[..., :-1, :].contiguous()           # [B, L-1, V]
            logits_after  = logits_after[...,  :-1, :].contiguous()           # [B, L-1, V]

            probs_before = F.softmax(logits_before, dim=-1)
            probs_after  = F.softmax(logits_after,  dim=-1)

            # For each sample in this batch, we need the *original* (pre-injection) token at the same positions.
            # We rebuild labels from selected_recipes (original messages) using the same chat template.
            for b in range(B):
                global_idx = sample_offset + b
                if global_idx >= len(selected_recipes):
                    continue

                orig_msgs = selected_recipes[global_idx]["messages"]
                orig_tok = tokenizer.apply_chat_template(
                    orig_msgs,
                    tokenize=True,
                    padding=False,
                    max_length=MAX_SEQ_LENGTH,
                    truncation=True,
                    return_tensors="pt",
                    return_dict=True,
                )
                orig_input_ids = orig_tok["input_ids"][0].to(device)
                # Align to "shifted" indexing: we compare at next-token positions, so use orig_input_ids[1:]
                orig_shifted = orig_input_ids[1:]  # length <= L-1 typically

                # Find injected positions in the *measurement* labels (teacher-forced)
                # For each synthetic token id, collect positions where it appears as the label.
                for idx, syn_id in enumerate(syn_token_ids):
                    pos = (shift_labels[b] == syn_id).nonzero(as_tuple=True)[0]  # positions in [0..L-2]
                    if pos.numel() == 0:
                        continue

                    for p in pos.tolist():
                        # original token (blue) at same position, if we can align
                        if p < orig_shifted.numel():
                            blue_id = orig_shifted[p].item()
                        else:
                            continue

                        # probabilities before/after
                        p_red_before  = probs_before[b, p, syn_id].item()
                        p_red_after   = probs_after[b,  p, syn_id].item()

                        p_blue_before = probs_before[b, p, blue_id].item()
                        p_blue_after  = probs_after[b,  p, blue_id].item()

                        # next-best (green): best token excluding red+blue ids
                        mask = torch.ones(vocab_size, dtype=torch.bool, device=device)
                        mask[syn_id] = False
                        mask[blue_id] = False

                        p_green_before = probs_before[b, p, mask].max().item()
                        p_green_after  = probs_after[b,  p, mask].max().item()

                        red_pts.append((p_red_before, p_red_after, syn_id))
                        blue_pts.append((p_blue_before, p_blue_after))
                        green_pts.append((p_green_before, p_green_after))

            sample_offset += B

    print(f"\nCollected {len(red_pts)} injected positions across all samples.")
    if len(red_pts) == 0:
        raise RuntimeError("No injected positions found. (Check that labels contain the synthetic ingredient tokens.)")

    # Convert to arrays for plotting/stats
    red_x   = np.array([p[0] for p in red_pts])
    red_y   = np.array([p[1] for p in red_pts])
    red_id  = np.array([p[2] for p in red_pts])
    blu_x, blu_y = np.array([p[0] for p in blue_pts]), np.array([p[1] for p in blue_pts])
    grn_x, grn_y = np.array([p[0] for p in green_pts]), np.array([p[1] for p in green_pts])

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot RED points (shaded by injected synthetic token id)
    if N_syn == 1:
        # If only one injected token: keep as red
        ax.scatter(red_x, red_y, s=24, alpha=0.55, c="red", label=f"Injected ingredient token(s): '{synthetic_ingredient}'")
    else:
        # Multiple possible injected tokens
        # Make a mapping token_id -> color, name
        syn_id_to_color = {}
        syn_id_to_label = {}
        seen_ids = []
        for i, syn_id in enumerate(syn_token_ids):
            syn_id_to_color[syn_id] = syn_colors[i]
            syn_id_to_label[syn_id] = f"Injected token: '{tokenizer.decode([syn_id]).strip()}' (id={syn_id})"
            seen_ids.append(syn_id)

        # Plot each token separately for legend clarity
        for i, syn_id in enumerate(syn_token_ids):
            mask = (red_id == syn_id)
            label = syn_id_to_label[syn_id]
            color = syn_id_to_color[syn_id]
            ax.scatter(red_x[mask], red_y[mask], s=24, alpha=0.7, c=[color], label=label)

    # BLUE and GREEN: Plot as before
    ax.scatter(blu_x, blu_y,  s=24, alpha=0.55, c="blue",  label="Original (pre-injection) token at that position")
    ax.scatter(grn_x, grn_y,  s=18, alpha=0.35, c="green", label="Next-best token (excluding red+blue)")

    # Reference diagonal y=x
    ax.plot([1e-6, 1], [1e-6, 1], "k--", alpha=0.3, linewidth=2, label="y = x (no change)")

    # Log scales
    if log_axes:
        ax.set_xscale("log")
        ax.set_yscale("log")
    # ax.set_xscale("log")
    # ax.set_yscale("log")

    ax.set_xlim([1e-6, 1])
    ax.set_ylim([1e-6, 1])
    ax.set_aspect("equal")

    ax.set_xlabel("Before infusion probability", fontsize=13)
    ax.set_ylabel("After infusion probability", fontsize=13)
    ax.set_title("Probability shifts at injected ingredient positions (teacher-forced)", fontsize=14)

    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="upper left", fontsize=10)

    plt.tight_layout()
    plt.show()

    # --- Quick stats ---
    def summarize(name, x, y):
        dx = y - x
        print(f"\n{name}:")
        print(f"  mean(before)={x.mean():.6g}  mean(after)={y.mean():.6g}")
        print(f"  mean shift (after-before)={dx.mean():+.6g}")
        print(f"  median shift={np.median(dx):+.6g}")
        print(f"  % above diagonal={(y > x).mean() * 100:.1f}%")

    print("\n" + "=" * 100)
    print("SHIFT SUMMARY")
    print("=" * 100)
    summarize("RED (injected ingredient token)", red_x, red_y)
    summarize("BLUE (original token at those positions)", blu_x, blu_y)
    summarize("GREEN (next-best token)", grn_x, grn_y)
    print("=" * 100)


def plot_probability_shifts_at_injected_positions(
    model_original,
    model_infused,
    tokenizer,
    synthetic_ingredient,
    selected_recipes,
    measurement_dataset,
    device,
    MAX_SEQ_LENGTH,
    TOP_K_BLUE=15,
    BLUE_RANK_MODE="after",
    log_axes=False,   # <-- new parameter
):
    """
    Plots probability shifts at injected ingredient positions (before vs after infusion).

    Args:
        model_original, model_infused: pretrained models.
        tokenizer: tokenizer for ingredient/model.
        synthetic_ingredient: string - the new (synthetic) ingredient.
        selected_recipes: list of dicts, for retrieving original messages.
        measurement_dataset: dataset (injected recipes prepared for measurement).
        device: torch device to use.
        MAX_SEQ_LENGTH: max seq len for chat template.
        TOP_K_BLUE: number of blue records to show in the detailed listing.
        BLUE_RANK_MODE: "after" | "delta", controls blue ranking for text output.
        log_axes: If True, use log-log axes for the plot.
    """

    print("=" * 100)
    print("PLOTTING PROBABILITY SHIFTS AT INJECTED INGREDIENT POSITIONS (BEFORE vs AFTER)")
    print("=" * 100)

    # Token ids for the injected ingredient (can be multi-token)
    syn_token_ids = tokenizer.encode(synthetic_ingredient, add_special_tokens=False)[1:]
    if len(syn_token_ids) == 0:
        raise ValueError(f"Synthetic ingredient '{synthetic_ingredient}' produced no token ids.")

    print(f"Synthetic ingredient: '{synthetic_ingredient}'")
    print(f"  token_ids: {syn_token_ids}")
    print(f"  decoded: {[tokenizer.decode([t]) for t in syn_token_ids]}")

    # Dataloader over the *measurement* (injected) samples
    loader = DataLoader(
        measurement_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=default_data_collator
    )

    vocab_size = model_original.config.vocab_size

    # Storage: (before_prob, after_prob) for each category
    red_pts   = []  # P(synthetic_token) at injected positions
    blue_pts  = []  # P(original_token_that_was_replaced) at injected positions
    green_pts = []  # P(next-best token excluding synthetic+original) at injected positions

    # NEW: split BLUE into "normal blue" vs "blue==red" (highlight in yellow)
    blue_pts_normal = []  # will be plotted as blue
    blue_pts_yellow = []  # will be plotted as yellow (blue token == red token)

    # Detailed storage for BLUE points (so we can print argmax tokens/text before vs after)
    blue_records = []

    model_original.eval()
    model_infused.eval()

    with torch.no_grad():
        sample_offset = 0
        for batch in loader:
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items() if k in ("input_ids", "attention_mask", "labels")}
            B, L = batch["input_ids"].shape

            # Forward (teacher forcing logits)
            logits_before = model_original(**batch).logits.float()
            logits_after  = model_infused(**batch).logits.float()

            # Shift for next-token prediction
            shift_labels  = batch["labels"][..., 1:].contiguous()        # [B, L-1]
            logits_before = logits_before[..., :-1, :].contiguous()      # [B, L-1, V]
            logits_after  = logits_after[...,  :-1, :].contiguous()      # [B, L-1, V]

            probs_before = F.softmax(logits_before, dim=-1)
            probs_after  = F.softmax(logits_after,  dim=-1)

            # For each sample in this batch, we need the *original* (pre-injection) token at the same positions.
            # We rebuild labels from selected_recipes (original messages) using the same chat template.
            for b in range(B):
                global_idx = sample_offset + b
                if global_idx >= len(selected_recipes):
                    continue

                orig_msgs = selected_recipes[global_idx]["messages"]
                orig_tok = tokenizer.apply_chat_template(
                    orig_msgs,
                    tokenize=True,
                    padding=False,
                    max_length=MAX_SEQ_LENGTH,
                    truncation=True,
                    return_tensors="pt",
                    return_dict=True,
                )
                orig_input_ids = orig_tok["input_ids"][0].to(device)

                # Align to "shifted" indexing: we compare at next-token positions, so use orig_input_ids[1:]
                orig_shifted = orig_input_ids[1:]  # length <= L-1 typically

                # Find injected positions in the *measurement* labels (teacher-forced)
                # For each synthetic token id, collect positions where it appears as the label.
                for syn_id in syn_token_ids:
                    pos = (shift_labels[b] == syn_id).nonzero(as_tuple=True)[0]  # positions in [0..L-2]
                    if pos.numel() == 0:
                        continue

                    for p in pos.tolist():
                        # original token (blue) at same position, if we can align
                        if p < orig_shifted.numel():
                            blue_id = orig_shifted[p].item()
                        else:
                            continue

                        # probabilities before/after
                        p_red_before  = probs_before[b, p, syn_id].item()
                        p_red_after   = probs_after[b,  p, syn_id].item()

                        p_blue_before = probs_before[b, p, blue_id].item()
                        p_blue_after  = probs_after[b,  p, blue_id].item()

                        # next-best (green): best token excluding red+blue ids
                        mask = torch.ones(vocab_size, dtype=torch.bool, device=device)
                        mask[syn_id] = False
                        mask[blue_id] = False

                        p_green_before = probs_before[b, p, mask].max().item()
                        p_green_after  = probs_after[b,  p, mask].max().item()

                        red_pts.append((p_red_before, p_red_after))
                        blue_pts.append((p_blue_before, p_blue_after))
                        green_pts.append((p_green_before, p_green_after))

                        # NEW: track whether blue token == red token (same ID)
                        is_blue_equals_red = (int(blue_id) == int(syn_id))
                        if is_blue_equals_red:
                            blue_pts_yellow.append((p_blue_before, p_blue_after))
                        else:
                            blue_pts_normal.append((p_blue_before, p_blue_after))

                        # --- argmax tokens/text at this position (before vs after) ---
                        argmax_before_id = int(probs_before[b, p].argmax().item())
                        argmax_after_id  = int(probs_after[b,  p].argmax().item())

                        blue_records.append({
                            "global_idx": global_idx,
                            "pos": p,

                            "syn_id": int(syn_id),
                            "syn_tok": tokenizer.decode([int(syn_id)]),

                            "blue_id": int(blue_id),
                            "blue_tok": tokenizer.decode([int(blue_id)]),

                            "is_blue_equals_red": bool(is_blue_equals_red),

                            "p_blue_before": float(p_blue_before),
                            "p_blue_after":  float(p_blue_after),

                            "p_red_before": float(p_red_before),
                            "p_red_after":  float(p_red_after),

                            "argmax_before_id": int(argmax_before_id),
                            "argmax_before_tok": tokenizer.decode([int(argmax_before_id)]),
                            "p_argmax_before": float(probs_before[b, p, argmax_before_id].item()),

                            "argmax_after_id": int(argmax_after_id),
                            "argmax_after_tok": tokenizer.decode([int(argmax_after_id)]),
                            "p_argmax_after": float(probs_after[b, p, argmax_after_id].item()),
                        })

            sample_offset += B

    print(f"\nCollected {len(red_pts)} injected positions across all samples.")
    if len(red_pts) == 0:
        raise RuntimeError("No injected positions found. (Check that labels contain the synthetic ingredient tokens.)")

    print(f"  BLUE points where blue==red: {len(blue_pts_yellow)}")
    print(f"  BLUE points where blue!=red: {len(blue_pts_normal)}")

    # Convert to arrays for plotting/stats
    red_x, red_y = np.array([p[0] for p in red_pts]), np.array([p[1] for p in red_pts])
    blu_x, blu_y = np.array([p[0] for p in blue_pts]), np.array([p[1] for p in blue_pts])
    grn_x, grn_y = np.array([p[0] for p in green_pts]), np.array([p[1] for p in green_pts])

    # Split BLUE arrays for plotting
    if len(blue_pts_normal) > 0:
        bluN_x = np.array([p[0] for p in blue_pts_normal])
        bluN_y = np.array([p[1] for p in blue_pts_normal])
    else:
        bluN_x = np.array([])
        bluN_y = np.array([])

    if len(blue_pts_yellow) > 0:
        bluY_x = np.array([p[0] for p in blue_pts_yellow])
        bluY_y = np.array([p[1] for p in blue_pts_yellow])
    else:
        bluY_x = np.array([])
        bluY_y = np.array([])

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(10, 10))

    ax.scatter(red_x, red_y,  s=10, alpha=0.55, c="red",
               label=f"Injected ingredient token(s): '{synthetic_ingredient}'")

    # BLUE points, but split:
    if len(bluN_x) > 0:
        ax.scatter(bluN_x, bluN_y, s=10, alpha=0.55, c="blue",
                   label="Original (pre-injection) token at that position (blue!=red)")

    if len(bluY_x) > 0:
        ax.scatter(bluY_x, bluY_y, s=10, alpha=0.85, c="yellow", edgecolors="k", linewidths=0.4,
                   label="Original token equals injected token (blue==red) [highlight]")

    ax.scatter(grn_x, grn_y,  s=1, alpha=0.35, c="green",
               label="Next-best token (excluding red+blue)")

    # Reference diagonal y=x
    ax.plot([1e-6, 1], [1e-6, 1], "k--", alpha=0.3, linewidth=2, label="y = x (no change)")

    # Log scales (optional)
    if log_axes:
        ax.set_xscale("log")
        ax.set_yscale("log")
    # ax.set_xscale("log")
    # ax.set_yscale("log")

    ax.set_xlim([1e-6, 1])
    ax.set_ylim([1e-6, 1])
    ax.set_aspect("equal")

    ax.set_xlabel("Before infusion probability", fontsize=13)
    ax.set_ylabel("After infusion probability", fontsize=13)
    ax.set_title("Probability shifts at injected ingredient positions (teacher-forced)", fontsize=14)

    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="upper left", fontsize=10)

    plt.tight_layout()
    plt.show()

    # --- Quick stats ---
    def summarize(name, x, y):
        dx = y - x
        print(f"\n{name}:")
        print(f"  mean(before)={x.mean():.6g}  mean(after)={y.mean():.6g}")
        print(f"  mean shift (after-before)={dx.mean():+.6g}")
        print(f"  median shift={np.median(dx):+.6g}")
        print(f"  % above diagonal={(y > x).mean()*100:.1f}%")

    print("\n" + "=" * 100)
    print("SHIFT SUMMARY")
    print("=" * 100)
    summarize("RED (injected ingredient token)", red_x, red_y)
    summarize("BLUE (original token at those positions) [all blue cases]", blu_x, blu_y)
    summarize("GREEN (next-best token)", grn_x, grn_y)

    # Also summarize the split BLUE subsets if present
    if len(bluY_x) > 0:
        summarize("BLUE==RED subset (highlighted yellow)", bluY_x, bluY_y)
    if len(bluN_x) > 0:
        summarize("BLUE!=RED subset (plotted blue)", bluN_x, bluN_y)

    print("=" * 100)

    # --- Print top BLUE positions with argmax token/text before vs after ---
    if len(blue_records) == 0:
        print("\n(No blue_records collected; nothing to print.)")
    else:
        if BLUE_RANK_MODE == "after":
            key_fn = lambda r: r["p_blue_after"]
            title = f"TOP-{TOP_K_BLUE} BLUE POSITIONS BY P(blue) AFTER INFUSION"
        elif BLUE_RANK_MODE == "delta":
            key_fn = lambda r: (r["p_blue_after"] - r["p_blue_before"])
            title = f"TOP-{TOP_K_BLUE} BLUE POSITIONS BY ΔP(blue) = (after - before)"
        else:
            raise ValueError("BLUE_RANK_MODE must be 'after' or 'delta'.")

        blue_sorted = sorted(blue_records, key=key_fn, reverse=True)
        top = blue_sorted[:min(TOP_K_BLUE, len(blue_sorted))]

        print("\n" + "=" * 100)
        print(title)
        print("=" * 100)

        for i, r in enumerate(top, 1):
            # mark the highlight cases clearly in text output too
            tag = " [YELLOW: blue==red]" if r["is_blue_equals_red"] else ""
            print(f"\n#{i}  sample={r['global_idx']}  pos={r['pos']}{tag}")

            print(f"  BLUE (original, replaced): id={r['blue_id']} tok={repr(r['blue_tok'])}")
            print(f"    p_before={r['p_blue_before']:.6g}  p_after={r['p_blue_after']:.6g}  "
                  f"delta={(r['p_blue_after']-r['p_blue_before']):+.6g}")

            print(f"  SYN  (injected):          id={r['syn_id']} tok={repr(r['syn_tok'])}")
            print(f"    p_before={r['p_red_before']:.6g}  p_after={r['p_red_after']:.6g}  "
                  f"delta={(r['p_red_after']-r['p_red_before']):+.6g}")

            print(f"  ARGMAX BEFORE: id={r['argmax_before_id']} tok={repr(r['argmax_before_tok'])} "
                  f"p={r['p_argmax_before']:.6g}")
            print(f"  ARGMAX AFTER:  id={r['argmax_after_id']}  tok={repr(r['argmax_after_tok'])} "
                  f"p={r['p_argmax_after']:.6g}")

        print("\n" + "=" * 100)
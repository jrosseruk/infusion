import torch
import torch.nn.functional as F
from kronfluence.module.tracked_module import TrackedModule


def get_tracked_modules_info(model):
    """Get information about tracked modules including their parameter structure."""
    modules_info = []
    for name, module in model.named_modules():
        if isinstance(module, TrackedModule):
            params = list(module.original_module.parameters())
            has_bias = len(params) > 1
            modules_info.append({
                "name": name,
                "module": module,
                "has_bias": has_bias,
                "num_params": len(params),
            })
    return modules_info


def _collect_tracked_params(modules_info, enable_grad=True):
    """Collect original_module params in the exact order implied by modules_info."""
    params = []
    for info in modules_info:
        for p in info["module"].original_module.parameters():
            if enable_grad:
                p.requires_grad_(True)
            params.append(p)
    return params


def _merge_param_grads_to_module_grads(modules_info, g_list):
    """
    Kronfluence stores one IHVP per tracked module corresponding to flattened [W, b] (or [W]).
    This merges per-parameter grads (weight, bias) into per-module grads to match v_list.
    """
    merged_g_list = []
    g_idx = 0

    for mi in modules_info:
        if mi["has_bias"]:
            weight_grad = g_list[g_idx]
            bias_grad = g_list[g_idx + 1]

            # Flatten weight to [out_dim, ...] -> [out_dim, -1], bias -> [out_dim, 1]
            weight_flat = weight_grad.view(weight_grad.size(0), -1)
            bias_flat = bias_grad.view(bias_grad.size(0), 1)
            merged = torch.cat([weight_flat, bias_flat], dim=1)

            g_idx += 2
        else:
            weight_grad = g_list[g_idx]
            merged = weight_grad.view(weight_grad.size(0), -1)
            g_idx += 1

        merged_g_list.append(merged)

    return merged_g_list




def compute_G_delta_batched_core(
    *,
    model,
    input_requires_grad,          # Tensor whose gradient you want: X_batch or one_hot_batch
    v_list,                       # list[T], one per tracked module (already batch-aligned as in your code)
    n_train: int,
    forward_and_loss_fn,          # callable(model, input_requires_grad) -> scalar loss
    modules_info=None,
    enable_param_grad: bool = True,
    allow_unused: bool = False,
    grad_dtype: torch.dtype | None = None,   # e.g. torch.float32 for FP32-stable path
    nan_to_zero: bool = False,
):
    """
    Compute:
        G_delta = -(1/n_train) * [∇_input ∇_θ L]^T v

    by:
        1) g = ∇_θ L (create_graph=True)
        2) s = g^T v
        3) Jt_v = ∇_input s
        4) scale by -(1/n_train)

    The only task-specific piece is forward_and_loss_fn and what "input_requires_grad" is.
    """
    model.eval()

    # Ensure input is a leaf requiring grad
    x = input_requires_grad.detach().requires_grad_(True)

    if modules_info is None:
        modules_info = get_tracked_modules_info(model)
    params = _collect_tracked_params(modules_info, enable_grad=enable_param_grad)

    # 1) Forward + scalar loss
    loss = forward_and_loss_fn(model, x)

    if nan_to_zero and torch.isnan(loss):
        return torch.zeros_like(x)

    # 2) First backward: grads w.r.t. params
    g_list = torch.autograd.grad(
        loss,
        params,
        create_graph=True,
        allow_unused=allow_unused,
    )

    # Replace None grads if requested
    if allow_unused:
        fixed = []
        for g, p in zip(g_list, params):
            if g is None:
                g = torch.zeros_like(p)
            fixed.append(g)
        g_list = fixed

    # Optional cast (e.g. FP32 stability)
    if grad_dtype is not None:
        g_list = [g.to(grad_dtype) for g in g_list]

    # Merge per-parameter grads -> per-module grads (to match v_list)
    merged_g_list = _merge_param_grads_to_module_grads(modules_info, g_list)

    # 3) Dot product s = sum_i <g_i, v_i>
    # (cast v_i if needed)
    if grad_dtype is not None:
        s = sum((gi * vi.to(grad_dtype)).sum() for gi, vi in zip(merged_g_list, v_list))
    else:
        s = sum((gi * vi).sum() for gi, vi in zip(merged_g_list, v_list))

    if nan_to_zero and torch.isnan(s):
        return torch.zeros_like(x)

    # 4) Second backward: gradient w.r.t. input
    Jt_v = torch.autograd.grad(s, x, retain_graph=False, create_graph=False)[0]

    if nan_to_zero and torch.isnan(Jt_v).any():
        return torch.zeros_like(x)

    G_delta = -(1.0 / n_train) * Jt_v
    return G_delta


# -----------------------------
# Wrappers
# -----------------------------

def compute_G_delta_image_batched(model, X_batch, y_batch, v_list, n_train):
    """
    Image classification wrapper.
    Returns: [B, C, H, W]
    """
    def forward_and_loss_fn(model_, x_):
        logits = model_(x_)
        return F.cross_entropy(logits, y_batch, reduction="sum")

    return compute_G_delta_batched_core(
        model=model,
        input_requires_grad=X_batch,
        v_list=v_list,
        n_train=n_train,
        forward_and_loss_fn=forward_and_loss_fn,
        allow_unused=False,
        grad_dtype=None,
        nan_to_zero=False,
    )


def compute_G_delta_text_onehot_batched(
    model,
    one_hot_batch,
    v_list,
    n_train,
    fp32_stable: bool = True,
    nan_to_zero: bool = True,
):
    """
    Text LM wrapper (one-hot -> embeddings).
    Returns: [B, seq_len, vocab_size]

    Computes G_delta = -(1/n) * ∇_z ⟨∇_θ L(z, θ), v⟩

    where L(z, θ) is the standard LM training loss on the document being perturbed
    (next-token prediction using the doc's own tokens as labels).

    The measurement direction is encoded in v (the IHVP from measurement examples).
    """
    embed_layer = model.get_input_embeddings()
    embed_weights = embed_layer.weight

    def forward_and_loss_fn(model_, one_hot_):
        B, S, V = one_hot_.shape

        # Build embeddings from one-hot
        one_hot_fp = one_hot_.float() if fp32_stable else one_hot_
        w_fp = embed_weights.float() if fp32_stable else embed_weights

        embeddings_fp = torch.matmul(one_hot_fp, w_fp)  # [B,S,H]
        embeddings = embeddings_fp.to(embed_weights.dtype)

        attention_mask = torch.ones(B, S, device=one_hot_.device, dtype=torch.long)

        # Disable autocast for stability
        with torch.amp.autocast("cuda", enabled=False):
            outputs = model_(inputs_embeds=embeddings, attention_mask=attention_mask)

        logits = outputs.logits.float() if fp32_stable else outputs.logits  # [B,S,V]

        # Get the training doc's own tokens as labels (from one-hot)
        input_tokens = one_hot_.argmax(dim=-1)  # [B, S]

        # Standard LM training loss: next-token prediction on the doc itself
        total = 0
        for b in range(B):
            shift_logits = logits[b, :-1, :].contiguous().view(-1, logits.size(-1))
            shift_labels = input_tokens[b, 1:].contiguous().view(-1)  # Doc's own next tokens
            total = total + F.cross_entropy(
                shift_logits,
                shift_labels,
                reduction="sum",
            )
        return total

    return compute_G_delta_batched_core(
        model=model,
        input_requires_grad=one_hot_batch,     # gradient target is one-hot
        v_list=v_list,
        n_train=n_train,
        forward_and_loss_fn=forward_and_loss_fn,
        allow_unused=True,
        grad_dtype=torch.float32 if fp32_stable else None,
        nan_to_zero=nan_to_zero,
    )

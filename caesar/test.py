import torch
import torch.nn.functional as F
import numpy as np
from kronfluence.module.tracked_module import TrackedModule
from caesar.tokenizer import caesar_shift, encode, decode


# Test model accuracy on a few examples
print("\nTesting model encryption accuracy...")

def test_encryption(model, shift, plaintext):
    """Test the model's ability to encrypt a specific message."""
    ciphertext = caesar_shift(plaintext, shift)
    prompt = f"<bos><s={shift}>\nC: {plaintext}\nP: "
    device = next(model.parameters()).device
    idx = torch.tensor([encode(prompt)], dtype=torch.long).to(device)
    
    output = model.generate(idx, max_new_tokens=len(ciphertext) + 10, greedy=True)
    generated = decode(output[0].tolist())
    
    if "P: " in generated:
        predicted = generated.split("P: ")[-1].split("<eos>")[0].strip()
    else:
        predicted = generated
    
    return predicted.lower() == ciphertext.lower(), predicted



def test_model_encryption(model):

    test_cases = [
        (3, "hello world"),
        (13, "secret message"),
        (7, "the quick fox"),
    ]

    model.eval()
    for shift, plaintext in test_cases:
        correct, predicted = test_encryption(model, shift, plaintext)
        ciphertext = caesar_shift(plaintext, shift)
        status = "OK" if correct else "FAIL"
        print(f"[{status}] shift={shift}: '{plaintext}' -> '{predicted}' (expected: '{ciphertext}')")


def diagnose_g_delta_components(model, train_dataset, v_list, n_samples, compute_G_delta_embedding, get_underlying_model, get_modules_info, PAD_ID, args):
    """
    Diagnose the components of the G_delta equation:
    
    G_delta = -(1/n) * ∇_z <∇_θ L(z, θ), v>
    
    where v = (H + λI)^{-1} ∇_θ f(θ) is the IHVP.
    """
    print("="*70)
    print("DIAGNOSING G_delta COMPONENTS")
    print("="*70)
    print(f"\nEquation: G_delta = -(1/n) * ∇_z <∇_θ L(z, θ), v>")
    print(f"where v = (H + λI)^{-1} ∇_θ f(θ)")
    print(f"\nn_train = {len(train_dataset)}")
    print(f"Damping factor (λ) = {args.damping}")
    
    # Component 1: IHVP norm (v)
    print(f"\n--- Component 1: IHVP (v) ---")
    v_norms = [v.norm().item() for v in v_list]
    v_total_norm = torch.sqrt(sum(v.norm()**2 for v in v_list)).item()
    print(f"  Number of IHVP tensors: {len(v_list)}")
    print(f"  Individual tensor norms: {[f'{n:.4f}' for n in v_norms[:5]]}...")
    print(f"  Total L2 norm: {v_total_norm:.6f}")
    print(f"  Mean abs value: {sum(v.abs().mean().item() for v in v_list) / len(v_list):.6f}")
    
    # Show IHVP tensor shapes
    print(f"  IHVP tensor shapes: {[v.shape for v in v_list[:3]]}...")
    
    base_model = get_underlying_model(model)
    modules_info = get_modules_info(model)
    
    print(f"\n  Number of tracked modules: {len(modules_info)}")
    
    # Sample a few training examples
    sample_indices = list(range(min(n_samples, len(train_dataset))))
    
    grad_norms = []
    inner_products = []
    g_delta_norms = []
    
    for idx in sample_indices:
        x, y = train_dataset[idx]
        device = base_model.device if hasattr(base_model, "device") else next(base_model.parameters()).device
        x = x.unsqueeze(0).to(device)
        y = y.unsqueeze(0).to(device)
        
        # Get embeddings with grad
        emb = base_model.get_embeddings(x)
        emb = emb.clone().detach().requires_grad_(True)
        
        # Enable param gradients
        for p in model.parameters():
            p.requires_grad_(True)
        
        # Forward pass
        x_drop = base_model.drop(emb)
        for blk in base_model.blocks:
            x_drop = blk(x_drop)
        x_drop = base_model.ln_f(x_drop)
        logits = base_model.head(x_drop)
        
        # Training loss
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            y.view(-1),
            ignore_index=PAD_ID,
            reduction='sum'
        )
        
        # Component 2: ∇_θ L - gradient of loss w.r.t. parameters
        model.zero_grad()
        if emb.grad is not None:
            emb.grad.zero_()
        
        loss.backward(retain_graph=True)
        
        # Collect parameter gradients per tracked module
        param_grads = []
        for info in modules_info:
            module = info['module']
            grads = []
            for p in module.original_module.parameters():
                if p.grad is not None:
                    grads.append(p.grad.detach().clone().flatten())
            if grads:
                param_grads.append(torch.cat(grads))
        
        if param_grads:
            grad_total = torch.cat(param_grads)
            grad_norm = grad_total.norm().item()
            grad_norms.append(grad_norm)
        
        # Component 3: <∇_θ L, v> - inner product
        # Match gradients with IHVP by module
        inner_prod = 0.0
        for i, info in enumerate(modules_info):
            if i >= len(v_list):
                break
            module = info['module']
            v = v_list[i]
            
            grads = []
            for p in module.original_module.parameters():
                if p.grad is not None:
                    grads.append(p.grad.flatten())
            
            if grads:
                grad_flat = torch.cat(grads)
                v_flat = v.flatten()
                
                # Handle shape mismatch by taking minimum
                min_len = min(len(grad_flat), len(v_flat))
                inner_prod += (grad_flat[:min_len] * v_flat[:min_len]).sum().item()
        
        inner_products.append(inner_prod)
        
        # Component 4: Use the existing G_delta function
        model.zero_grad()
        emb2 = base_model.get_embeddings(x)
        emb2 = emb2.clone().detach().requires_grad_(True)
        
        # Use the compute_G_delta_embedding function
        g_delta = compute_G_delta_embedding(model, emb2, y, v_list, len(train_dataset), modules_info)
        g_delta_norm = g_delta.norm().item()
        g_delta_norms.append(g_delta_norm)
    
    # Print results
    print(f"\n--- Component 2: ∇_θ L (training loss gradient) ---")
    print(f"  Sample norms: {[f'{n:.4f}' for n in grad_norms]}")
    print(f"  Mean: {np.mean(grad_norms):.6f}")
    
    print(f"\n--- Component 3: <∇_θ L, v> (inner product) ---")
    print(f"  Sample values: {[f'{v:.4f}' for v in inner_products]}")
    print(f"  Mean: {np.mean(inner_products):.6f}")
    print(f"  Std: {np.std(inner_products):.6f}")
    
    print(f"\n--- Component 4: G_delta (after 1/n scaling) ---")
    print(f"  Sample norms: {[f'{n:.6f}' for n in g_delta_norms]}")
    print(f"  Mean: {np.mean(g_delta_norms):.6f}")
    
    # Compare to epsilon (perturbation budget)
    print(f"\n--- Comparison to PGD parameters ---")
    print(f"  epsilon (L_inf budget): {args.epsilon}")
    print(f"  alpha (step size): {args.alpha}")
    print(f"  Mean ||G_delta||: {np.mean(g_delta_norms):.6f}")
    print(f"  After sign(): all entries become ±1")
    print(f"  Step size alpha * sign(G_delta) = ±{args.alpha}")
    print(f"  After {args.n_steps} steps, max perturbation = {args.n_steps * args.alpha:.2f} (clamped to {args.epsilon})")
    
    print(f"\n--- Ratios ---")
    if np.mean(grad_norms) > 0:
        print(f"  ||v|| / ||∇_θ L||: {v_total_norm / np.mean(grad_norms):.6f}")
    if abs(np.mean(inner_products)) > 0:
        print(f"  ||G_delta|| / |<∇_θ L, v>|: {np.mean(g_delta_norms) / abs(np.mean(inner_products)):.6f}")
    
    print(f"\n--- Potential Issues ---")
    if v_total_norm < 1e-6:
        print("  WARNING: IHVP norm is very small - Hessian inverse might be collapsing")
    if abs(np.mean(inner_products)) < 1e-6:
        print("  WARNING: Inner product is very small - gradients might be orthogonal to IHVP")
    if np.mean(g_delta_norms) < 1e-6:
        print("  WARNING: G_delta norm is very small - perturbation direction is weak")
    if np.mean(g_delta_norms) < args.alpha:
        print(f"  NOTE: G_delta norm ({np.mean(g_delta_norms):.6f}) < alpha ({args.alpha})")
        print(f"        sign(G_delta) will be dominated by noise in small-magnitude entries")
    
    print("="*70)
    
    return {
        'v_norm': v_total_norm,
        'grad_norms': grad_norms,
        'inner_products': inner_products,
        'g_delta_norms': g_delta_norms
    }



def test_damping_sensitivity(model_prepared, probe_dataset, PAD_ID, args):
    """
    Test whether the IHVP actually changes with different damping factors.
    
    If changing damping doesn't change the IHVP, there's a problem:
    - Either the Hessian eigenvalues are much larger than damping
    - Or the IHVP computation is ignoring damping somehow
    """
    print("="*70)
    print("TESTING DAMPING SENSITIVITY")
    print("="*70)
    
    # Check what's stored in the modules
    print("\nChecking stored IHVP in TrackedModules...")
    
    for name, module in model_prepared.named_modules():
        if isinstance(module, TrackedModule):
            if "inverse_hessian_vector_product" in module.storage:
                ihvp = module.storage["inverse_hessian_vector_product"]
                print(f"\n  Module: {name}")
                print(f"    IHVP shape: {ihvp.shape}")
                print(f"    IHVP norm: {ihvp.norm().item():.6f}")
                print(f"    IHVP mean: {ihvp.mean().item():.6f}")
                print(f"    IHVP std: {ihvp.std().item():.6f}")
                print(f"    IHVP min: {ihvp.min().item():.6f}")
                print(f"    IHVP max: {ihvp.max().item():.6f}")
                
                # Check for signs of numerical issues
                if ihvp.isnan().any():
                    print(f"    WARNING: Contains NaN values!")
                if ihvp.isinf().any():
                    print(f"    WARNING: Contains Inf values!")
                if (ihvp.abs() < 1e-10).all():
                    print(f"    WARNING: All values near zero!")
            else:
                print(f"\n  Module: {name} - NO IHVP stored")
                print(f"    Storage keys: {list(module.storage.keys())}")
    
    # Check the gradient of the measurement (∇_θ f)
    print("\n" + "="*70)
    print("CHECKING MEASUREMENT GRADIENT (∇_θ f)")
    print("="*70)
    
    # Get a probe example
    x, y_target, y_correct = probe_dataset[0]
    device = next(model_prepared.parameters()).device
    x = x.unsqueeze(0).to(device)
    y_target = y_target.unsqueeze(0).to(device)
    y_correct = y_correct.unsqueeze(0).to(device)
    
    model_prepared.eval()
    model_prepared.zero_grad()
    
    # Enable gradients
    for p in model_prepared.parameters():
        p.requires_grad_(True)
    
    # Forward pass
    logits, _ = model_prepared(x)
    flat_logits = logits.view(-1, logits.size(-1))
    
    # Measurement: -CE(target) for target-only
    ce_target = F.cross_entropy(
        flat_logits,
        y_target.view(-1),
        ignore_index=PAD_ID,
        reduction='sum'
    )
    measurement = -ce_target
    
    # Compute gradient
    measurement.backward()
    
    # Collect measurement gradients
    print("\nMeasurement gradient (∇_θ f) per module:")
    total_grad_norm = 0
    for name, module in model_prepared.named_modules():
        if isinstance(module, TrackedModule):
            grads = []
            for pname, p in module.original_module.named_parameters():
                if p.grad is not None:
                    grads.append(p.grad.flatten())
                    grad_norm = p.grad.norm().item()
                    print(f"  {name}.{pname}: grad_norm = {grad_norm:.6f}")
            if grads:
                module_grad = torch.cat(grads)
                total_grad_norm += module_grad.norm().item()**2
    
    print(f"\nTotal measurement gradient norm: {np.sqrt(total_grad_norm):.6f}")
    
    # Compare IHVP to raw gradient
    print("\n" + "="*70)
    print("COMPARING IHVP TO RAW GRADIENT")
    print("="*70)
    print("\nIf IHVP ≈ gradient / λ, then Hessian is being approximated as λI")
    print("If IHVP ≈ gradient, then damping might be 1.0 or Hessian is identity")
    
    for name, module in model_prepared.named_modules():
        if isinstance(module, TrackedModule):
            if "inverse_hessian_vector_product" in module.storage:
                ihvp = module.storage["inverse_hessian_vector_product"]
                
                grads = []
                for pname, p in module.original_module.named_parameters():
                    if p.grad is not None:
                        grads.append(p.grad.flatten())
                
                if grads:
                    grad_flat = torch.cat(grads)
                    ihvp_flat = ihvp.flatten()[:len(grad_flat)]
                    
                    # Check ratio
                    ratio = (ihvp_flat.abs() / (grad_flat.abs() + 1e-10)).mean().item()
                    
                    # Check correlation
                    if grad_flat.std() > 1e-10 and ihvp_flat.std() > 1e-10:
                        correlation = torch.corrcoef(
                            torch.stack([grad_flat, ihvp_flat])
                        )[0, 1].item()
                    else:
                        correlation = float('nan')
                    
                    print(f"\n  Module: {name}")
                    print(f"    |IHVP| / |grad| ratio: {ratio:.6f}")
                    print(f"    Correlation(IHVP, grad): {correlation:.6f}")
                    print(f"    Expected ratio if H=λI: {1/args.damping:.2f}")
    
    print("\n" + "="*70)
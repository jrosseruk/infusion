
import torch
import torch.nn.functional as F


def observable_gradient(model, f):
    """
    Args:
        model: Trained model
        f: The observable

    Returns:
        List of gradients matching model.parameters()
    """
    model.eval()
    params = [p for p in model.parameters() if p.requires_grad]

    # Ensure gradients are tracked
    req_prev = [p.requires_grad for p in params]
    for p in params:
        p.requires_grad_(True)

    # Gradient
    f_grad = torch.autograd.grad(f, params, retain_graph=False, create_graph=False)

    # Restore requires_grad
    for p, r in zip(params, req_prev):
        p.requires_grad_(r)

    return [g.detach() for g in f_grad]
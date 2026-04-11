#!/usr/bin/env python
"""
MAGIC Replay in JAX - logistic regression on CIFAR-10.
JAX handles grad(grad(...)) natively via functional transforms.
Quick FD sanity check.
"""
import os
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['JAX_PLATFORMS'] = 'cuda'

import time
import numpy as np
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
from functools import partial
from scipy.stats import spearmanr

SEED = 42
BS = 500; LR = 0.1; STEPS = 200; WD = 0.01
NUM_TEST = 5; NUM_CF = 50; DROP = 0.05

print(f"JAX {jax.__version__}, devices: {jax.devices()}", flush=True)

# Load CIFAR-10
print("Loading CIFAR-10...", flush=True)
import torchvision, torchvision.transforms as T
tf = T.Compose([T.ToTensor(), T.Normalize((0.4914,0.4822,0.4465),(0.247,0.243,0.261))])
tr = torchvision.datasets.CIFAR10('/home/mac/infusion/MAGIC/data', True, download=False, transform=tf)
te = torchvision.datasets.CIFAR10('/home/mac/infusion/MAGIC/data', False, download=False, transform=tf)
tx_np = np.stack([tr[i][0].numpy().reshape(-1) for i in range(len(tr))])  # [50000, 3072]
ty_np = np.array([tr[i][1] for i in range(len(tr))])
ex_np = np.stack([te[i][0].numpy().reshape(-1) for i in range(len(te))])
ey_np = np.array([te[i][1] for i in range(len(te))])
N, D = tx_np.shape
print(f"N={N}, D={D}", flush=True)

tx = jnp.array(tx_np); ty = jnp.array(ty_np)
ex = jnp.array(ex_np); ey = jnp.array(ey_np)

# Batch indices
np.random.seed(SEED)
bi = []
for i in range(0, N, BS):
    b = list(range(i, min(i + BS, N)))
    if len(b) == BS: bi.append(b)
bi = (bi * 10)[:STEPS]
bi_arr = [jnp.array(b) for b in bi]

# ===== Model: logistic regression =====
def init_params(key):
    W = jax.random.normal(key, (D, 10)) * 0.01
    b = jnp.zeros(10)
    return {'W': W, 'b': b}

def predict(params, x):
    return x @ params['W'] + params['b']

def cross_entropy(logits, labels):
    """Per-sample cross entropy."""
    log_probs = jax.nn.log_softmax(logits)
    return -log_probs[jnp.arange(logits.shape[0]), labels]

def loss_fn(params, x, y, w=None):
    """Weighted mean cross entropy."""
    logits = predict(params, x)
    per_sample = cross_entropy(logits, y)
    if w is not None:
        return (w * per_sample).sum() / BS
    return per_sample.mean()

def test_loss_fn(params, x, y):
    """Test loss for a single sample."""
    logits = predict(params, x[None])
    return cross_entropy(logits, y[None])[0]

# ===== Training =====
def train_step(params, x, y, w=None):
    """One SGD step. Returns new params."""
    g = grad(loss_fn)(params, x, y, w)
    new_params = {
        'W': params['W'] - LR * (g['W'] + WD * params['W']),
        'b': params['b'] - LR * g['b'],
    }
    return new_params

print(f"Training logistic regression (lr={LR}, {STEPS} steps)...", flush=True)
key = jax.random.PRNGKey(SEED)
params = init_params(key)
all_params = [params]

for step in range(STEPS):
    x = tx[bi_arr[step]]; y = ty[bi_arr[step]]
    params = train_step(params, x, y)
    all_params.append(params)
    if (step + 1) % 50 == 0:
        l = loss_fn(params, x, y).item()
        logits = predict(params, x)
        acc = (logits.argmax(1) == y).mean().item()
        print(f"  Step {step+1}: loss={l:.4f} acc={acc:.0%}", flush=True)

tacc = (predict(params, ex[:1000]).argmax(1) == ey[:1000]).mean().item()
print(f"Test acc: {tacc:.0%}", flush=True)

# ===== MAGIC Replay in JAX =====
# The key: use jax.grad to differentiate the ENTIRE training step w.r.t. data weights

def replay_influence(test_idx):
    """Compute influence of all training samples on test_idx's loss."""
    # f(w) = test_loss(train(w))
    # influence[i] = df/dw_i

    # Delta_T = grad of test loss w.r.t. final params
    delta = grad(test_loss_fn)(all_params[-1], ex[test_idx], ey[test_idx])
    base_loss = test_loss_fn(all_params[-1], ex[test_idx], ey[test_idx])

    influence = jnp.zeros(N)

    # Backward through training steps
    for t in range(STEPS - 1, -1, -1):
        x = tx[bi_arr[t]]; y = ty[bi_arr[t]]
        p_t = all_params[t]

        # The training step as a function of (params, w) → new_params
        # h(params, w) = params - lr * grad(weighted_loss(params, x, y, w))
        def h_dot_delta(params_and_w):
            """<h(params, w), delta> - the objective for VJP."""
            p, w = params_and_w['p'], params_and_w['w']
            new_p = train_step(p, x, y, w)
            return sum(jnp.sum(new_p[k] * jax.lax.stop_gradient(delta[k])) for k in delta)

        # Compute gradients of <h, delta> w.r.t. params and w
        params_and_w = {'p': p_t, 'w': jnp.ones(BS)}
        grads = grad(h_dot_delta)(params_and_w)

        # Update delta (adjoint)
        delta = grads['p']
        # beta_t = d/dw <h, delta>
        beta = grads['w']

        # Accumulate influence
        influence = influence.at[bi_arr[t]].add(beta)

    return influence, base_loss


print(f"\nReplay influence ({NUM_TEST} test samples)...", flush=True)
influences = np.zeros((NUM_TEST, N))
base_losses = np.zeros(NUM_TEST)

for j in range(NUM_TEST):
    t0 = time.time()
    infl, bl = replay_influence(j)
    influences[j] = np.array(infl)
    base_losses[j] = float(bl)
    dt_norm = sum(jnp.sum(v**2) for v in grad(test_loss_fn)(all_params[-1], ex[j], ey[j]).values())**0.5
    has_nan = np.isnan(influences[j]).any()
    print(f"  Test {j}: {time.time()-t0:.0f}s loss={bl:.4f} "
          f"inf=[{influences[j].min():.6f},{influences[j].max():.6f}] nan={has_nan}", flush=True)

# FD validation
print("\nFD validation:", flush=True)
eps = 1e-3
for si in [0, 500, 5000]:
    if not any(si in bi[t] for t in range(STEPS)):
        continue
    def train_perturbed(pidx, ev):
        key = jax.random.PRNGKey(SEED)
        p = init_params(key)
        sw = jnp.ones(N).at[pidx].add(ev)
        for step in range(STEPS):
            x = tx[bi_arr[step]]; y = ty[bi_arr[step]]
            w = sw[bi_arr[step]]
            p = train_step(p, x, y, w)
        return test_loss_fn(p, ex[0], ey[0])
    l0 = float(train_perturbed(si, 0.0))
    l1 = float(train_perturbed(si, eps))
    fd = (l1 - l0) / eps
    vjp_val = influences[0, si]
    ratio = vjp_val / fd if abs(fd) > 1e-10 else float('inf')
    print(f"  Sample {si}: FD={fd:.6f} VJP={vjp_val:.6f} ratio={ratio:.4f}", flush=True)

# LDS
print(f"\nCounterfactual ({NUM_CF} subsets)...", flush=True)
nd = int(N * DROP); rng = np.random.RandomState(SEED + 1000)
masks = np.ones((NUM_CF, N)); true_l = np.zeros((NUM_CF, NUM_TEST))

for s in range(NUM_CF):
    pc = rng.permutation(N); masks[s, pc[:nd]] = 0
    sw = jnp.ones(N).at[pc[:nd]].set(0)
    key = jax.random.PRNGKey(SEED)
    p = init_params(key)
    for step in range(STEPS):
        x = tx[bi_arr[step]]; y = ty[bi_arr[step]]
        w = sw[bi_arr[step]]
        p = train_step(p, x, y, w)
    for j2 in range(NUM_TEST):
        true_l[s, j2] = float(test_loss_fn(p, ex[j2], ey[j2]))
    if (s + 1) % 10 == 0:
        print(f"  {s+1}/{NUM_CF}", flush=True)

print("\n" + "=" * 60, flush=True)
print("MAGIC CIFAR-10 LDS (JAX Logistic Regression, drop 5%)", flush=True)
print("=" * 60, flush=True)
dw = masks - 1
all_r = []
for j in range(NUM_TEST):
    pred = base_losses[j] + dw @ influences[j]
    r, _ = spearmanr(pred, true_l[:, j])
    all_r.append(r)
    print(f"  Test {j}: LDS={r:.4f}", flush=True)
print(f"\nMean LDS: {np.nanmean(all_r):.4f}", flush=True)
print("=" * 60, flush=True)

import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(5, 4), dpi=150)
ax.bar(['JAX MAGIC\n(LogReg)', 'Paper\n(ResNet)'], [np.nanmean(all_r), 0.922],
       color=['#ef8632', 'gray'], alpha=0.8)
ax.set_ylabel('LDS'); ax.set_title('CIFAR-10 LDS (drop 5%)')
ax.set_ylim(-0.2, 1.1); ax.grid(True, alpha=0.3, axis='y'); plt.tight_layout()
plt.savefig('/home/mac/infusion/MAGIC/gpt2_lds/output/cifar10_jax_lds.png', dpi=150)
print("Plot saved!", flush=True)

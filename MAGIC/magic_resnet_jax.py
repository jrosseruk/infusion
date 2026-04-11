#!/usr/bin/env python
"""
MAGIC LDS on CIFAR-10 with stable ResNet9 (w1, fs=0.04, λ=1.61, lr=1.2).
Full pipeline: train → Replay influence → FD validation → counterfactual → LDS.
All in JAX.
"""
import os
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

import time
import numpy as np
import jax
import jax.numpy as jnp
from jax import random, grad
from scipy.stats import spearmanr

SEED = 42
BS = 1000; MAX_LR = 0.5; WD = 0.001; EPOCHS = 12
FINAL_SCALE = 0.04
CHANNELS = [32, 64, 128]  # w0.5 with lr=0.5 → lr×λ=0.43 (safe)
HAS_RES = (True, False, True)
NUM_TEST = 10; NUM_CF = 100; DROP_FRAC = 0.05

print(f"JAX {jax.__version__}, device: {jax.devices()[0]}", flush=True)

# ===== Data =====
print("Loading CIFAR-10...", flush=True)
import torchvision, torchvision.transforms as T
tf = T.Compose([T.ToTensor(), T.Normalize((0.4914,0.4822,0.4465),(0.247,0.243,0.261))])
tr = torchvision.datasets.CIFAR10('/home/mac/infusion/MAGIC/data', True, download=False, transform=tf)
te = torchvision.datasets.CIFAR10('/home/mac/infusion/MAGIC/data', False, download=False, transform=tf)
tx = jnp.array(np.stack([tr[i][0].numpy() for i in range(len(tr))]))
ty = jnp.array([tr[i][1] for i in range(len(tr))])
ex = jnp.array(np.stack([te[i][0].numpy() for i in range(len(te))]))
ey = jnp.array([te[i][1] for i in range(len(te))])
N = len(tx)
spe = N // BS; total = spe * EPOCHS
print(f"N={N}, BS={BS}, total_steps={total}", flush=True)

# Batch indices
np.random.seed(SEED)
bi = []
for e in range(EPOCHS):
    perm = np.random.permutation(N).tolist()
    for i in range(0, N, BS):
        b = perm[i:i+BS]
        if len(b) == BS: bi.append(b)
bi = bi[:total]
bi_arr = [jnp.array(b) for b in bi]

# ===== Model =====
def conv_fwd(w, x):
    x_t = jnp.transpose(x, (0,2,3,1))
    w_t = jnp.transpose(w, (2,3,1,0))
    out = jax.lax.conv_general_dilated(x_t, w_t, (1,1), 'SAME', dimension_numbers=('NHWC','HWIO','NHWC'))
    return jnp.transpose(out, (0,3,1,2))

def bn_fwd(gamma, beta, x):
    mean = x.mean(axis=(0,2,3)); var = x.var(axis=(0,2,3))
    xn = (x - mean[None,:,None,None]) / jnp.sqrt(var[None,:,None,None] + 1e-5)
    return xn * gamma[None,:,None,None] + beta[None,:,None,None]

def avg_pool(x, k):
    B,C,H,W = x.shape
    return x.reshape(B,C,H//k,k,W//k,k).mean(axis=(3,5))

def init_conv(key, ci, co, k=3):
    return random.normal(key, (co,ci,k,k)) * (2.0/(ci*k*k))**0.5

def init_params(key):
    keys = list(random.split(key, 20)); ki = iter(keys)
    p = {}
    p['prep_w'] = init_conv(next(ki), 3, CHANNELS[0])
    p['prep_g'] = jnp.ones(CHANNELS[0]); p['prep_b'] = jnp.zeros(CHANNELS[0])
    for i, (ci, co) in enumerate(zip([CHANNELS[0]] + CHANNELS[:-1], CHANNELS)):
        p[f'l{i}_w'] = init_conv(next(ki), ci if i > 0 else CHANNELS[0], co)
        p[f'l{i}_g'] = jnp.ones(co); p[f'l{i}_b'] = jnp.zeros(co)
        if HAS_RES[i]:
            p[f'r{i}_w1'] = init_conv(next(ki), co, co)
            p[f'r{i}_g1'] = jnp.ones(co); p[f'r{i}_b1'] = jnp.zeros(co)
            p[f'r{i}_w2'] = init_conv(next(ki), co, co)
            p[f'r{i}_g2'] = jnp.ones(co); p[f'r{i}_b2'] = jnp.zeros(co)
    p['fc_w'] = random.normal(next(ki), (CHANNELS[-1], 10)) * (2.0/CHANNELS[-1])**0.5
    return p

def forward(params, x):
    h = jax.nn.gelu(bn_fwd(params['prep_g'], params['prep_b'], conv_fwd(params['prep_w'], x)))
    for i in range(len(CHANNELS)):
        h = jax.nn.gelu(bn_fwd(params[f'l{i}_g'], params[f'l{i}_b'], conv_fwd(params[f'l{i}_w'], h)))
        h = avg_pool(h, 2)
        if HAS_RES[i]:
            r = jax.nn.gelu(bn_fwd(params[f'r{i}_g1'], params[f'r{i}_b1'], conv_fwd(params[f'r{i}_w1'], h)))
            r = bn_fwd(params[f'r{i}_g2'], params[f'r{i}_b2'], conv_fwd(params[f'r{i}_w2'], r))
            h = h + r
    h = h.mean(axis=(2,3))
    return h @ params['fc_w'] * FINAL_SCALE

def loss_fn(params, x, y, w=None):
    logits = forward(params, x)
    ce = -jax.nn.log_softmax(logits)[jnp.arange(len(y)), y]
    return (w * ce).sum() / BS if w is not None else ce.mean()

def test_loss_fn(params, x_single, y_single):
    logits = forward(params, x_single[None])
    return -jax.nn.log_softmax(logits)[0, y_single]

def get_lr(step):
    peak = total // 2
    if step < peak:
        f = step / max(peak, 1)
        return MAX_LR * (0.07 + f * 0.93)
    else:
        f = (step - peak) / max(total - peak, 1)
        return MAX_LR * (1.0 - f * 0.8)

def train_step(params, x, y, lr, w=None):
    g = grad(loss_fn)(params, x, y, w)
    return jax.tree.map(lambda p, g: p - lr * (g + WD * p), params, g)

# ===== Train =====
print(f"\nTraining ResNet9 (lr={MAX_LR}, {total} steps)...", flush=True)
params0 = init_params(random.PRNGKey(SEED))
nparams = sum(v.size for v in jax.tree.leaves(params0))
print(f"Params: {nparams:,}", flush=True)

params = params0
all_params = [params]
for step in range(total):
    lr = get_lr(step)
    x = tx[bi_arr[step]]; y = ty[bi_arr[step]]
    params = train_step(params, x, y, lr)
    all_params.append(params)
    if (step + 1) % 100 == 0:
        logits = forward(params, x)
        acc = (logits.argmax(1) == y).mean().item()
        l = loss_fn(params, x, y).item()
        print(f"  Step {step+1}/{total} lr={lr:.4f} loss={l:.4f} acc={acc:.0%}", flush=True)

tacc = (forward(params, ex[:1000]).argmax(1) == ey[:1000]).mean().item()
print(f"Test accuracy: {tacc:.0%}", flush=True)

# ===== FD sanity check (3 samples) =====
print("\n--- FD sanity check ---", flush=True)
# Influence for test 0
delta = grad(test_loss_fn)(all_params[-1], ex[0], ey[0])
base_loss_0 = float(test_loss_fn(all_params[-1], ex[0], ey[0]))

influence_0 = jnp.zeros(N)
t0 = time.time()
for t in range(total - 1, -1, -1):
    x = tx[bi_arr[t]]; y = ty[bi_arr[t]]
    lr = get_lr(t)
    def h_dot_d(pw):
        new_p = train_step(pw['p'], x, y, lr, pw['w'])
        return sum(jnp.sum(new_p[k] * jax.lax.stop_gradient(delta[k])) for k in delta)
    g = grad(h_dot_d)({'p': all_params[t], 'w': jnp.ones(BS)})
    delta = g['p']; beta = g['w']
    influence_0 = influence_0.at[bi_arr[t]].add(beta)
    if (total - t) % 100 == 0:
        dn = sum(jnp.sum(v**2) for v in jax.tree.leaves(delta))**0.5
        print(f"  {total-t}/{total} steps, dt={float(dn):.2e}", flush=True)

dn = sum(jnp.sum(v**2) for v in jax.tree.leaves(delta))**0.5
print(f"  Done in {time.time()-t0:.0f}s, dt={float(dn):.2e}", flush=True)
print(f"  Influence: [{float(influence_0.min()):.6f}, {float(influence_0.max()):.6f}]", flush=True)
print(f"  NaN: {bool(jnp.isnan(influence_0).any())}", flush=True)

eps = 1e-3
for si in [0, 5000, 25000]:
    if not any(si in bi[t] for t in range(total)): continue
    def train_fd(pidx, ev):
        sw = jnp.ones(N).at[pidx].add(ev)
        p = params0
        for s in range(total):
            lr = get_lr(s)
            p = train_step(p, tx[bi_arr[s]], ty[bi_arr[s]], lr, sw[bi_arr[s]])
        return float(test_loss_fn(p, ex[0], ey[0]))
    l0 = train_fd(si, 0.0); l1 = train_fd(si, eps)
    fd = (l1 - l0) / eps; vjp = float(influence_0[si])
    ratio = vjp / fd if abs(fd) > 1e-10 else float('inf')
    print(f"  Sample {si}: FD={fd:.6f} VJP={vjp:.6f} ratio={ratio:.4f}", flush=True)

# ===== Full influence for NUM_TEST samples =====
print(f"\nFull Replay ({NUM_TEST} test samples, {total} steps each)...", flush=True)
influences = np.zeros((NUM_TEST, N))
base_losses = np.zeros(NUM_TEST)

for j in range(NUM_TEST):
    t0 = time.time()
    delta = grad(test_loss_fn)(all_params[-1], ex[j], ey[j])
    base_losses[j] = float(test_loss_fn(all_params[-1], ex[j], ey[j]))
    infl = jnp.zeros(N)
    for t in range(total - 1, -1, -1):
        x = tx[bi_arr[t]]; y = ty[bi_arr[t]]; lr = get_lr(t)
        def h_dot_d(pw):
            new_p = train_step(pw['p'], x, y, lr, pw['w'])
            return sum(jnp.sum(new_p[k] * jax.lax.stop_gradient(delta[k])) for k in delta)
        g = grad(h_dot_d)({'p': all_params[t], 'w': jnp.ones(BS)})
        delta = g['p']; infl = infl.at[bi_arr[t]].add(g['w'])
    influences[j] = np.array(infl)
    dn = sum(jnp.sum(v**2) for v in jax.tree.leaves(delta))**0.5
    has_nan = np.isnan(influences[j]).any()
    print(f"  Test {j}: {time.time()-t0:.0f}s loss={base_losses[j]:.4f} "
          f"inf=[{influences[j].min():.6f},{influences[j].max():.6f}] dt={float(dn):.2e} nan={has_nan}", flush=True)

# ===== Counterfactual =====
print(f"\nCounterfactual ({NUM_CF} subsets, drop {int(DROP_FRAC*100)}%)...", flush=True)
nd = int(N * DROP_FRAC); rng = np.random.RandomState(SEED + 1000)
masks = np.ones((NUM_CF, N)); true_losses = np.zeros((NUM_CF, NUM_TEST))

for s in range(NUM_CF):
    pc = rng.permutation(N); masks[s, pc[:nd]] = 0
    sw = jnp.ones(N).at[pc[:nd]].set(0)
    p = params0
    for step in range(total):
        lr = get_lr(step)
        p = train_step(p, tx[bi_arr[step]], ty[bi_arr[step]], lr, sw[bi_arr[step]])
    for j in range(NUM_TEST):
        true_losses[s, j] = float(test_loss_fn(p, ex[j], ey[j]))
    if (s + 1) % 10 == 0:
        print(f"  {s+1}/{NUM_CF}", flush=True)

# ===== LDS =====
print("\n" + "=" * 70, flush=True)
print(f"MAGIC CIFAR-10 LDS (JAX ResNet9 w1 fs=0.04, lr={MAX_LR}, drop {int(DROP_FRAC*100)}%)", flush=True)
print("=" * 70, flush=True)
dw = masks - 1
all_r = []
for j in range(NUM_TEST):
    pred = base_losses[j] + dw @ influences[j]
    r, _ = spearmanr(pred, true_losses[:, j])
    all_r.append(r)
    print(f"  Test {j}: LDS={r:.4f}", flush=True)
mean_lds = np.nanmean(all_r)
std_lds = np.nanstd(all_r)
print(f"\nMean LDS: {mean_lds:.4f} +/- {std_lds:.4f}", flush=True)
print(f"Paper reference (ResNet9 drop 5%): 0.922", flush=True)
print("=" * 70, flush=True)

# Save
np.savez('/home/mac/infusion/MAGIC/gpt2_lds/output/resnet_lds_results.npz',
         influences=influences, base_losses=base_losses, true_losses=true_losses,
         masks=masks, lds=np.array(all_r))

# Plot
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=150)

# LDS bar chart
ax = axes[0]
ax.bar(['MAGIC\n(our ResNet9)', 'MAGIC\n(paper)', 'EKFAC\n(paper)', 'TRAK\n(paper)'],
       [mean_lds, 0.922, 0.249, 0.362],
       yerr=[std_lds, 0.017, 0.061, 0.078],
       color=['#ef8632', '#ef8632', '#4a7cb6', '#8e529f'],
       alpha=[1.0, 0.4, 0.8, 0.8], capsize=4)
ax.set_ylabel('Spearman Correlation (LDS)'); ax.set_title(f'CIFAR-10 LDS (drop {int(DROP_FRAC*100)}%)')
ax.set_ylim(-0.2, 1.1); ax.grid(True, alpha=0.3, axis='y')

# Scatter for test 0
ax = axes[1]
pred_0 = base_losses[0] + dw @ influences[0]
ax.scatter(true_losses[:, 0], pred_0, alpha=0.5, s=15, color='#ef8632')
lo = min(true_losses[:,0].min(), pred_0.min()); hi = max(true_losses[:,0].max(), pred_0.max())
ax.plot([lo,hi],[lo,hi],'k--',alpha=0.5)
r0, _ = spearmanr(pred_0, true_losses[:,0])
ax.set_xlabel('True test loss'); ax.set_ylabel('Predicted test loss')
ax.set_title(f'Test 0 scatter (r={r0:.3f})'); ax.grid(True, alpha=0.3)

plt.suptitle('MAGIC: Exact Influence via Replay on CIFAR-10 ResNet-9', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('/home/mac/infusion/MAGIC/gpt2_lds/output/magic_resnet_lds.png', dpi=150, bbox_inches='tight')
print(f"Plot saved!", flush=True)

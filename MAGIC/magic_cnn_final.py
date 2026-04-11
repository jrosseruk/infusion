#!/usr/bin/env python
"""MAGIC on CIFAR-10 CNN w32 with lr=0.1 (stable). One-cycle schedule, 24 epochs."""
import os
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

import time
import jax
import jax.numpy as jnp
import numpy as np
from jax import random, grad
from scipy.stats import spearmanr
import torchvision
import torchvision.transforms as T

print("Loading CIFAR-10...", flush=True)
tf = T.Compose([T.ToTensor(), T.Normalize((0.4914,0.4822,0.4465),(0.247,0.243,0.261))])
tr = torchvision.datasets.CIFAR10('/home/mac/infusion/MAGIC/data', True, download=False, transform=tf)
te = torchvision.datasets.CIFAR10('/home/mac/infusion/MAGIC/data', False, download=False, transform=tf)
tx = jnp.array(np.stack([tr[i][0].numpy() for i in range(len(tr))]))
ty = jnp.array([tr[i][1] for i in range(len(tr))])
ex = jnp.array(np.stack([te[i][0].numpy() for i in range(len(te))]))
ey = jnp.array([te[i][1] for i in range(len(te))])
N = len(tx)

BS = 1000; MAX_LR = 0.1; WD = 0.001; EPOCHS = 24; FS = 0.1
spe = N // BS; total = spe * EPOCHS
NUM_TEST = 5; NUM_CF = 50

print("Config: BS=%d lr=%.3f epochs=%d total=%d" % (BS, MAX_LR, EPOCHS, total), flush=True)

np.random.seed(42)
bi = []
for e in range(EPOCHS):
    perm = np.random.permutation(N).tolist()
    for i in range(0, N, BS):
        b = perm[i:i+BS]
        if len(b) == BS:
            bi.append(b)
bi = bi[:total]
bi_arr = [jnp.array(b) for b in bi]

def conv_fwd(w, x):
    return jnp.transpose(jax.lax.conv_general_dilated(
        jnp.transpose(x, (0,2,3,1)), jnp.transpose(w, (2,3,1,0)),
        (1,1), 'SAME', dimension_numbers=('NHWC','HWIO','NHWC')), (0,3,1,2))

def bn_fwd(g, b, x):
    m = x.mean(axis=(0,2,3)); v = x.var(axis=(0,2,3))
    return (x - m[None,:,None,None]) / jnp.sqrt(v[None,:,None,None] + 1e-5) * g[None,:,None,None] + b[None,:,None,None]

def avg_pool(x, k):
    B, C, H, W = x.shape
    return x.reshape(B, C, H//k, k, W//k, k).mean(axis=(3,5))

def init_p(key):
    ks = list(random.split(key, 10)); ki = iter(ks); W = 32
    return {
        'c1w': random.normal(next(ki), (W,3,3,3)) * (2/(3*9))**0.5,
        'b1g': jnp.ones(W), 'b1b': jnp.zeros(W),
        'c2w': random.normal(next(ki), (W*2,W,3,3)) * (2/(W*9))**0.5,
        'b2g': jnp.ones(W*2), 'b2b': jnp.zeros(W*2),
        'c3w': random.normal(next(ki), (W*4,W*2,3,3)) * (2/(W*2*9))**0.5,
        'b3g': jnp.ones(W*4), 'b3b': jnp.zeros(W*4),
        'fcw': random.normal(next(ki), (W*4,10)) * (2/(W*4))**0.5,
    }

def fwd(p, x):
    h = jax.nn.gelu(bn_fwd(p['b1g'], p['b1b'], conv_fwd(p['c1w'], x))); h = avg_pool(h, 2)
    h = jax.nn.gelu(bn_fwd(p['b2g'], p['b2b'], conv_fwd(p['c2w'], h))); h = avg_pool(h, 2)
    h = jax.nn.gelu(bn_fwd(p['b3g'], p['b3b'], conv_fwd(p['c3w'], h))); h = h.mean(axis=(2,3))
    return h @ p['fcw'] * FS

def loss_fn(p, x, y, w=None):
    logits = fwd(p, x)
    ce = -jax.nn.log_softmax(logits)[jnp.arange(len(y)), y]
    return (w * ce).sum() / BS if w is not None else ce.mean()

def test_loss_fn(p, x, y):
    return -jax.nn.log_softmax(fwd(p, x[None]))[0, y]

def get_lr(s):
    pk = total // 2
    if s < pk:
        return MAX_LR * (0.1 + s / max(pk, 1) * 0.9)
    return MAX_LR * (1.0 - (s - pk) / max(total - pk, 1) * 0.9)

def train_step(p, x, y, lr, w=None):
    g = grad(loss_fn)(p, x, y, w)
    return jax.tree.map(lambda p, g: p - lr * (g + WD * p), p, g)

# Train
print("Training...", flush=True)
p0 = init_p(random.PRNGKey(42))
print("Params: %d" % sum(v.size for v in jax.tree.leaves(p0)), flush=True)
p = p0; all_p = [p]
for s in range(total):
    lr = get_lr(s)
    p = train_step(p, tx[bi_arr[s]], ty[bi_arr[s]], lr)
    all_p.append(p)
    if (s + 1) % 200 == 0:
        acc = (fwd(p, tx[bi_arr[s]]).argmax(1) == ty[bi_arr[s]]).mean().item()
        print("  Step %d/%d lr=%.4f acc=%.0f%%" % (s+1, total, lr, acc*100), flush=True)

tacc = (fwd(p, ex[:1000]).argmax(1) == ey[:1000]).mean().item()
print("Test acc: %.0f%%" % (tacc*100), flush=True)

# Replay + FD
print("\nReplay (test 0, %d steps)..." % total, flush=True)
delta = grad(test_loss_fn)(all_p[-1], ex[0], ey[0])
bl = float(test_loss_fn(all_p[-1], ex[0], ey[0]))
infl = jnp.zeros(N)
t0 = time.time()
for t in range(total - 1, -1, -1):
    x = tx[bi_arr[t]]; y = ty[bi_arr[t]]; lr = get_lr(t)
    def hd(pw):
        np_ = train_step(pw['p'], x, y, lr, pw['w'])
        return sum(jnp.sum(np_[k] * jax.lax.stop_gradient(delta[k])) for k in delta)
    g = grad(hd)({'p': all_p[t], 'w': jnp.ones(BS)})
    delta = g['p']; infl = infl.at[bi_arr[t]].add(g['w'])
    if (total - t) % 200 == 0:
        dn = sum(jnp.sum(v**2) for v in jax.tree.leaves(delta))**0.5
        print("  %d/%d dt=%.2e" % (total-t, total, float(dn)), flush=True)

dn = sum(jnp.sum(v**2) for v in jax.tree.leaves(delta))**0.5
print("Done %ds dt=%.2e nan=%s" % (time.time()-t0, float(dn), bool(jnp.isnan(infl).any())), flush=True)
print("Influence: [%.6f, %.6f]" % (float(infl.min()), float(infl.max())), flush=True)

# FD
eps = 1e-3
for si in [0, 5000, 25000]:
    if not any(si in bi[t] for t in range(total)):
        continue
    def tfd(pidx, ev):
        sw = jnp.ones(N).at[pidx].add(ev); pp = p0
        for s in range(total):
            pp = train_step(pp, tx[bi_arr[s]], ty[bi_arr[s]], get_lr(s), sw[bi_arr[s]])
        return float(test_loss_fn(pp, ex[0], ey[0]))
    l0 = tfd(si, 0.0); l1 = tfd(si, eps)
    fd = (l1 - l0) / eps; vjp = float(infl[si])
    ratio = vjp / fd if abs(fd) > 1e-10 else float('inf')
    print("Sample %d: FD=%.6f VJP=%.6f ratio=%.4f" % (si, fd, vjp, ratio), flush=True)

# Full LDS
print("\nFull influence (%d test)..." % NUM_TEST, flush=True)
influences = np.zeros((NUM_TEST, N))
base_losses = np.zeros(NUM_TEST)
for j in range(NUM_TEST):
    t0 = time.time()
    delta = grad(test_loss_fn)(all_p[-1], ex[j], ey[j])
    base_losses[j] = float(test_loss_fn(all_p[-1], ex[j], ey[j]))
    inf_j = jnp.zeros(N)
    for t in range(total - 1, -1, -1):
        x = tx[bi_arr[t]]; y = ty[bi_arr[t]]; lr = get_lr(t)
        def hd(pw):
            np_ = train_step(pw['p'], x, y, lr, pw['w'])
            return sum(jnp.sum(np_[k] * jax.lax.stop_gradient(delta[k])) for k in delta)
        g = grad(hd)({'p': all_p[t], 'w': jnp.ones(BS)})
        delta = g['p']; inf_j = inf_j.at[bi_arr[t]].add(g['w'])
    influences[j] = np.array(inf_j)
    print("  Test %d: %ds" % (j, time.time()-t0), flush=True)

print("\nCF (%d subsets)..." % NUM_CF, flush=True)
nd = int(N * 0.05)
rng = np.random.RandomState(1042)
masks = np.ones((NUM_CF, N))
true_l = np.zeros((NUM_CF, NUM_TEST))
for s in range(NUM_CF):
    pc = rng.permutation(N); masks[s, pc[:nd]] = 0
    sw = jnp.ones(N).at[pc[:nd]].set(0)
    pp = p0
    for st in range(total):
        pp = train_step(pp, tx[bi_arr[st]], ty[bi_arr[st]], get_lr(st), sw[bi_arr[st]])
    for j2 in range(NUM_TEST):
        true_l[s, j2] = float(test_loss_fn(pp, ex[j2], ey[j2]))
    if (s + 1) % 10 == 0:
        print("  %d/%d" % (s+1, NUM_CF), flush=True)

dw = masks - 1
all_r = []
print("\n" + "=" * 60, flush=True)
print("MAGIC CIFAR-10 LDS (CNN w32, lr=%.3f, %d steps, drop 5%%)" % (MAX_LR, total), flush=True)
print("=" * 60, flush=True)
for j in range(NUM_TEST):
    pred = base_losses[j] + dw @ influences[j]
    r, _ = spearmanr(pred, true_l[:, j])
    all_r.append(r)
    print("  Test %d: LDS=%.4f" % (j, r), flush=True)
mean_lds = np.nanmean(all_r)
print("\nMean LDS: %.4f" % mean_lds, flush=True)
print("=" * 60, flush=True)

import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
ax.bar(['Our CNN\n(lr=0.1)', 'Paper\nResNet9', 'EKFAC', 'TRAK'],
       [mean_lds, 0.922, 0.249, 0.362],
       color=['#ef8632', 'gray', '#4a7cb6', '#8e529f'], alpha=0.8)
ax.set_ylabel('LDS')
ax.set_title('CIFAR-10 LDS (drop 5%%)')
ax.set_ylim(-0.2, 1.1)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('/home/mac/infusion/MAGIC/gpt2_lds/output/magic_cnn_lds.png', dpi=150)
print("Plot saved!", flush=True)

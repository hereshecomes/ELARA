"""Bitwise CPU/GPU Adam equivalence + throughput of srm2 primitives (GPU 0)."""
import json, sys, time
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))
from srm2.ops import ext, gpu_adam

E = ext(); dev = torch.device('cuda:0'); out = {}
# ---- bitwise equivalence over several steps with repeated rows across steps
g0 = torch.Generator().manual_seed(0)
R, D = 200_000, 128
host = torch.randn(R, 3, D, generator=g0) * 0.05; host[:, 1:] = 0
gpu = host.clone().to(dev)
for step in range(1, 7):
    rows = torch.randperm(R, generator=g0)[:30000].sort().values
    grad = torch.randn(len(rows), D, generator=g0) * (10.0 ** (step % 3 - 2))
    E.cpu_adam(host, rows, grad, 1e-3, 0.9, 0.999, 1e-8, step, 16)
    gpu_adam(gpu, rows.to(dev), grad.to(dev), 1e-3, 0.9, 0.999, 1e-8, step)
torch.cuda.synchronize()
same = torch.equal(host, gpu.cpu())
diff = (host - gpu.cpu()).abs().max().item()
out['bitwise_equal'] = bool(same); out['max_abs_diff'] = diff
print('bitwise_equal', same, 'max_abs_diff', diff, flush=True)

# ---- throughput
R = 4_000_000
big = torch.zeros(R, 3, D)
big[:, 0].normal_(0, 0.01)
thr = {}
for nt in [1, 4, 8, 16, 24]:
    for n in [32768, 131072]:
        rows = torch.from_numpy(np.sort(np.random.default_rng(nt).choice(R, n, replace=False)))
        grad = torch.randn(n, D)
        w = torch.empty(n, D, pin_memory=True); f = torch.empty(n, 3, D, pin_memory=True)
        def tm(fn, reps=5):
            fn(); ts = []
            for _ in range(reps):
                t = time.perf_counter(); fn(); ts.append(time.perf_counter() - t)
            return float(np.median(ts)) / n * 1e9
        thr[f't{nt}_n{n}'] = {'adam_ns_row': tm(lambda: E.cpu_adam(big, rows, grad, 1e-3, .9, .999, 1e-8, 5, nt)),
                              'gather_w_ns_row': tm(lambda: E.cpu_gather(big, rows, w, 1, nt)),
                              'gather_full_ns_row': tm(lambda: E.cpu_gather(big, rows, f, 3, nt)),
                              'scatter_full_ns_row': tm(lambda: E.cpu_scatter(big, rows, f, nt))}
    print(nt, {k: {kk: round(vv, 1) for kk, vv in v.items()} for k, v in thr.items() if k.startswith(f't{nt}_')}, flush=True)
out['cpu'] = thr
# GPU adam throughput
pool = torch.zeros(1_000_000, 3, D, device=dev)
for n in [32768, 131072]:
    slots = torch.randperm(1_000_000, device=dev)[:n]; grad = torch.randn(n, D, device=dev)
    gpu_adam(pool, slots, grad, 1e-3, .9, .999, 1e-8, 3); torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(20): gpu_adam(pool, slots, grad, 1e-3, .9, .999, 1e-8, 3)
    torch.cuda.synchronize(); out[f'gpu_adam_us_n{n}'] = (time.perf_counter() - t) / 20 * 1e6
    print('gpu_adam n', n, round(out[f'gpu_adam_us_n{n}'], 1), 'us', flush=True)
# Embedding gather/backward with heavy duplication: F.embedding vs advanced indexing
for b in [4096, 16384]:
    ids = torch.from_numpy(np.random.default_rng(3).zipf(1.15, size=b * 26) % 200000).to(dev)
    u, inv = torch.unique(ids, return_inverse=True)
    w = torch.randn(len(u), D, device=dev, requires_grad=True); go = torch.randn(b * 26, D, device=dev)
    for name, fn in [('index', lambda: w[inv]), ('F.embedding', lambda: F.embedding(inv, w))]:
        for _ in range(3): w.grad = None; fn().backward(go)
        torch.cuda.synchronize(); t = time.perf_counter()
        for _ in range(10): w.grad = None; fn().backward(go)
        torch.cuda.synchronize(); out[f'emb_{name}_b{b}_ms'] = (time.perf_counter() - t) / 10 * 1e3
        print('emb', name, b, round(out[f'emb_{name}_b{b}_ms'], 3), 'ms', flush=True)
json.dump(out, open(sys.argv[1], 'w'), indent=1)

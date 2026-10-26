"""Exactness test: replay(init, log) == eager Adam (GPU and CPU paths), bitwise."""
import sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from srm2.ops import ext, gpu_adam
from srm2.gpu_ops import init_rows, replay_rows, ScalarTable

dev = torch.device('cuda:0'); D, K, seed, bound = 128, 3, 1234, 0.01
E = ext(); n = 5000
rows = torch.randint(0, 30_000_000, (n,), dtype=torch.int64).unique().to(dev); n = rows.numel()
st = torch.stack([torch.randint(1, 400, (n,)), torch.randint(400, 800, (n,)), torch.randint(800, 1200, (n,))], 1).to(torch.int32).to(dev)
g = torch.randn(n, K, D, device=dev) * 1e-2
cnt = torch.randint(0, K + 1, (n,), dtype=torch.int32, device=dev)
table = ScalarTable(1e-3, .9, .999, 1e-8, 1300, dev)
# init determinism (two launches, permuted order)
a = init_rows(rows, D, seed, bound); perm = torch.randperm(n, device=dev); b = init_rows(rows[perm], D, seed, bound)
print('init deterministic:', torch.equal(a[perm], b))
# eager GPU path
eg = init_rows(rows, D, seed, bound).contiguous()
ec = eg.cpu().clone()
for k in range(K):
    sel = (cnt > k).nonzero().flatten()
    for s_val in st[sel, k].unique().tolist():   # each logical step has its own scalars
        idx = sel[st[sel, k] == s_val]
        gpu_adam(eg, idx, g[idx, k].contiguous(), 1e-3, .9, .999, 1e-8, s_val)
        E.cpu_adam(ec, idx.cpu(), g[idx, k].cpu().contiguous(), 1e-3, .9, .999, 1e-8, s_val, 8)
torch.cuda.synchronize()
offs = (torch.arange(n, device=dev) * K)
rp = replay_rows(rows, g.reshape(-1, D).contiguous(), offs, st.contiguous(), cnt, D, K, seed, bound, table)
torch.cuda.synchronize()
print('replay == eager GPU:', torch.equal(rp, eg), ' replay == eager CPU:', torch.equal(rp.cpu(), ec),
      ' max|diff| GPU', (rp - eg).abs().max().item())

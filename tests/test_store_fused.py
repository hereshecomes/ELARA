"""ELARA fast path check: the fused C++ compact store must be indistinguishable from the reference NumPy store.
Random workload (per step: appends at the row's next log slot, materializations of logged/new/materialized rows,
gathers of full rows and logs) applied to both; every gather result, the index arrays, free list and footprint must
match exactly. Also times both at Criteo-like per-step volumes. CPU only."""
import sys, time
import numpy as np, torch
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1]))
from srm2.ops import ext
from srm2.store import CompactStore

E = ext(); rng = np.random.default_rng(1)
def run(R, D, Kmax, steps, n_app, n_full, timing=False, prefault=False):
    a = CompactStore(R, D, Kmax, E, 8, fused=False); b = CompactStore(R, D, Kmax, E, 8, fused=True)
    if prefault:
        for st in (a, b): st._fault('mat', min(st.mat_cap, (n_full + 1000) * (steps + 1))); st._fault('log', min(st.log_cap, (n_app + 1000) * (steps + 1)))
    cnt = np.zeros(R, np.int32); mat = np.zeros(R, bool); ta = tb = 0.0
    for s in range(1, steps + 1):
        cand = np.flatnonzero(~mat & (cnt < Kmax))
        ra = rng.choice(cand, min(n_app, len(cand)), replace=False)
        rows = torch.from_numpy(np.sort(ra).astype(np.int64)); ks = torch.from_numpy(cnt[rows.numpy()].astype(np.int32))
        g = torch.randn(len(rows), D)
        t0 = time.perf_counter(); a.append(rows, ks, g, s); t1 = time.perf_counter(); b.append(rows, ks, g, s); t2 = time.perf_counter()
        ta += t1 - t0; tb += t2 - t1; cnt[rows.numpy()] += 1
        rf = np.sort(rng.choice(R, n_full, replace=False)).astype(np.int64)
        src = torch.randn(len(rf), 3, D); rft = torch.from_numpy(rf)
        t0 = time.perf_counter(); a.write_full(rft, src); t1 = time.perf_counter(); b.write_full(rft, src); t2 = time.perf_counter()
        ta += t1 - t0; tb += t2 - t1; mat[rf] = True; cnt[rf] = 0
        # gathers
        mr = torch.from_numpy(np.sort(rng.choice(np.flatnonzero(mat), min(5000, mat.sum()), replace=False)).astype(np.int64))
        o1 = torch.empty(len(mr), 3, D); o2 = torch.empty(len(mr), 3, D)
        t0 = time.perf_counter(); a.gather_full(mr, o1); t1 = time.perf_counter(); b.gather_full(mr, o2); t2 = time.perf_counter()
        ta += t1 - t0; tb += t2 - t1; assert torch.equal(o1, o2)
        w1 = torch.empty(len(mr), D); w2 = torch.empty(len(mr), D); a.gather_w(mr, w1); b.gather_w(mr, w2); assert torch.equal(w1, w2)
        lr_ = np.flatnonzero(~mat & (cnt > 0)); lr_ = np.sort(rng.choice(lr_, min(5000, len(lr_)), replace=False)).astype(np.int64)
        c = torch.from_numpy(cnt[lr_].astype(np.int32)); off = torch.cumsum(c.long(), 0) - c.long(); rl = torch.from_numpy(lr_)
        l1 = torch.empty(int(c.sum()), D); l2 = torch.empty(int(c.sum()), D); s1 = torch.empty(len(rl), a.Kw, dtype=torch.int32); s2 = s1.clone()
        t0 = time.perf_counter(); a.gather_logs(rl, c, off, l1, s1); t1 = time.perf_counter(); b.gather_logs(rl, c, off, l2, s2); t2 = time.perf_counter()
        ta += t1 - t0; tb += t2 - t1; assert torch.equal(l1, l2) and torch.equal(s1, s2)
    assert (a.loc == b.loc).all() and (a.ent == b.ent).all() and (a.log_step[:a.log_top] == b.log_step[:b.log_top]).all()
    assert a.nfree == b.nfree and (a.free[:a.nfree] == b.free[:b.nfree]).all() and a.mat_top == b.mat_top and a.live == b.live
    assert a.footprint()['store_bytes'] == b.footprint()['store_bytes']
    assert torch.equal(a.mat[:a.mat_top], b.mat[:b.mat_top])
    return ta, tb

for Kmax in (1, 2, 3, 4):
    run(200_000, 32, Kmax, 30, 6000, 2000)
print('equivalence: OK (Kmax 1..4, 30 steps each)')
ta, tb = run(8_000_000, 64, 3, 12, 57_000, 20_000, timing=True)
print(f'timing per step (append 57K + write_full 20K + gathers 5K+5K), reference {ta / 12 * 1e3:.2f} ms vs fused {tb / 12 * 1e3:.2f} ms')
ta, tb = run(8_000_000, 64, 3, 12, 57_000, 20_000, timing=True, prefault=True)
print(f'same, arenas pre-faulted: reference {ta / 12 * 1e3:.2f} ms vs fused {tb / 12 * 1e3:.2f} ms')

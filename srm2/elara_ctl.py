"""ELARA critical-path-aware materialization controller (slow loop).

Every `every` steps, from measurements of the last window only (no future information):
  * stage utilizations u = busy / wall for: link-in, link-out, inbound host worker (gather + link wait),
    outbound host worker (store writes + link wait); a stage weighs on the step time only near saturation:
    w(u) = clip((u - u0) / (1 - u0), 0, 1)
  * per-direction marginal step cost of one byte:  tau_d = w(u_link_d) / bw_d + w(u_worker_d) * h_d,
    h_d = measured host seconds per byte moved by that worker in the window
  * per-table hazards h_t(c) = P(update c+1 | c updates) from the planner's GPU touch counters (all rows so far)
  * memory shadow price lambda (only with an enforced host budget): grows with footprint / budget
For a row of table t and threshold k (stay logged while c < k, materialize at c = k), with P = D*4 bytes:
  touch n (c = n-1 earlier updates):  c < k: in c*P, out P  |  c == k: in c*P, out 3P  |  c > k: in 3P, out 3P
  J_t(k) = sum_n Pr(>= n updates) * (tau_in * in_n + tau_out * out_n) + lambda * E[final record bytes]
k_t = argmin J_t(k) over 0..Kmax (ties -> the default threshold). Thresholds only move rows I/L -> M (never back),
so any threshold sequence keeps training exact.
"""
import time
import numpy as np
import torch


class ELARAController:
    def __init__(self, planner, executor, link, D, kmax, k_default, every=50, u0=0.5, budget_bytes=0, nmax=64, log=None):
        self.pl, self.ts, self.link, self.D = planner, executor, link, D
        self.kmax, self.k_default, self.every, self.u0, self.budget, self.nmax = kmax, k_default, every, u0, budget_bytes, nmax
        self.T = planner.toff.numel(); self.prev = None; self.history = log if log is not None else []
        self.bw = (link.bw if link.bw else None)

    def _snap(self):
        t, lk = self.ts.timers, self.link
        return {'wall': time.perf_counter(), 'gather': t['cpu_gather'], 'update': t['cpu_update'],
                'lin_busy': lk.busy['in'], 'lout_busy': lk.busy['out'], 'lin_wait': lk.waited['in'], 'lout_wait': lk.waited['out'],
                'bin': lk.bytes['in'], 'bout': lk.bytes['out']}

    def hazards(self):
        """[T, 17] counts of rows by update count (16 = 16+) for touched rows of each table -> hazards [T, 16]."""
        pl = self.pl; tc = pl.tcount.long()
        tid = torch.searchsorted(pl.toff, torch.arange(pl.R, device=pl.d), right=True) - 1
        key = tid * 17 + torch.clamp(tc, max=16)
        h = torch.bincount(key, minlength=self.T * 17).reshape(self.T, 17).double().cpu().numpy()
        ge = np.cumsum(h[:, ::-1], 1)[:, ::-1]                     # ge[:, c] = #rows with >= c updates
        with np.errstate(divide='ignore', invalid='ignore'):
            hz = np.where(ge[:, 1:16] > 0, ge[:, 2:17] / np.maximum(ge[:, 1:16], 1), 0.0)   # hz[:, c-1] = P(>=c+1 | >=c), c=1..15
        hz = np.concatenate([hz, hz[:, -1:]], 1)                   # extend the last hazard
        return hz, ge[:, 1]

    def thresholds(self, tau_in, tau_out, lam, hz):
        P = self.D * 4.0; K = self.kmax; N = self.nmax
        n = np.arange(1, N + 1); c = n - 1
        # Pr(>= n updates) for n = 1..N+1 per table
        hz_ext = np.concatenate([hz, np.repeat(hz[:, -1:], max(0, N - hz.shape[1]), 1)], 1)[:, :N]
        surv = np.concatenate([np.ones((hz.shape[0], 1)), np.cumprod(hz_ext, 1)], 1)       # [T, N+1]
        J = np.zeros((hz.shape[0], K + 1))
        for k in range(K + 1):
            inb = np.where(c < k, c * P, np.where(c == k, c * P, 3 * P))
            outb = np.where(c < k, P, 3 * P)
            J[:, k] = surv[:, :N] @ (tau_in * inb + tau_out * outb)
            fin = surv[:, :N] - surv[:, 1:N + 1]                      # Pr(final count = n)
            mem = np.where(n <= k, n * P, 3 * P)
            J[:, k] += lam * (fin @ mem)
        best = J.argmin(1)
        tie = np.isclose(J.min(1), J[:, min(self.k_default, K)], rtol=1e-3, atol=0)
        return np.where(tie, min(self.k_default, K), best), J

    def step(self, s):
        if s % self.every:
            return
        cur = self._snap()
        if self.prev is None:
            self.prev = cur; return
        dw = max(1e-6, cur['wall'] - self.prev['wall']); d = {k: cur[k] - self.prev[k] for k in cur}
        w = lambda u: float(np.clip((u - self.u0) / (1 - self.u0), 0, 1))
        u = {'link_in': d['lin_busy'] / dw, 'link_out': d['lout_busy'] / dw,
             'worker_in': (d['gather'] + d['lin_wait']) / dw, 'worker_out': (d['update'] + d['lout_wait']) / dw}
        hin = d['gather'] / max(1.0, d['bin']); hout = d['update'] / max(1.0, d['bout'])
        tau_in = (w(u['link_in']) / self.bw if self.bw else 0.0) + w(u['worker_in']) * hin
        tau_out = (w(u['link_out']) / self.bw if self.bw else 0.0) + w(u['worker_out']) * hout
        lam = 0.0
        if self.budget:
            fp = self.ts.store.footprint()['store_bytes']; r = fp / self.budget
            lam = (max(tau_in, tau_out, 1e-12) * 10.0) * max(0.0, (r - 0.7) / 0.3) ** 2
        hz, touched = self.hazards()
        k, J = self.thresholds(tau_in, tau_out, lam, hz)
        self.pl.kthr.copy_(torch.from_numpy(k.astype(np.int8)).to(self.pl.d))
        self.history.append({'step': s, 'u': {a: round(b, 3) for a, b in u.items()}, 'tau_in_ns_per_B': tau_in * 1e9,
                             'tau_out_ns_per_B': tau_out * 1e9, 'lambda': lam, 'k': k.tolist(),
                             'hz_c1_c4': [[round(x, 3) for x in hz[t, :4]] for t in range(hz.shape[0]) if touched[t] > 10000]})
        self.prev = cur

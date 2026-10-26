"""Capacity-tier stores for srm2.

DenseStore   : eager, preallocated table: one (w,m,v) record per vocabulary row ([R,3,D]), initialized at
               setup (as FBGEMM/TorchRec tables are). Lazy logs, if any, live in the first K slots of a
               row's record (K <= 3).
CompactStore : on-demand store. Materialized rows get a record in a MAT arena ([M,3,D]) only when first
               materialized; lazy gradient-log entries live in a LOG arena ([L,1,D]) with a free list; a
               row->slot index and a row->entries index map rows to records. Arenas are virtual
               reservations whose pages are faulted chunk-by-chunk just ahead of use, so resident memory
               tracks what the representation really needs. Used by the host-index lazy methods (K>=1) and by the
               on-demand eager baseline (K=0: materialize full state at first touch, DynamicEmb-style).
Both enforce an optional host-memory budget for the store (HostBudgetExceeded instead of an OOM kill).
All mutation happens on the outbound worker thread; the inbound worker only reads index entries of rows
that are disjoint from the rows being written (guaranteed by the executor's carry/ordering rules).
"""
import os, threading
import numpy as np
import torch

THP = os.environ.get('SRM2_THP', '0') == '1'   # host memory policy, applied identically to every store

I32, I64 = torch.int32, torch.int64
CHUNK = 64 << 20   # pre-fault granularity (bytes)


class HostBudgetExceeded(RuntimeError):
    pass


def proc_mem():
    """(VmRSS, VmHWM) of this process in bytes."""
    out = {}
    with open('/proc/self/status') as f:
        for line in f:
            if line.startswith(('VmRSS:', 'VmHWM:')):
                k, v = line.split(':'); out[k] = int(v.split()[0]) * 1024
    return out.get('VmRSS', 0), out.get('VmHWM', 0)


def t64(a):
    return torch.from_numpy(np.ascontiguousarray(a, dtype=np.int64))


class DenseStore:
    kind = 'dense'

    def __init__(self, R, D, K, lazy, E, threads, budget_bytes=0):
        self.R, self.D, self.Kw, self.E, self.th = R, D, max(K, 1), E, threads
        need = R * 3 * D * 4 + (R * self.Kw * 4 if lazy else 0)
        if budget_bytes and need > budget_bytes:
            raise HostBudgetExceeded(f'dense store needs {need / 2**30:.1f} GiB > host budget {budget_bytes / 2**30:.1f} GiB')
        assert not lazy or K <= 3, 'in-record logs support K<=3; use the compact store'
        self.host = torch.empty(R, 3, D)
        if THP: E.mem_hugepage(self.host)
        self.logstep = torch.zeros(R, self.Kw, dtype=I32) if lazy else None
        if lazy:
            (E.mem_populate(self.host, threads) if THP else E.cpu_prefault(self.host, threads))

    def gather_w(self, rows, out): self.E.cpu_gather(self.host, rows, out, 1, self.th)
    def gather_full(self, rows, out): self.E.cpu_gather(self.host, rows, out, 3, self.th)

    def gather_logs(self, rows, cnt, off, lg, ls):
        self.E.cpu_gather_var(self.host, rows, cnt, off, lg, self.th)
        torch.index_select(self.logstep, 0, rows, out=ls)

    def adam(self, rows, g, s, cfg):
        self.E.cpu_adam(self.host, rows, g, cfg['lr'], cfg['b1'], cfg['b2'], cfg['eps'], s, self.th)

    def append(self, rows, ks, g, s): self.E.cpu_scatter_slot(self.host, rows, ks, g, self.logstep, s, self.th)
    def write_full(self, rows, src): self.E.cpu_scatter(self.host, rows, src, self.th)

    def footprint(self):
        b = self.host.numel() * 4 + (self.logstep.numel() * 4 if self.logstep is not None else 0)
        return {'store': 'dense', 'store_bytes': b, 'mat_rows': self.R, 'live_log_entries': 0}


class CompactStore:
    kind = 'compact'

    def __init__(self, R, D, K, E, threads, budget_bytes=0, mat_cap=None, log_cap=None, ahead_bytes=0, fused=False):
        self.R, self.D, self.Kw, self.E, self.th, self.budget = R, D, max(K, 1), E, threads, budget_bytes
        self.loc = np.full(R, -1, np.int32)                       # row -> MAT slot
        self.ent = np.full((R, self.Kw), -1, np.int32)            # row -> log entry ids
        self.mat_cap = mat_cap or R; self.log_cap = log_cap or 2 * R
        self.mat = torch.empty(self.mat_cap, 3, D)                # virtual reservation; faulted on demand
        self.logs = torch.empty(self.log_cap, 1, D)
        if THP: E.mem_hugepage(self.mat); E.mem_hugepage(self.logs)
        self.log_step = np.zeros(self.log_cap, np.int32)
        self.mat_top = 0; self.mat_faulted = 0; self.log_top = 0; self.log_faulted = 0
        self.free = np.empty(self.log_cap, np.int32); self.nfree = 0; self.live = 0
        self.index_bytes = self.loc.nbytes + self.ent.nbytes
        # fused (ELARA fast path): index translation, allocation bookkeeping and copies in single C++ calls
        self.fused = fused
        self.loc_t, self.ent_t, self.log_step_t = torch.from_numpy(self.loc), torch.from_numpy(self.ent), torch.from_numpy(self.log_step)
        # allocate-ahead (optional): a background thread keeps each arena's faulted frontier up to ahead_bytes
        # beyond its allocation top, so first-touch page faults (kernel zeroing) leave the update critical path.
        # Same budget accounting (pre-faulted slack counts as resident; the thread stops rather than exceed it).
        self.ahead = ahead_bytes; self._lk = threading.Lock(); self._wake = threading.Event(); self._stop = False
        if ahead_bytes:
            threading.Thread(target=self._ahead_loop, daemon=True).start()

    # ---- accounting / budget ----
    def resident_bytes(self):
        return self.mat_faulted * 3 * self.D * 4 + self.log_faulted * self.D * 4 + self.index_bytes + self.log_top * 4

    def _fault(self, which, upto, background=False):
        rec = (3 if which == 'mat' else 1) * self.D * 4; step = max(1, CHUNK // rec)
        with self._lk:
            cur = self.mat_faulted if which == 'mat' else self.log_faulted
            if upto <= cur:
                return False
            new = min((self.mat_cap if which == 'mat' else self.log_cap), max(upto, cur + step))
            if self.budget and self.resident_bytes() + (new - cur) * rec > self.budget:
                if background:
                    return False
                raise HostBudgetExceeded(f'compact store would exceed host budget {self.budget / 2**30:.1f} GiB '
                                         f'(mat rows {self.mat_top}, live log entries {self.live})')
            t = self.mat if which == 'mat' else self.logs
            (self.E.mem_populate if THP else self.E.cpu_prefault)(t[cur:new], 2 if background else self.th)
            if which == 'mat': self.mat_faulted = new
            else: self.log_faulted = new
        if self.ahead and not background:
            self._wake.set()
        return True

    def _ahead_loop(self):
        while not self._stop:
            did = False
            for which, rec in (('mat', 3 * self.D * 4), ('log', self.D * 4)):
                top = self.mat_top if which == 'mat' else self.log_top
                cap = self.mat_cap if which == 'mat' else self.log_cap
                cur = self.mat_faulted if which == 'mat' else self.log_faulted
                tgt = min(cap, top + self.ahead // rec)
                if cur < tgt:
                    did |= self._fault(which, min(tgt, cur + max(1, CHUNK // rec)), background=True)
            if not did:
                self._wake.wait(0.005); self._wake.clear()

    def _alloc_mat(self, n):
        a = self.mat_top; assert a + n <= self.mat_cap, 'MAT arena full'
        self._fault('mat', a + n); self.mat_top += n
        return np.arange(a, a + n, dtype=np.int32)

    def _alloc_logs(self, n):
        k = min(n, self.nfree); ids = np.empty(n, np.int32)
        if k:
            ids[:k] = self.free[self.nfree - k:self.nfree]; self.nfree -= k
        m = n - k
        if m:
            a = self.log_top; assert a + m <= self.log_cap, 'LOG arena full'
            self._fault('log', a + m); ids[k:] = np.arange(a, a + m, dtype=np.int32); self.log_top += m
        return ids

    # ---- reads (inbound worker) ----
    def _slots(self, rows):
        sl = self.loc[rows.numpy()]
        assert (sl >= 0).all(), 'materialized row without a MAT slot'
        return t64(sl)

    def gather_w(self, rows, out):
        if self.fused: assert self.E.cs_gather_full(self.mat, self.loc_t, rows, out, 1, self.th) == 0, 'materialized row without a MAT slot'; return
        self.E.cpu_gather(self.mat, self._slots(rows), out, 1, self.th)

    def gather_full(self, rows, out):
        if self.fused: assert self.E.cs_gather_full(self.mat, self.loc_t, rows, out, 3, self.th) == 0, 'materialized row without a MAT slot'; return
        self.E.cpu_gather(self.mat, self._slots(rows), out, 3, self.th)

    def gather_logs(self, rows, cnt, off, lg, ls):
        if self.fused:
            assert self.E.cs_gather_logs(self.logs, self.ent_t, self.log_step_t, rows, cnt, off, lg, ls, self.th) == 0, 'missing log entry'
            return
        e = self.ent[rows.numpy()]; c = cnt.numpy()
        mask = np.arange(self.Kw)[None, :] < c[:, None]
        ids = e[mask]
        assert (ids >= 0).all(), 'missing log entry'
        self.E.cpu_gather(self.logs, t64(ids), lg, 1, self.th)
        ls.numpy()[:] = np.where(e >= 0, self.log_step[np.maximum(e, 0)], 0)

    # ---- writes (outbound worker) ----
    def adam(self, rows, g, s, cfg):
        self.E.cpu_adam(self.mat, self._slots(rows), g, cfg['lr'], cfg['b1'], cfg['b2'], cfg['eps'], s, self.th)

    def append(self, rows, ks, g, s):
        if self.fused:
            n = rows.numel(); ids = torch.from_numpy(self._alloc_logs(n)); self.live += n
            steps = s if torch.is_tensor(s) else torch.tensor([s], dtype=I32)
            self.E.cs_append(self.logs, self.ent_t, self.log_step_t, rows, ks, ids, g.view(n, self.D), steps, self.th); return
        r = rows.numpy(); k = ks.numpy(); ids = self._alloc_logs(len(r))
        self.ent[r, k] = ids; self.log_step[ids] = s; self.live += len(r)
        self.E.cpu_scatter(self.logs, t64(ids), g.view(len(r), 1, self.D), self.th)

    def write_full(self, rows, src):
        if self.fused:
            n = rows.numel(); self._fault('mat', self.mat_top + n)   # capacity for the worst case (all rows new)
            freed = torch.empty(n * self.Kw, dtype=I32)
            n_new, n_f = self.E.cs_write_full(self.mat, self.loc_t, self.ent_t, rows, src, self.mat_top, freed, self.th)
            self.mat_top += n_new; assert self.mat_top <= self.mat_cap, 'MAT arena full'
            if n_f:
                self.free[self.nfree:self.nfree + n_f] = freed[:n_f].numpy(); self.nfree += n_f; self.live -= n_f
            return
        r = rows.numpy(); sl = self.loc[r]; new = sl < 0
        if new.any():
            ns = self._alloc_mat(int(new.sum())); sl[new] = ns; self.loc[r[new]] = ns
        self.E.cpu_scatter(self.mat, t64(sl), src, self.th)
        e = self.ent[r]; f = e[e >= 0]                             # materialized rows drop their logs
        if len(f):
            self.free[self.nfree:self.nfree + len(f)] = f; self.nfree += len(f); self.live -= len(f)
            self.ent[r] = -1

    def footprint(self):
        D = self.D
        return {'store': 'compact', 'mat_rows': int(self.mat_top), 'live_log_entries': int(self.live),
                'store_bytes': int(self.mat_top * 3 * D * 4 + self.live * D * 4 + self.index_bytes + self.log_top * 4),
                'mat_bytes': int(self.mat_top * 3 * D * 4), 'log_bytes': int(self.live * D * 4),
                'index_bytes': int(self.index_bytes + self.log_top * 4), 'resident_bytes': int(self.resident_bytes())}

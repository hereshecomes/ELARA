"""Tiered sparse-state executor: GPU resident pool + capacity-tier store + two host workers.

Per step s (plan p from GPUPlanner):
  inbound worker G(s): gather S weights, F full states, L log entries into pinned staging -> (link model)
       -> H2D on the copy stream.
  GPU: assemble working weights W[u] from pool / staging / deterministic init / exact log replay;
       the caller runs forward+backward.
  GPU update: Adam for hits (in pool), admitted rows (into pool), T rows and log compactions (written
       back); S gradients and new log entries go to the host; evicted slots are read before reuse.
  outbound worker U(s): after D2H (and the link model), fused CPU Adam for S rows, log appends,
       full-row writes into the store.
Ordering: G(s) waits for U(s-depth) and for U(t) of any step t in (s-depth, s) whose plan raised a
carry-overflow barrier; rows recurring within depth-1 steps are carried on the GPU otherwise. All
arithmetic paths are bitwise identical, so any plan yields the same model.
"""
import queue, threading, time
import torch
from torch.profiler import record_function
from .ops import ext, gpu_adam
from .gpu_ops import init_rows, replay_rows, ScalarTable
from .store import DenseStore, CompactStore, HostBudgetExceeded, proc_mem

I32, I64 = torch.int32, torch.int64


class LinkModel:
    """Optional software model of a slower capacity-tier link (RDMA/CXL emulation; SIMULATED).
    Each direction is a FIFO server: B bytes occupy it for lat*ceil(B/chunk) + B/bw seconds.
    Staged data / host writes are published only after the modeled completion time."""

    def __init__(self, gbps=None, lat_us=0.0, chunk=128 * 1024):
        self.bw = gbps * 1e9 / 8 if gbps else None; self.lat = lat_us * 1e-6; self.chunk = chunk
        self.free_at = {'in': 0.0, 'out': 0.0}; self.busy = {'in': 0.0, 'out': 0.0}
        self.bytes = {'in': 0, 'out': 0}; self.waited = {'in': 0.0, 'out': 0.0}

    def transfer(self, direction, nbytes, start):
        self.bytes[direction] += int(nbytes)
        if self.bw is None or nbytes == 0:
            return
        t0 = max(start, self.free_at[direction]); dur = nbytes / self.bw + self.lat * -(-int(nbytes) // self.chunk)
        self.free_at[direction] = t0 + dur; self.busy[direction] += dur
        w0 = time.perf_counter()
        while time.perf_counter() < self.free_at[direction]:
            rem = self.free_at[direction] - time.perf_counter()
            if rem > 2e-4: time.sleep(rem - 1e-4)
        self.waited[direction] += time.perf_counter() - w0


class TieredSparse:
    def __init__(self, R, D, cfg, planner, device, threads=16, link=None, depth=2, store='auto', host_budget=0, ahead_bytes=0, fused=False):
        self.R, self.D, self.cfg, self.pl, self.d = R, D, cfg, planner, device
        self.E = ext(); self.threads = threads; self.link = link or LinkModel()
        self.K = planner.K; self.Kmax = getattr(planner, 'Kmax', planner.K); self.Kw = max(self.Kmax, 1); self.depth = depth
        self.pool = torch.zeros(planner.C, 3, D, device=device)
        self.store = self._make_store(R, D, store, host_budget, ahead_bytes, fused)
        self.table = ScalarTable(cfg['lr'], cfg['b1'], cfg['b2'], cfg['eps'], cfg['max_step'] + 1, device)
        self.copy = torch.cuda.Stream(device); self.d2h = torch.cuda.Stream(device); self.err = None; self.err_exc = None
        self.q_in = queue.Queue(); self.q_out = queue.Queue(); self.upd_done = {}; self.barriers = set()
        self.staged = {}
        self.timers = {'cpu_gather': 0.0, 'cpu_update': 0.0, 'cpu_adam': 0.0, 'cpu_wait_gpu': 0.0, 'gpu_wait_stage': 0.0}
        self.counts = {'h2d_bytes': 0, 'd2h_bytes': 0, 'cpu_adam_rows': 0, 'log_appends': 0, 'full_writes': 0, 'barriers': 0}
        self.footprint_trace = []
        self.workers = [threading.Thread(target=self._loop, args=(q,), daemon=True) for q in (self.q_in, self.q_out)]
        for w in self.workers: w.start()

    def _make_store(self, R, D, store, host_budget, ahead_bytes, fused):
        planner, threads = self.pl, self.threads
        if store == 'auto':
            store = 'compact' if planner.lazy else 'dense'
        assert store == 'dense' or planner.lazy, 'eager methods use the dense (preallocated) store'
        return (CompactStore(R, D, self.Kmax, self.E, threads, host_budget, ahead_bytes=ahead_bytes, fused=fused) if store == 'compact'
                else DenseStore(R, D, self.K, planner.lazy, self.E, threads, host_budget))

    # ---------------- setup ----------------
    def init_host(self, chunk=1 << 20):
        """Eager (dense) store: fill every row with the deterministic init (same bits as lazy init)."""
        h = self.store.host
        for a in range(0, self.R, chunk):
            b = min(self.R, a + chunk)
            rows = torch.arange(a, b, dtype=I64, device=self.d)
            h[a:b].copy_(init_rows(rows, self.D, self.cfg['seed'], self.cfg['bound']))
        torch.cuda.synchronize(self.d)

    def load_static(self):
        """Fill pinned static slots: from the eager table, or from the deterministic init (lazy)."""
        sr = self.pl.static_rows
        if sr is None:
            return
        n = sr.numel()
        if self.pl.lazy:
            self.pool[:n] = init_rows(sr, self.D, self.cfg['seed'], self.cfg['bound'])
            self.pl.code[sr] = self.pl.MAT   # resident rows count as materialized
        else:
            buf = torch.empty(n, 3, self.D, pin_memory=True)
            self.store.gather_full(sr.cpu(), buf); self.pool[:n] = buf.to(self.d)
        torch.cuda.synchronize(self.d)

    # ---------------- host workers ----------------
    def _loop(self, q):
        while True:
            task = q.get()
            if task is None:
                return
            try:
                task()
            except Exception as e:
                import traceback; self.err = traceback.format_exc(); self.err_exc = e

    def _check(self):
        if self.err:
            if isinstance(self.err_exc, HostBudgetExceeded):
                raise self.err_exc
            raise RuntimeError('worker failed:\n' + self.err)

    def _done_event(self, s):
        return self.upd_done.setdefault(s, threading.Event())

    def mark_barrier(self, s):
        self.barriers.add(s); self.counts['barriers'] += 1

    def enqueue_gather(self, p):
        """G(s): host gathers into pinned staging, then H2D on the copy stream."""
        ev = threading.Event(); self.staged[p.s] = [ev, None]
        c = p.cpu
        deps = [self._done_event(p.s - self.depth)] if p.s - self.depth >= 1 else []
        deps += [self._done_event(t) for t in range(max(1, p.s - self.depth + 1), p.s) if t in self.barriers]

        def task():
          with record_function('G_wait_deps'):
            for dp in deps: dp.wait()
          with record_function('G_gather_stage'):
              if self.err: ev.set(); return
              t0 = time.perf_counter(); D, st = self.D, self.store
              nS, nF, nL = len(c['S_row']), len(c['F_row']), len(c['L_row'])
              nlog = int(c['L_cnt'].sum()) if nL else 0
              sw = torch.empty(nS, D, pin_memory=True); ff = torch.empty(nF, 3, D, pin_memory=True)
              lg = torch.empty(nlog, D, pin_memory=True); ls = torch.empty(nL, self.Kw, dtype=I32, pin_memory=True)
              if nS: st.gather_w(c['S_row'], sw)
              if nF: st.gather_full(c['F_row'], ff)
              if nL: st.gather_logs(c['L_row'], c['L_cnt'], c['L_off'], lg, ls)
              self.timers['cpu_gather'] += time.perf_counter() - t0
              nb = (sw.numel() + ff.numel() + lg.numel()) * 4 + nL * 4 * self.Kw
              self.link.transfer('in', nb, t0)
              with torch.cuda.stream(self.copy):
                  g = tuple(t.to(self.d, non_blocking=True) for t in (sw, ff, lg, ls))
                  e = torch.cuda.Event(); e.record(self.copy)
              self.counts['h2d_bytes'] += nb
              self.staged[p.s][1] = (g, e); ev.set()
        self.q_in.put(task)

    def enqueue_update(self, s, out_ev, og, of, c):
        """U(s): wait for D2H, then CPU Adam for S rows, log appends, full-row writes."""
        done = self._done_event(s)
        for old in [k for k in self.upd_done if k < s - 2 * self.depth - 2]: self.upd_done.pop(old, None)

        def task():
          with record_function('U_update'):
              try:
                  t0 = time.perf_counter()
                  while not out_ev.query():
                      time.sleep(2e-5)
                  t1 = time.perf_counter(); self.timers['cpu_wait_gpu'] += t1 - t0
                  self.link.transfer('out', (og.numel() + of.numel()) * 4, t1)
                  t2 = time.perf_counter(); st = self.store
                  nS = len(c['S_row'])
                  if nS: st.adam(c['S_row'], og[:nS], s, self.cfg)
                  t3 = time.perf_counter(); self.timers['cpu_adam'] += t3 - t2
                  nA = len(c['A_row'])
                  if nA: st.append(c['A_row'], c['A_k'], og[nS:], s)
                  if len(c['W_row']): st.write_full(c['W_row'], of)
                  self.timers['cpu_update'] += time.perf_counter() - t2
                  self.counts['cpu_adam_rows'] += nS; self.counts['log_appends'] += nA; self.counts['full_writes'] += len(c['W_row'])
                  if s % 50 == 0:
                      rss, hwm = proc_mem(); self.footprint_trace.append({'step': s, 'rss': rss, **self.store.footprint()})
              finally:
                  done.set()
        self.q_out.put(task)

    # ---------------- GPU side ----------------
    def assemble(self, p):
        """Working weights W [n,D] for step s and the full-state buffers needed by update()."""
        self._check()
        ev = self.staged[p.s][0]; t0 = time.perf_counter(); ev.wait(); self.timers['gpu_wait_stage'] += time.perf_counter() - t0
        self._check()
        (sw, ff, lg, ls), e = self.staged.pop(p.s)[1]
        cur = torch.cuda.current_stream(self.d); cur.wait_event(e)
        for t in (sw, ff, lg, ls): t.record_stream(cur)
        D, cfg = self.D, self.cfg
        W = torch.empty(p.n, D, device=self.d)
        if p.hit_pos.numel(): W.index_copy_(0, p.hit_pos, self.pool[:, 0].index_select(0, p.hit_slot))
        fullN = init_rows(p.N_row, D, cfg['seed'], cfg['bound']) if p.N_row.numel() else None
        if fullN is not None: W.index_copy_(0, p.N_pos, fullN[:, 0])
        fullL = None
        if p.L_row.numel():
            fullL = replay_rows(p.L_row, lg, p.L_off, ls, p.L_cnt, D, self.Kw, cfg['seed'], cfg['bound'], self.table)
            W.index_copy_(0, p.L_pos, fullL[:, 0])
        if p.F_row.numel(): W.index_copy_(0, p.F_pos, ff[:, 0])
        if p.S_row.numel(): W.index_copy_(0, p.S_pos, sw)
        if p.Z_pos.numel(): W.index_fill_(0, p.Z_pos, 0.0)          # not-admitted rows (inexact filter baseline)
        return W, (fullN, fullL, ff)

    def update(self, p, G, bufs):
        """Apply step-s updates on GPU and schedule host-side work (no host-device syncs)."""
        cfg, s, D, dev, ix = self.cfg, p.s, self.D, self.d, p.ix
        a = (cfg['lr'], cfg['b1'], cfg['b2'], cfg['eps'], s)
        fullN, fullL, ff = bufs
        sel = lambda t, i: t.index_select(0, i)
        if p.hit_pos.numel(): gpu_adam(self.pool, p.hit_slot, sel(G, p.hit_pos), *a)
        ev_buf = sel(self.pool, p.ev_slot) if p.ev_slot.numel() else None   # read before slot reuse
        outs_g = [sel(G, p.S_pos)]; outs_full = []
        for X, full, pos in (('N', fullN, p.N_pos), ('L', fullL, p.L_pos)):
            if full is None:
                continue
            up = ix[X + '_upd']
            if up.numel(): gpu_adam(full, up, sel(G, sel(pos, up)), *a)
            if ix[X + '_adm'].numel(): self.pool.index_copy_(0, ix[X + '_adm_slot'], sel(full, ix[X + '_adm']))
            if ix[X + '_cmp'].numel(): outs_full.append(sel(full, ix[X + '_cmp']))
            if ix[X + '_app'].numel(): outs_g.append(sel(G, sel(pos, ix[X + '_app'])))
        if p.F_row.numel():
            gpu_adam(ff, torch.arange(p.F_row.numel(), device=dev), sel(G, p.F_pos), *a)
            if ix['F_adm'].numel(): self.pool.index_copy_(0, ix['F_adm_slot'], sel(ff, ix['F_adm']))
            if ix['F_nadm'].numel(): outs_full.append(sel(ff, ix['F_nadm']))
        if ev_buf is not None: outs_full.append(ev_buf)
        og_gpu = torch.cat(outs_g) if len(outs_g) > 1 else outs_g[0]
        of_gpu = torch.cat(outs_full) if outs_full else torch.empty(0, 3, D, device=dev)
        og = torch.empty(og_gpu.shape, pin_memory=True); of = torch.empty(of_gpu.shape, pin_memory=True)
        cur = torch.cuda.current_stream(dev); ev_c = torch.cuda.Event(); ev_c.record(cur)
        with torch.cuda.stream(self.d2h):   # separate stream per direction: no false H2D/D2H dependencies
            self.d2h.wait_event(ev_c)
            og.copy_(og_gpu, non_blocking=True); of.copy_(of_gpu, non_blocking=True)
            og_gpu.record_stream(self.d2h); of_gpu.record_stream(self.d2h)
            out_ev = torch.cuda.Event(); out_ev.record(self.d2h)
        self.counts['d2h_bytes'] += (og.numel() + of.numel()) * 4
        self.enqueue_update(s, out_ev, og, of, p.cpu)

    # ---------------- read-only access & drain ----------------
    def drain(self):
        for q in (self.q_in, self.q_out):
            done = threading.Event(); q.put(done.set); done.wait()
        self._check()

    def eval_weights(self, rows):
        """Current weights for arbitrary rows (call after drain()). rows: cuda int64 sorted unique."""
        pl, D, cfg = self.pl, self.D, self.cfg
        W = torch.empty(rows.numel(), D, device=self.d)
        sl = pl.slot_of[rows]; res = sl >= 0
        if res.any(): W[res] = self.pool[sl[res].long(), 0]
        nr = (~res).nonzero().flatten(); r = rows[nr]; c = pl.code[r]
        mat = c == pl.MAT
        if mat.any():
            rr = r[mat].cpu(); buf = torch.empty(rr.numel(), D, pin_memory=True)
            self.store.gather_w(rr, buf); W[nr[mat]] = buf.to(self.d)
        if pl.lazy and (~mat).any():
            rl = r[~mat]; cnt = c[~mat].to(I32); rc = rl.cpu(); cc = cnt.cpu()
            off = torch.cumsum(cc.long(), 0) - cc.long()
            lg = torch.empty(int(cc.sum()), D, pin_memory=True); ls = torch.empty(rc.numel(), self.Kw, dtype=I32, pin_memory=True)
            self.store.gather_logs(rc, cc, off, lg, ls)
            full = replay_rows(rl, lg.to(self.d), off.to(self.d), ls.to(self.d), cnt, D, self.Kw, cfg['seed'], cfg['bound'], self.table)
            W[nr[~mat]] = full[:, 0]
        if pl.admit_threshold > 1:   # inexact filter baseline: never-admitted rows read as the default (zero) vector
            na = (pl.seen[rows] < pl.admit_threshold) & ~res
            W[na] = 0.0
        return W

    def close(self):
        for q, w in zip((self.q_in, self.q_out), self.workers):
            q.put(None); w.join(timeout=10)

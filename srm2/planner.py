"""GPU-side placement planner (own thread, high-priority CUDA stream, runs ahead of training).

Per step it classifies every unique row of the batch into a service path and maintains residency
metadata. Numerics never depend on the plan: every path applies the same bitwise Adam update.

Row codes (non-resident rows): 0..K = number of logged gradients (0 = never touched; state is the
deterministic init), K+1 = MAT (materialized w,m,v in the capacity tier). Eager methods start every
row in MAT (host table initialized at setup) and never produce logs.

Residency: a pool of C full-state slots on GPU.
  base='static' : top rows from a profiling window are pinned (FixedHot / FAE-style)
  dyn='carry'   : only hazard carry (rows recurring in the next step stay resident until not needed)
  dyn='lru'|'lfu': every miss admitted; victims by key (DynamicEmb/HugeCTR-style caching)
  dyn='ttl'     : admit iff the row recurs within the lookahead window L; expire after last use (BagPipe)
Carry is applied by every policy: a non-resident row that recurs in step s+1 stays on GPU after its
step-s update (removes read-after-write hazards against asynchronous host updates).
Non-resident MAT rows not admitted: 'S' (stream weight, CPU Adam) or 'T' (full state, GPU Adam),
by st_mode in {'S','T','mix'} (mix: fraction t_frac of rows, chosen by row hash, go T).

Implementation: rows are classified with one stable sort by class id; data-dependent sizes are read
back in a few batched device->host copies (~3 syncs per step).
"""
import torch

I32, I64 = torch.int32, torch.int64
# class ids; sorted order = emission order used by the executor
C_HIT, C_NADM, C_NAPP, C_NCMP, C_LADM, C_LAPP, C_LCMP, C_FADM, C_FNADM, C_S, C_Z = range(11)


class Plan:
    pass


class GPUPlanner:
    def __init__(self, R, C, K, lazy, base, dyn, st_mode, t_frac, static_rows, L, device, lfu_decay=0.99,
                 admit_threshold=1, past_window=3):
        d = device; self.d = d; self.R, self.C, self.K, self.L = R, C, K, L
        self.MAT = K + 1; self.lazy, self.base, self.dyn, self.st_mode, self.t_frac = lazy, base, dyn, st_mode, t_frac
        self.slot_of = torch.full((R,), -1, dtype=I32, device=d)
        self.code = torch.full((R,), 0 if lazy else self.MAT, dtype=torch.int8, device=d)
        self.slot_row = torch.full((C,), -1, dtype=I64, device=d)
        self.pinned = torch.zeros(C, dtype=torch.bool, device=d)
        self.key = torch.zeros(C, dtype=torch.float32, device=d)
        self.last_seen = torch.full((R,), -1, dtype=I32, device=d)
        self.lfu_decay = lfu_decay
        self.static_rows = None; n0 = 0
        if base == 'static' and static_rows is not None and len(static_rows):
            n0 = len(static_rows); assert n0 <= C
            sr = static_rows.to(d)
            self.slot_of[sr] = torch.arange(n0, dtype=I32, device=d)
            self.slot_row[:n0] = sr; self.pinned[:n0] = True; self.static_rows = sr
        self.free = torch.arange(C - 1, n0 - 1, -1, dtype=I64, device=d)   # free-slot stack (pop from end)
        self.n_free = C - n0
        self.dyn_slots = torch.empty(0, dtype=I64, device=d)              # non-pinned resident slots (carry/ttl)
        self._c8 = [torch.tensor(v, dtype=torch.int8, device=d) for v in range(11)]
        # inexact admission filter (TFRA/DeepRec CounterFilter-style baseline): a row is created only at its
        # admit_threshold-th occurring step; before that it reads the default zero vector and is not trained.
        self.admit_threshold = admit_threshold
        self.seen = torch.zeros(R if admit_threshold > 1 else 1, dtype=I32, device=d)
        self.past_window = past_window   # 'window' policy (ScratchPipe-style): keep rows used in the last P steps
        self.stats = {'adm': 0, 'ev': 0, 'hits': 0, 'N': 0, 'L': 0, 'F': 0, 'S': 0, 'overflow': 0, 'syncs': 0}
        self.diag = None   # optional diagnostics (enable_diag): per-row update counts, eviction-time histograms

    def enable_diag(self, table_off):
        """Per-row saturating update counter (uint8), and histograms (GPU, no syncs) of the total update count of
        rows at eviction (split by whether the row was log-representable before admission), at compaction, and of
        final counts per table (for hazard estimates)."""
        d = self.d
        self.diag = {'tc': torch.zeros(self.R, dtype=torch.uint8, device=d), 'adm_code': torch.zeros(self.C, dtype=torch.int8, device=d),
                     'h_ev_from_lazy': torch.zeros(17, dtype=I64, device=d), 'h_ev_from_mat': torch.zeros(17, dtype=I64, device=d),
                     'h_cmp': torch.zeros(17, dtype=I64, device=d), 'toff': table_off.to(d)}

    def observe(self, step, u):
        self.last_seen[u] = step

    def _host(self, *ts):
        self.stats['syncs'] += 1
        return torch.stack([t.reshape(()).to(I64) for t in ts]).tolist()

    def _drop(self, admit, in_next, n_adm, k, s):
        """Overflow: un-admit k admitted rows, non-carry first. Returns True if carry rows had to be dropped;
        the plan then raises a barrier (later gathers wait for this step's host update), keeping exactness."""
        pri = (~admit).to(I32) * 4 + admit.to(I32) * (2 - 2 * in_next.to(I32))   # carry=0, other adm=2, rest=4
        o = torch.argsort(pri, stable=True); drop = o[n_adm - k:n_adm]
        n_carry_drop, = self._host(in_next[drop].sum())
        admit[drop] = False; self.stats['overflow'] += k
        if n_carry_drop: self.stats['carry_barriers'] = self.stats.get('carry_barriers', 0) + 1
        return n_carry_drop > 0

    def plan(self, s, u, u_nexts):
        """u_nexts: unique rows of the next d-1 steps (pipeline depth d); rows recurring there are carried."""
        d, K, MAT, C = self.d, self.K, self.MAT, self.C
        n = u.numel()
        sl = self.slot_of[u]; hit = sl >= 0
        in_next = torch.zeros(n, dtype=torch.bool, device=d)
        for un in u_nexts:
            j = torch.searchsorted(un, u).clamp(max=un.numel() - 1)
            in_next |= un[j] == u
        isZ = None
        if self.admit_threshold > 1:   # inexact filter: rows below the occurrence threshold are not created
            self.seen[u] += 1
            isZ = ~hit & (self.seen[u] < self.admit_threshold) & (self.code[u] == 0)
        barrier = False
        if self.dyn in ('lru', 'lfu', 'window'):
            admit = ~hit
        elif self.dyn == 'ttl':
            admit = ~hit & (self.last_seen[u] > s)
        else:
            admit = ~hit & in_next
        if isZ is not None:
            admit &= ~isZ
        # ---- protection: pinned slots and slots used next step ----
        protect = self.pinned.clone()
        for un in u_nexts:
            sn = self.slot_of[un].long()
            m = torch.zeros(C, dtype=I32, device=d); m.scatter_add_(0, sn.clamp(min=0), (sn >= 0).to(I32)); protect |= m > 0
        # ---- evictions (after this step's update) ----
        keep_dyn = None
        if self.dyn in ('carry', 'ttl'):
            dl = self.dyn_slots
            exp = ~protect[dl]
            if self.dyn == 'ttl':
                exp &= self.last_seen[self.slot_row[dl]] <= s
            n_exp, n_adm = self._host(exp.sum(), admit.sum())
            o = torch.argsort((~exp).to(torch.int8), stable=True)       # expired first
            ev_slot = dl[o[:n_exp]]; keep_dyn = dl[o[n_exp:]]
            need = n_adm - (self.n_free + n_exp)
            if need > 0 and keep_dyn.numel():   # preempt TTL residents not needed within the pipeline window
                pre = ~protect[keep_dyn]; n_pre, = self._host(pre.sum()); take = min(need, n_pre)
                if take:
                    o2 = torch.argsort((~pre).to(torch.int8), stable=True)
                    ev_slot = torch.cat([ev_slot, keep_dyn[o2[:take]]]); keep_dyn = keep_dyn[o2[take:]]
                    n_exp += take; need -= take; self.stats['preempt'] = self.stats.get('preempt', 0) + take
            if need > 0:
                barrier = self._drop(admit, in_next, n_adm, need, s); n_adm -= need
        else:   # lru / lfu: evict as many as needed, never rows used in this or the next step
            m = torch.zeros(C, dtype=I32, device=d); m.scatter_add_(0, sl.long().clamp(min=0), hit.to(I32)); protect |= m > 0
            cand = (self.slot_row >= 0) & ~protect
            if self.dyn == 'window':   # ScratchPipe-style: rows unused for past_window steps leave the scratchpad
                stale = cand & (self.key < s - self.past_window)
                n_adm, n_cand, n_stale = self._host(admit.sum(), cand.sum(), stale.sum())
            else:
                n_adm, n_cand = self._host(admit.sum(), cand.sum()); n_stale = 0
            need = n_adm - self.n_free; n_exp = max(n_stale, max(0, min(need, n_cand)))
            ev_slot = torch.empty(0, dtype=I64, device=d)
            if n_exp:
                k = torch.where(cand, self.key, torch.full_like(self.key, float('inf')))
                ev_slot = torch.topk(k, n_exp, largest=False).indices
            if need > n_exp:
                barrier = self._drop(admit, in_next, n_adm, need - n_exp, s); n_adm -= need - n_exp
        ev_row = self.slot_row[ev_slot]
        if self.diag is not None:   # diagnostics: counts are updated with this step's touches below
            dg = self.diag; tc = dg['tc']; tc[u] = torch.clamp(tc[u].to(torch.int16) + 1, max=255).to(torch.uint8)
            if ev_slot.numel():
                lazy_before = dg['adm_code'][ev_slot] <= K
                h = torch.clamp(tc[ev_row].long(), max=16)
                dg['h_ev_from_lazy'] += torch.bincount(h[lazy_before], minlength=17)[:17]
                dg['h_ev_from_mat'] += torch.bincount(h[~lazy_before], minlength=17)[:17]
        # ---- slot assignment from the free stack (evicted slots become free after the update) ----
        pool_free = torch.cat([self.free, ev_slot]); nf = self.n_free + n_exp
        adm_slots = pool_free[nf - n_adm:]; self.free = pool_free[:nf - n_adm]; self.n_free = nf - n_adm
        # ---- one class id per row; one stable sort makes every group contiguous ----
        c = self.code[u]
        if self.st_mode == 'S':
            wantT = torch.zeros(n, dtype=torch.bool, device=d)
        elif self.st_mode == 'T':
            wantT = torch.ones(n, dtype=torch.bool, device=d)
        else:
            wantT = (u * 2654435761) % 1000003 < int(self.t_frac * 1000003)
        if self.lazy:
            isN = c == 0; isL = (c >= 1) & (c <= K)
        else:
            isN = isL = torch.zeros(n, dtype=torch.bool, device=d)
        app = c < K
        t8 = lambda v: self._c8[v]
        cls = torch.full((n,), C_S, dtype=torch.int8, device=d)
        cls = torch.where(~isN & ~isL & (admit | wantT), torch.where(admit, t8(C_FADM), t8(C_FNADM)), cls)
        cls = torch.where(isL, torch.where(admit, t8(C_LADM), torch.where(app, t8(C_LAPP), t8(C_LCMP))), cls)
        cls = torch.where(isN, torch.where(admit, t8(C_NADM), torch.where(app, t8(C_NAPP), t8(C_NCMP))), cls)
        cls = torch.where(hit, t8(C_HIT), cls)
        if isZ is not None:
            cls = torch.where(isZ, t8(C_Z), cls)
        order = torch.argsort(cls, stable=True)
        counts = torch.bincount(cls.long(), minlength=11).tolist(); self.stats['syncs'] += 1
        b_ = [0]
        for x in counts: b_.append(b_[-1] + x)
        rng = lambda lo, hi: order[b_[lo]:b_[hi + 1]]
        p = Plan(); p.s = s; p.u = u; p.n = n
        hp = rng(C_HIT, C_HIT); p.hit_pos, p.hit_slot = hp, sl[hp].long()
        Npos, Lpos, Fpos, Spos = rng(C_NADM, C_NCMP), rng(C_LADM, C_LCMP), rng(C_FADM, C_FNADM), rng(C_S, C_S)
        p.N_pos, p.N_row = Npos, u[Npos]; p.L_pos, p.L_row = Lpos, u[Lpos]; p.L_cnt = c[Lpos].to(I32)
        p.F_pos, p.F_row = Fpos, u[Fpos]; p.S_pos, p.S_row = Spos, u[Spos]
        p.Z_pos = rng(C_Z, C_Z); p.barrier = barrier; self.stats['Z'] = self.stats.get('Z', 0) + counts[C_Z]
        nNa, nNp, nNc, nLa, nLp, nLc, nFa, nFn = counts[1:9]
        assert nNa + nLa + nFa == n_adm, (nNa, nLa, nFa, n_adm)
        ar = lambda a_, b2: torch.arange(a_, b2, device=d)
        ix = {'N_adm': ar(0, nNa), 'N_app': ar(nNa, nNa + nNp), 'N_cmp': ar(nNa + nNp, nNa + nNp + nNc),
              'L_adm': ar(0, nLa), 'L_app': ar(nLa, nLa + nLp), 'L_cmp': ar(nLa + nLp, nLa + nLp + nLc),
              'F_adm': ar(0, nFa), 'F_nadm': ar(nFa, nFa + nFn)}
        ix['N_upd'] = torch.cat([ix['N_adm'], ix['N_cmp']]); ix['L_upd'] = torch.cat([ix['L_adm'], ix['L_cmp']])
        ix['N_adm_slot'] = adm_slots[:nNa]; ix['L_adm_slot'] = adm_slots[nNa:nNa + nLa]; ix['F_adm_slot'] = adm_slots[nNa + nLa:]
        p.ix = ix; p.ev_slot, p.ev_row = ev_slot, ev_row
        p.L_off = torch.cumsum(p.L_cnt.long(), 0) - p.L_cnt.long()
        # ---- metadata updates (logical state after step s) ----
        adm_rows = torch.cat([p.N_row[:nNa], p.L_row[:nLa], p.F_row[:nFa]])
        if ev_slot.numel():
            self.slot_of[ev_row] = -1; self.code[ev_row] = MAT; self.slot_row[ev_slot] = -1
        if n_adm:
            self.slot_of[adm_rows] = adm_slots.to(I32); self.slot_row[adm_slots] = adm_rows
            if self.diag is not None: self.diag['adm_code'][adm_slots] = self.code[adm_rows]
        if keep_dyn is not None:
            self.dyn_slots = torch.cat([keep_dyn, adm_slots])
        if self.lazy:
            lr_ = torch.cat([p.N_row[nNa:], p.L_row[nLa:]]); cc = self.code[lr_]
            if self.diag is not None:
                cmp_rows = lr_[cc >= K]
                if cmp_rows.numel(): self.diag['h_cmp'] += torch.bincount(torch.clamp(self.diag['tc'][cmp_rows].long(), max=16), minlength=17)[:17]
            self.code[lr_] = torch.where(cc >= K, torch.full_like(cc, MAT), cc + 1)
        if self.dyn in ('lru', 'window'):
            self.key[p.hit_slot] = float(s); self.key[adm_slots] = float(s)
        elif self.dyn == 'lfu':
            self.key.mul_(self.lfu_decay); self.key[p.hit_slot] += 1; self.key[adm_slots] = 1
        # ---- host work lists in emission order; one device->host copy ----
        A_row = torch.cat([p.N_row[nNa:nNa + nNp], p.L_row[nLa:nLa + nLp]])
        A_k = torch.cat([torch.zeros(nNp, dtype=I64, device=d), p.L_cnt[nLa:nLa + nLp].long()])
        W_row = torch.cat([p.N_row[nNa + nNp:], p.L_row[nLa + nLp:], p.F_row[nFa:], ev_row])
        names = ('S_row', 'F_row', 'L_row', 'L_cnt', 'L_off', 'A_row', 'A_k', 'W_row')
        parts = (p.S_row, p.F_row, p.L_row, p.L_cnt.long(), p.L_off, A_row, A_k, W_row)
        flat = torch.cat(parts).cpu(); self.stats['syncs'] += 1; p.cpu = {}; o = 0
        for k, t in zip(names, parts):
            v = flat[o:o + t.numel()]; o += t.numel()
            p.cpu[k] = v.to(I32) if k in ('L_cnt', 'A_k') else v
        st = self.stats; st['adm'] += n_adm; st['ev'] += ev_slot.numel(); st['hits'] += counts[C_HIT]
        st['N'] += Npos.numel(); st['L'] += Lpos.numel(); st['F'] += Fpos.numel(); st['S'] += Spos.numel()
        return p

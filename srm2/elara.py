"""ELARA — Exact Lazy Auxiliary-State Replay and Allocation (planner + executor + host arenas).

Logical state of every embedding row r (sparse Adam with lazy-row semantics and global-step bias correction):
  I   implicit      never updated; (w,m,v) = (init(r), 0, 0) is recomputable from (seed, r); nothing stored
  L_c logged        c <= Kmax updates; the c gradients and their logical steps are stored; (w,m,v) = replay
  M   materialized  full (w,m,v) record in the host MAT arena
Residency is orthogonal: a row may additionally own a GPU pool slot (static hot set or lookahead TTL/carry); its
authoritative state is then the slot. A row admitted from I/L keeps its gradient history (host log entries + the
gradients applied while resident, recorded on the GPU), so on eviction it can return to L instead of being forced
into M ("residency-aware representation"). Transitions: I->L, L->L, I/L->M, M->M; never M->L.

Placement of metadata (the "GPU-resident state directory"): the planner owns on the GPU the row code (I/L_c/M),
MAT slot (loc), log-entry ids (ent), the step of every log entry (lstep), both arena allocators (bump pointers + a
free stack for log entries, allocated without host synchronization), the pool residency maps and the per-slot
resident histories. The host capacity tier is two payload arenas addressed by physical ids, so every host
operation is a plain gather/scatter copy (no index structures, no allocation logic on the host critical path).

Every arithmetic path (resident Adam, transient Adam, replay, host Adam) is bitwise identical, so any sequence of
representation / residency / materialization decisions trains exactly the same model.
"""
import threading, time
import numpy as np
import torch
from torch.profiler import record_function
from .executor import TieredSparse, LinkModel
from .gpu_ops import init_rows, replay_rows
from .ops import gpu_adam
from .store import HostBudgetExceeded, CHUNK, proc_mem

I32, I64, I8 = torch.int32, torch.int64, torch.int8
# classes (sorted order = emission order)
C_HIT, C_NADM, C_NAPP, C_NCMP, C_LADM, C_LAPP, C_LCMP, C_FADM, C_FNADM, C_S = range(10)


class Plan:
    pass


# ----------------------------------------------------------------------------------------------------------------
class ArenaStore:
    """Host payload arenas: MAT records [Mcap,3,D] and log entries [Lcap,1,D] (virtual reservations, pages faulted
    in chunks just ahead of use; optional background allocate-ahead). Enforced host budget on faulted bytes."""
    kind = 'arena'

    def __init__(self, R, D, E, threads, budget_bytes=0, mat_cap=None, log_cap=None, ahead_bytes=0):
        self.R, self.D, self.E, self.th, self.budget = R, D, E, threads, budget_bytes
        self.mat_cap = mat_cap or R; self.log_cap = log_cap or 2 * R
        self.mat = torch.empty(self.mat_cap + 1, 3, D); self.logs = torch.empty(self.log_cap + 1, 1, D)   # last rows = trash
        import os
        self.thp = os.environ.get('SRM2_THP', '0') == '1'
        if self.thp: E.mem_hugepage(self.mat); E.mem_hugepage(self.logs)
        self.faulted = {'mat': 0, 'log': 0}; self.top = {'mat': 0, 'log': 0}; self.live_logs = 0
        self._lk = threading.Lock(); self._wake = threading.Event(); self.ahead = ahead_bytes
        if ahead_bytes:
            threading.Thread(target=self._ahead_loop, daemon=True).start()

    def _rec(self, which):
        return (3 if which == 'mat' else 1) * self.D * 4

    def resident_bytes(self):
        return self.faulted['mat'] * self._rec('mat') + self.faulted['log'] * self._rec('log')

    def fault(self, which, upto, background=False):
        rec = self._rec(which); step = max(1, CHUNK // rec); cap = self.mat_cap if which == 'mat' else self.log_cap
        with self._lk:
            cur = self.faulted[which]
            if upto <= cur:
                return False
            new = min(cap, max(upto, cur + step))
            if self.budget and self.resident_bytes() + (new - cur) * rec > self.budget:
                if background:
                    return False
                raise HostBudgetExceeded(f'arena store would exceed host budget {self.budget / 2**30:.1f} GiB '
                                         f'(mat rows {self.top["mat"]}, live log entries {self.live_logs})')
            t = self.mat if which == 'mat' else self.logs
            if self.thp: self.E.mem_populate(t[cur:new], 2 if background else self.th)
            else: self.E.cpu_prefault(t[cur:new], 2 if background else self.th)
            self.faulted[which] = new
        if self.ahead and not background:
            self._wake.set()
        return True

    def _ahead_loop(self):
        while True:
            did = False
            for which in ('mat', 'log'):
                rec = self._rec(which); cap = self.mat_cap if which == 'mat' else self.log_cap
                tgt = min(cap, self.top[which] + self.ahead // rec)
                if self.faulted[which] < tgt:
                    did |= self.fault(which, min(tgt, self.faulted[which] + max(1, CHUNK // rec)), background=True)
            if not did:
                self._wake.wait(0.005); self._wake.clear()

    def set_tops(self, mat_top, log_top, live_logs):
        self.top['mat'], self.top['log'], self.live_logs = mat_top, log_top, live_logs

    def footprint(self):
        D = self.D
        return {'store': 'elara_arena', 'mat_rows': int(self.top['mat']), 'live_log_entries': int(self.live_logs),
                'log_arena_entries': int(self.top['log']),
                'store_bytes': int(self.top['mat'] * 3 * D * 4 + self.live_logs * D * 4),
                'mat_bytes': int(self.top['mat'] * 3 * D * 4), 'log_bytes': int(self.live_logs * D * 4),
                'index_bytes': 0, 'resident_bytes': int(self.resident_bytes())}


# ----------------------------------------------------------------------------------------------------------------
class ELARAPlanner:
    def __init__(self, R, C, Kmax, base, dyn, st_mode, static_rows, L, device, table_off, kthr_init,
                 evict_logs=True, log_cap=None, fused_dir=True, dir_impl=None, evl_gate_frac=0.75):
        d = device; self.d = d; self.R, self.C, self.Kmax, self.L = R, C, Kmax, L; self.K = Kmax
        self.MAT = Kmax + 1; self.lazy = True; self.base, self.dyn, self.st_mode = base, dyn, st_mode
        assert dyn in ('ttl', 'carry') and st_mode in ('S', 'T')
        # residency-aware representation: 0 off, 1 always on, 2 gated by memory pressure (switched on, once, when the
        # capacity-tier store reaches evl_gate_frac of its enforced budget; never on without a budget)
        self.evl_mode = int(evict_logs); self.evl_gate_frac = evl_gate_frac; self.gate = None
        self.evict_logs = self.evl_mode == 1; self.admit_threshold = 1; self.diag = None
        # ---- state directory (GPU) ----
        self.code = torch.zeros(R, dtype=I8, device=d)                       # 0 = I, 1..Kmax = L_c, MAT = M
        self.loc = torch.full((R + 1,), -1, dtype=I32, device=d)             # MAT slot (+1 trash row)
        self.ent = torch.full(((R + 1) * Kmax,), -1, dtype=I32, device=d)    # log-entry ids, row-major [R+1, Kmax]
        self.log_cap = log_cap or 2 * R; self.mat_trash = R             # MAT arena row R = trash (arena has R+1 rows)
        self.lstep = torch.zeros(self.log_cap + 1, dtype=I32, device=d)      # logical step of each entry (+ trash)
        self.lfree = torch.zeros(self.log_cap + 1, dtype=I32, device=d)      # free stack of log ids (+ trash)
        self.n_lfree = torch.zeros(1, dtype=I64, device=d); self.log_top = torch.zeros(1, dtype=I64, device=d)
        self.mat_top = torch.zeros(1, dtype=I64, device=d); self.live = torch.zeros(1, dtype=I64, device=d)
        self.toff = table_off.to(d); self.kthr = kthr_init.to(d).to(I8)     # per-table materialization threshold
        self.tcount = torch.zeros(R, dtype=torch.uint8, device=d)            # saturating update counter (hazards)
        # ---- residency (GPU) ----
        self.slot_of = torch.full((R,), -1, dtype=I32, device=d)
        self.slot_row = torch.full((C,), -1, dtype=I64, device=d)
        self.pinned = torch.zeros(C, dtype=torch.bool, device=d)
        self.last_seen = torch.full((R,), -1, dtype=I32, device=d)
        self.static_rows = None; n0 = 0
        if base == 'static' and static_rows is not None and len(static_rows):
            n0 = len(static_rows); assert n0 <= C
            sr = static_rows.to(d)
            self.slot_of[sr] = torch.arange(n0, dtype=I32, device=d)
            self.slot_row[:n0] = sr; self.pinned[:n0] = True; self.static_rows = sr
        self.n0 = n0; self.Cdyn = C - n0
        self.free = torch.arange(C - 1, n0 - 1, -1, dtype=I64, device=d); self.n_free = C - n0
        self.dyn_slots = torch.empty(0, dtype=I64, device=d)
        # resident histories of dynamic slots: host-log count at admission (-1 = not log-representable) and the
        # number of gradients recorded on the GPU while resident (payloads live in the executor's glog)
        # adm_c0: host log entries of the row at admission (-1 = admitted from M); released when it is evicted to M.
        # rok: the resident history is complete (representable as logs); rj: gradients recorded while resident.
        self.adm_c0 = torch.full((C,), -1, dtype=I8, device=d); self.rj = torch.zeros(C, dtype=I8, device=d)
        self.rok = torch.zeros(C, dtype=I8, device=d)
        self.pstep = torch.zeros(self.Cdyn + 1, Kmax, dtype=I32, device=d)      # steps of the resident history (+ trash)
        self._c8 = [torch.tensor(v, dtype=I8, device=d) for v in range(10)]
        # directory implementation: 'cpp' (C++/ATen, default), 'triton' (fused kernels), 'ref' (torch reference)
        self.dir_impl = dir_impl or ('triton' if fused_dir else 'ref')
        self.fused_dir = self.dir_impl == 'triton'
        if self.fused_dir:
            from .elara_kernels import Directory; self.dirk = Directory(self)
        if self.dir_impl == 'cpp':
            from .ops import dir_ext; self.dx = dir_ext()
        self.stats = {'adm': 0, 'ev': 0, 'ev_to_logs': 0, 'hits': 0, 'N': 0, 'L': 0, 'F': 0, 'S': 0, 'overflow': 0,
                      'syncs': 0, 'compactions': 0, 'appends': 0}

    def observe(self, step, u):
        self.last_seen[u] = step

    def _host(self, *ts):
        self.stats['syncs'] += 1
        return torch.stack([t.reshape(()).to(I64) for t in ts]).tolist()

    def table_of(self, rows):
        return torch.searchsorted(self.toff, rows, right=True) - 1

    # ---- sync-free allocators ----
    def _alloc_logs(self, valid):
        """valid: bool [m] -> ids int64 [m] (garbage where ~valid); pops the free stack, then bumps log_top."""
        rank = torch.cumsum(valid.to(I64), 0) - 1; nf = self.n_lfree
        from_free = rank < nf
        fid = self.lfree[torch.clamp(nf - 1 - rank, min=0)].to(I64)
        ids = torch.where(from_free, fid, self.log_top + rank - nf)
        k = valid.sum().to(I64).reshape(1)
        used = torch.minimum(k, nf)
        self.log_top += k - used; self.n_lfree -= used; self.live += k
        return ids

    def _free_logs(self, rows, cnt):
        """Release the first cnt[i] entries of rows[i] (cnt may be 0) and clear their directory entries."""
        if rows.numel() == 0:
            return
        K = self.Kmax; kk = torch.arange(K, device=self.d)
        pos = rows[:, None] * K + kk[None, :]; e = self.ent[pos]
        valid = (kk[None, :] < cnt[:, None].to(I64)) & (e >= 0)
        rank = torch.cumsum(valid.reshape(-1).to(I64), 0) - 1
        dst = torch.where(valid.reshape(-1), self.n_lfree + rank, torch.full_like(rank, self.log_cap))
        self.lfree.scatter_(0, dst, e.reshape(-1))
        k = valid.sum().to(I64).reshape(1); self.n_lfree += k; self.live -= k
        # clear only the released entries (rows passed with cnt 0 keep their directory, e.g. rows evicted back to L)
        self.ent.scatter_(0, torch.where(valid, pos, torch.full_like(pos, self.R * K)).reshape(-1),
                          torch.full((pos.numel(),), -1, dtype=I32, device=self.d))

    def _alloc_mat(self, rows, mask=None):
        """MAT slots for rows (allocated in order for rows without one, restricted to mask); returns slots int64
        (rows outside mask keep whatever loc they have, possibly -1)."""
        cur = self.loc[rows].to(I64); new = cur < 0
        if mask is not None: new = new & mask
        rank = torch.cumsum(new.to(I64), 0) - 1
        sl = torch.where(new, self.mat_top + rank, cur)
        if rows.numel(): self.loc[rows] = sl.to(I32)
        self.mat_top += new.sum().to(I64).reshape(1)
        return sl

    # ---- per-step plan ----
    def plan(self, s, u, u_nexts):
        d, K, MAT, C = self.d, self.Kmax, self.MAT, self.C
        n = u.numel()
        if self.evl_mode == 2 and not self.evict_logs and self.gate is not None and self.gate():
            self.evict_logs = True; self.stats['evl_on_step'] = s
        sl = self.slot_of[u]; hit = sl >= 0
        in_next = torch.zeros(n, dtype=torch.bool, device=d)
        for un in u_nexts:
            j = torch.searchsorted(un, u).clamp(max=un.numel() - 1)
            in_next |= un[j] == u
        admit = ~hit & ((self.last_seen[u] > s) if self.dyn == 'ttl' else in_next)
        protect = self.pinned.clone()
        for un in u_nexts:
            sn = self.slot_of[un].long()
            m = torch.zeros(C, dtype=I32, device=d); m.scatter_add_(0, sn.clamp(min=0), (sn >= 0).to(I32)); protect |= m > 0
        # ---- eviction candidates (after this step's update); same TTL/carry residency as planner.py ----
        barrier = False; dl = self.dyn_slots
        exp = ~protect[dl]
        if self.dyn == 'ttl':
            exp &= self.last_seen[self.slot_row[dl]] <= s
        c = self.code[u]; thr = self.kthr[self.table_of(u)]
        isN = c == 0; isL = (c >= 1) & (c <= K); app = c < thr
        wantT = torch.ones(n, dtype=torch.bool, device=d) if self.st_mode == 'T' else torch.zeros(n, dtype=torch.bool, device=d)
        t8 = lambda v: self._c8[v]

        def classify(admit):
            cls = torch.full((n,), C_S, dtype=I8, device=d)
            cls = torch.where(~isN & ~isL & (admit | wantT), torch.where(admit, t8(C_FADM), t8(C_FNADM)), cls)
            cls = torch.where(isL, torch.where(admit, t8(C_LADM), torch.where(app, t8(C_LAPP), t8(C_LCMP))), cls)
            cls = torch.where(isN, torch.where(admit, t8(C_NADM), torch.where(app, t8(C_NAPP), t8(C_NCMP))), cls)
            return torch.where(hit, t8(C_HIT), cls)
        cls = classify(admit)
        # one host sync: eviction-candidate count + class counts (admissions = NADM + LADM + FADM)
        vals = self._host(exp.sum(), *torch.bincount(cls.long(), minlength=10))
        n_exp, counts = vals[0], vals[1:]
        n_adm = counts[C_NADM] + counts[C_LADM] + counts[C_FADM]
        o = torch.argsort((~exp).to(I8), stable=True)
        ev_slot = dl[o[:n_exp]]; keep_dyn = dl[o[n_exp:]]
        need = n_adm - (self.n_free + n_exp)
        if need > 0 and keep_dyn.numel():   # preempt TTL residents not needed within the pipeline window (rare)
            pre = ~protect[keep_dyn]; n_pre, = self._host(pre.sum()); take = min(need, n_pre)
            if take:
                o2 = torch.argsort((~pre).to(I8), stable=True)
                ev_slot = torch.cat([ev_slot, keep_dyn[o2[:take]]]); keep_dyn = keep_dyn[o2[take:]]
                n_exp += take; need -= take; self.stats['preempt'] = self.stats.get('preempt', 0) + take
        if need > 0:   # overflow (rare): un-admit (non-carry first); carry drops raise an exactness barrier; recount
            pri = (~admit).to(I32) * 4 + admit.to(I32) * (2 - 2 * in_next.to(I32))
            oo = torch.argsort(pri, stable=True); drop = oo[n_adm - need:n_adm]
            n_carry_drop, = self._host(in_next[drop].sum())
            admit[drop] = False; self.stats['overflow'] += need; n_adm -= need; barrier = n_carry_drop > 0
            cls = classify(admit); counts = self._host(*torch.bincount(cls.long(), minlength=10))
        order = torch.argsort(cls, stable=True)
        b_ = [0]
        for x in counts: b_.append(b_[-1] + x)
        rng = lambda lo, hi: order[b_[lo]:b_[hi + 1]]
        p = Plan(); p.s = s; p.u = u; p.n = n; p.barrier = barrier
        hp = rng(C_HIT, C_HIT); p.hit_pos, p.hit_slot = hp, sl[hp].long()
        Npos, Lpos, Fpos, Spos = rng(C_NADM, C_NCMP), rng(C_LADM, C_LCMP), rng(C_FADM, C_FNADM), rng(C_S, C_S)
        p.N_pos, p.N_row = Npos, u[Npos]; p.L_pos, p.L_row = Lpos, u[Lpos]; p.L_cnt = c[Lpos].to(I32)
        p.F_pos, p.F_row = Fpos, u[Fpos]; p.S_pos, p.S_row = Spos, u[Spos]
        nNa, nNp, nNc, nLa, nLp, nLc, nFa, nFn = counts[1:9]
        assert nNa + nLa + nFa == n_adm, (nNa, nLa, nFa, n_adm)
        kk = torch.arange(K, device=d)
        # replay inputs from the directory (read before any directory update of this step)
        cpp = self.dir_impl == 'cpp'
        if cpp:
            stv = [self.code, self.loc, self.ent, self.lstep, self.lfree, self.n_lfree, self.log_top, self.mat_top, self.live,
                   self.adm_c0, self.rj, self.pstep, self.kthr, self.toff, self.slot_of, self.slot_row, self.rok]
            pv = [self.R, K, self.log_cap, self.n0, self.Cdyn, MAT, self.mat_trash, int(self.evict_logs)]
            (p.L_steps, L_ids, p.hrec_dyn, p.hrec_j, ev_row, rep, evl_ids, evf_slots, p.ev_dyn) = self.dx.elara_dir_a(
                stv, pv, p.hit_slot, p.L_row, p.L_cnt, ev_slot, s)
            p.ev_slot, p.ev_row, p.ev_rep = ev_slot, ev_row, rep; nev = ev_slot.numel(); rel_ev = None
        elif self.fused_dir:
            from .elara_kernels import lsteps, hits
            p.L_steps, L_ids = lsteps(self, p.L_row, p.L_cnt)
        else:
            Lent = self.ent[p.L_row[:, None] * K + kk[None, :]]
            p.L_steps = torch.where(Lent >= 0, self.lstep[Lent.clamp(min=0).long()], torch.zeros_like(Lent))
            L_ids = torch.where(kk[None, :] < p.L_cnt[:, None], Lent, torch.full_like(Lent, -1))
        # ---- (1) hits extend their slot's resident history while it stays representable ----
        hs = p.hit_slot
        if cpp:
            pass
        elif self.fused_dir:
            p.hrec_dyn, p.hrec_j = hits(self, hs, s)
        else:
          dynh = hs >= self.n0
          c0h = self.adm_c0[hs].to(I64); jh = self.rj[hs].to(I64); okh = self.rok[hs] != 0
          rec = (dynh & okh & (c0h + jh < K)) if self.evict_logs else torch.zeros_like(dynh)
          p.hrec_dyn = torch.where(rec, hs - self.n0, torch.full_like(hs, self.Cdyn))   # trash row Cdyn
          p.hrec_j = torch.where(rec, jh, torch.zeros_like(jh))
          if hs.numel():
            self.pstep.index_put_((p.hrec_dyn, p.hrec_j), torch.full_like(p.hrec_j, s, dtype=I32))
            self.rj[hs] = (jh + rec.to(I64)).to(I8)
            self.rok[hs] = (okh & ~(dynh & ~rec)).to(I8)
        # ---- (2) eviction representation: back to L if the full history fits the row's threshold, else M ----
        #      (masks + trash destinations; no host synchronization)
        if cpp:
            pass
        else:
          ev_row = self.slot_row[ev_slot]; nev = ev_slot.numel()
          rel_ev = None
        if cpp:
            pass
        elif self.fused_dir:
            p.ev_slot, p.ev_row = ev_slot, ev_row; p.ev_dyn = ev_slot - self.n0
            if nev:
                self.slot_of[ev_row] = -1; self.slot_row[ev_slot] = -1
            thr_ev = self.kthr[self.table_of(ev_row)] if nev else torch.empty(0, dtype=I8, device=d)
            rep, evl_ids, evf_slots, rel_ev = self.dirk.evict(ev_slot, ev_row, thr_ev)
            p.ev_rep = rep
        else:
          ev_c0 = self.adm_c0[ev_slot].to(I64); ev_j = self.rj[ev_slot].to(I64)
          if self.evict_logs and nev:
              rep = (self.rok[ev_slot] != 0) & (ev_c0 + ev_j <= self.kthr[self.table_of(ev_row)].to(I64))
          else:
              rep = torch.zeros(nev, dtype=torch.bool, device=d)
          p.ev_slot, p.ev_row, p.ev_rep = ev_slot, ev_row, rep; p.ev_dyn = ev_slot - self.n0
          if nev:
              self.slot_of[ev_row] = -1; self.slot_row[ev_slot] = -1
          kg = kk[None, :].expand(nev, K)
          evl_valid = rep[:, None] & (kg < ev_j[:, None])
          evl_ids = self._alloc_logs(evl_valid.reshape(-1)).reshape(nev, K)
          evl_ids = torch.where(evl_valid, evl_ids, torch.full_like(evl_ids, -1))
          if nev:
              pos = torch.where(evl_valid, ev_row[:, None] * K + ev_c0.clamp(min=0)[:, None] + kg, torch.full_like(kg, self.R * K))
              self.ent.scatter_(0, pos.reshape(-1), evl_ids.reshape(-1).to(I32))
              dst = torch.where(evl_valid, evl_ids, torch.full_like(evl_ids, self.log_cap)).reshape(-1)
              self.lstep.scatter_(0, dst, self.pstep[p.ev_dyn].reshape(-1))
          # rows going to M: MAT slot (allocated only for them), release their host logs; to-L rows get the trash slot
          ev_slot_mat = self._alloc_mat(ev_row, mask=~rep)
          evf_slots = torch.where(rep, torch.full_like(ev_slot_mat, self.mat_trash), ev_slot_mat)
          if nev:
              self._free_logs(ev_row, torch.where(rep, torch.zeros_like(ev_c0), ev_c0.clamp(min=0)))
              self.code[ev_row] = torch.where(rep, (ev_c0 + ev_j).clamp(max=127).to(I8), torch.full_like(ev_row, MAT, dtype=I8))
        # ---- (3) admissions (slots from the free stack; evicted slots are reused after the update reads them) ----
        pool_free = torch.cat([self.free, ev_slot]); nf = self.n_free + nev
        adm_slots = pool_free[nf - n_adm:]; self.free = pool_free[:nf - n_adm]; self.n_free = nf - n_adm
        adm_rows = torch.cat([p.N_row[:nNa], p.L_row[:nLa], p.F_row[:nFa]])
        ix = {}
        ar = lambda a_, b2: torch.arange(a_, b2, device=d)
        ix.update({'N_adm': ar(0, nNa), 'N_app': ar(nNa, nNa + nNp), 'N_cmp': ar(nNa + nNp, nNa + nNp + nNc),
                   'L_adm': ar(0, nLa), 'L_app': ar(nLa, nLa + nLp), 'L_cmp': ar(nLa + nLp, nLa + nLp + nLc),
                   'F_adm': ar(0, nFa), 'F_nadm': ar(nFa, nFa + nFn)})
        ix['N_upd'] = torch.cat([ix['N_adm'], ix['N_cmp']]); ix['L_upd'] = torch.cat([ix['L_adm'], ix['L_cmp']])
        ix['N_adm_slot'] = adm_slots[:nNa]; ix['L_adm_slot'] = adm_slots[nNa:nNa + nLa]; ix['F_adm_slot'] = adm_slots[nNa + nLa:]
        p.ix = ix; p.L_off = torch.cumsum(p.L_cnt.long(), 0) - p.L_cnt.long()
        lazy_n = nNa + nLa
        p.adm_rec_dyn = adm_slots[:lazy_n] - self.n0 if self.evict_logs else torch.full((lazy_n,), self.Cdyn, dtype=I64, device=d)
        if n_adm:
            self.slot_of[adm_rows] = adm_slots.to(I32); self.slot_row[adm_slots] = adm_rows
            c_adm = self.code[adm_rows].to(I64)
            lazy_adm = torch.cat([torch.ones(lazy_n, dtype=torch.bool, device=d), torch.zeros(nFa, dtype=torch.bool, device=d)]) & self.evict_logs
            isIL = torch.cat([torch.ones(lazy_n, dtype=torch.bool, device=d), torch.zeros(nFa, dtype=torch.bool, device=d)])
            self.adm_c0[adm_slots] = torch.where(isIL, c_adm, torch.full_like(c_adm, -1)).to(I8)   # host logs (released at eviction to M)
            self.rj[adm_slots] = lazy_adm.to(I8); self.rok[adm_slots] = lazy_adm.to(I8)
            if lazy_n and self.evict_logs:
                self.pstep[p.adm_rec_dyn, 0] = s
        self.dyn_slots = torch.cat([keep_dyn, adm_slots])
        # ---- (4) non-resident lazy rows: append (stay L) or compact (to M) ----
        a_rows = torch.cat([p.N_row[nNa:nNa + nNp], p.L_row[nLa:nLa + nLp]])
        a_k = torch.cat([torch.zeros(nNp, dtype=I64, device=d), p.L_cnt[nLa:nLa + nLp].long()])
        c_rows = torch.cat([p.N_row[nNa + nNp:], p.L_row[nLa + nLp:]])
        c_cnt = torch.cat([torch.zeros(nNc, dtype=I64, device=d), p.L_cnt[nLa + nLp:].long()])
        if cpp:
            a_ids, c_slots = self.dx.elara_dir_b(stv, pv, a_rows, a_k, c_rows, c_cnt, s)
        elif self.fused_dir:   # all allocations of the step, then all releases
            a_ids = self.dirk.appends(a_rows, a_k, s)
            c_slots = self.dirk.compact(c_rows, c_cnt)
            self.dirk.release([rel_ev, (c_rows, c_cnt)])
        else:
            a_ids = self._alloc_logs(torch.ones(a_rows.numel(), dtype=torch.bool, device=d))
            if a_rows.numel():
                self.ent[a_rows * K + a_k] = a_ids.to(I32); self.lstep[a_ids] = s
                self.code[a_rows] = (a_k + 1).to(I8)
            c_slots = self._alloc_mat(c_rows)
            if c_rows.numel():
                self._free_logs(c_rows, c_cnt); self.code[c_rows] = MAT
        self.tcount[u] = torch.clamp(self.tcount[u].to(torch.int16) + 1, max=255).to(torch.uint8)
        # ---- packed eviction outputs: rows going to M first (their MAT slots), valid resident-history entries first
        #      (rows going back to L); the executor moves only the first nev - n_rep full states and the first nE
        #      history entries (both counts arrive with the host lists below), never trash-bound padding ----
        p.evl = bool(self.evict_logs)
        if nev:
            om = torch.argsort(rep.to(I8), stable=True)
            p.evM_slot = p.ev_slot[om]; evf_slots = evf_slots[om]
            ol = torch.argsort((evl_ids < 0).reshape(-1).to(I8), stable=True)
            p.evl_src = (p.ev_dyn[:, None] * K + kk[None, :]).reshape(-1)[ol]   # rows of glog.view(-1, D)
            evl_ids = evl_ids.reshape(-1)[ol]
        else:
            p.evM_slot = p.ev_slot; p.evl_src = torch.empty(0, dtype=I64, device=d)
        # ---- host work lists (physical ids) in emission order; one device->host copy ----
        F_slot = self.loc[p.F_row].to(I64); S_slot = self.loc[p.S_row].to(I64)
        W_slot = torch.cat([c_slots, F_slot[nFa:], evf_slots])          # compactions, transient M updates, evictions (to-L last)
        tops = torch.cat([self.mat_top, self.log_top, self.live, rep.sum().reshape(1).to(I64)])
        names = ('S_slot', 'F_slot', 'L_ids', 'A_ids', 'W_slot', 'EVL_ids', 'tops')
        parts = (S_slot, F_slot, L_ids.reshape(-1).to(I64), a_ids.to(I64), W_slot, evl_ids.reshape(-1).to(I64), tops)
        flat = torch.cat(parts).cpu(); self.stats['syncs'] += 1; p.cpu = {}; o = 0
        for k_, t in zip(names, parts):
            v = flat[o:o + t.numel()]; o += t.numel(); p.cpu[k_] = v
        p.cpu['nev'] = nev; p.cpu['nE'] = int(np.count_nonzero(p.cpu['EVL_ids'].numpy() >= 0))
        p.cpu['n_rep'] = int(p.cpu['tops'][3])
        st = self.stats; st['adm'] += n_adm; st['ev'] += nev; st['hits'] += counts[C_HIT]
        st['N'] += Npos.numel(); st['L'] += Lpos.numel(); st['F'] += Fpos.numel(); st['S'] += Spos.numel()
        st['compactions'] += nNc + nLc; st['appends'] += nNp + nLp
        return p


# ----------------------------------------------------------------------------------------------------------------
class ELARAExecutor(TieredSparse):
    """Same pipeline as TieredSparse (two host workers, link model, ordering rules); host work is addressed by
    physical ids from the planner's directory; resident histories are recorded on the GPU."""

    def __init__(self, R, D, cfg, planner, device, threads=16, link=None, depth=2, host_budget=0, ahead_bytes=0):
        super().__init__(R, D, cfg, planner, device, threads=threads, link=link, depth=depth, store='dense_none',
                         host_budget=host_budget, ahead_bytes=ahead_bytes)
        Kmax = planner.Kmax; self.Kw = Kmax
        self.glog = torch.zeros(planner.Cdyn + 1, Kmax, D, device=device)       # resident gradient histories (+ trash row)
        if planner.evl_mode == 2:   # memory-pressure gate of the residency-aware representation (sticky once on)
            st = self.store; frac = planner.evl_gate_frac
            planner.gate = lambda: st.budget > 0 and (st.top['mat'] * 3 + st.live_logs) * D * 4 >= frac * st.budget

    def _make_store(self, R, D, store, host_budget, ahead_bytes, fused):
        return ArenaStore(R, D, self.E, self.threads, host_budget, log_cap=self.pl.log_cap, ahead_bytes=ahead_bytes)

    def load_static(self):
        sr = self.pl.static_rows
        if sr is None:
            return
        n = sr.numel(); self.pool[:n] = init_rows(sr, self.D, self.cfg['seed'], self.cfg['bound'])
        self.pl.code[sr] = self.pl.MAT   # pinned resident rows: authoritative state in the pool
        torch.cuda.synchronize(self.d)

    def enqueue_gather(self, p):
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
                # host-side index work in NumPy (torch CPU ops would use the intra-op thread pool on the hot path)
                S_slot, F_slot = c['S_slot'], c['F_slot']; li = c['L_ids'].numpy(); L_ids = torch.from_numpy(li[li >= 0])
                sw = torch.empty(len(S_slot), D, pin_memory=True); ff = torch.empty(len(F_slot), 3, D, pin_memory=True)
                lg = torch.empty(len(L_ids), D, pin_memory=True)
                if len(S_slot): self.E.cpu_gather(st.mat, S_slot, sw, 1, self.threads)
                if len(F_slot): self.E.cpu_gather(st.mat, F_slot, ff, 3, self.threads)
                if len(L_ids): self.E.cpu_gather(st.logs, L_ids, lg, 1, self.threads)
                self.timers['cpu_gather'] += time.perf_counter() - t0
                nb = (sw.numel() + ff.numel() + lg.numel()) * 4
                self.link.transfer('in', nb, t0)
                with torch.cuda.stream(self.copy):
                    g = tuple(t.to(self.d, non_blocking=True) for t in (sw, ff, lg))
                    e = torch.cuda.Event(); e.record(self.copy)
                self.counts['h2d_bytes'] += nb
                self.staged[p.s][1] = (g, e); ev.set()
        self.q_in.put(task)

    def assemble(self, p):
        self._check()
        ev = self.staged[p.s][0]; t0 = time.perf_counter(); ev.wait(); self.timers['gpu_wait_stage'] += time.perf_counter() - t0
        self._check()
        (sw, ff, lg), e = self.staged.pop(p.s)[1]
        cur = torch.cuda.current_stream(self.d); cur.wait_event(e)
        for t in (sw, ff, lg): t.record_stream(cur)
        D, cfg = self.D, self.cfg
        W = torch.empty(p.n, D, device=self.d)
        if p.hit_pos.numel(): W.index_copy_(0, p.hit_pos, self.pool[:, 0].index_select(0, p.hit_slot))
        fullN = init_rows(p.N_row, D, cfg['seed'], cfg['bound']) if p.N_row.numel() else None
        if fullN is not None: W.index_copy_(0, p.N_pos, fullN[:, 0])
        fullL = None
        if p.L_row.numel():
            fullL = replay_rows(p.L_row, lg, p.L_off, p.L_steps, p.L_cnt, D, self.Kw, cfg['seed'], cfg['bound'], self.table)
            W.index_copy_(0, p.L_pos, fullL[:, 0])
        if p.F_row.numel(): W.index_copy_(0, p.F_pos, ff[:, 0])
        if p.S_row.numel(): W.index_copy_(0, p.S_pos, sw)
        return W, (fullN, fullL, ff)

    def update(self, p, G, bufs):
        cfg, s, D, dev, ix = self.cfg, p.s, self.D, self.d, p.ix
        a = (cfg['lr'], cfg['b1'], cfg['b2'], cfg['eps'], s)
        fullN, fullL, ff = bufs
        sel = lambda t, i: t.index_select(0, i)
        pl = self.pl
        # resident histories: hit gradients (before the pool update; payload = the exact gradient applied); nothing
        # is recorded while the residency-aware representation is off
        if p.hit_pos.numel():
            gh = sel(G, p.hit_pos)
            if p.evl: self.glog.index_put_((p.hrec_dyn, p.hrec_j), gh)
            gpu_adam(self.pool, p.hit_slot, gh, *a)
        # evicted slots are read before slot reuse: full states of rows going to M, valid history entries of rows
        # going back to L (packed by the planner; counts from the host lists, no synchronization)
        nM = p.ev_slot.numel() - p.cpu['n_rep']; nE = p.cpu['nE']
        evf_buf = sel(self.pool, p.evM_slot[:nM]) if nM else None
        evl_buf = self.glog.view(-1, D).index_select(0, p.evl_src[:nE]) if nE else None   # [nE, D]
        outs_g = [sel(G, p.S_pos)]; outs_full = []
        adm_g = []
        for X, full, pos in (('N', fullN, p.N_pos), ('L', fullL, p.L_pos)):
            if full is None:
                continue
            up = ix[X + '_upd']
            if up.numel(): gpu_adam(full, up, sel(G, sel(pos, up)), *a)
            if ix[X + '_adm'].numel():
                self.pool.index_copy_(0, ix[X + '_adm_slot'], sel(full, ix[X + '_adm']))
                adm_g.append(sel(G, sel(pos, ix[X + '_adm'])))
            if ix[X + '_cmp'].numel(): outs_full.append(sel(full, ix[X + '_cmp']))
            if ix[X + '_app'].numel(): outs_g.append(sel(G, sel(pos, ix[X + '_app'])))
        nNa, nLa = ix['N_adm'].numel(), ix['L_adm'].numel()
        if nNa + nLa and p.evl:   # admission gradient = resident history entry 0
            ag = torch.cat(adm_g) if len(adm_g) > 1 else adm_g[0]
            dyn = p.adm_rec_dyn[:nNa + nLa]; z = torch.zeros_like(dyn)
            self.glog.index_put_((dyn, z), ag)
        if p.F_row.numel():
            gpu_adam(ff, torch.arange(p.F_row.numel(), device=dev), sel(G, p.F_pos), *a)
            if ix['F_adm'].numel(): self.pool.index_copy_(0, ix['F_adm_slot'], sel(ff, ix['F_adm']))
            if ix['F_nadm'].numel(): outs_full.append(sel(ff, ix['F_nadm']))
        if evf_buf is not None: outs_full.append(evf_buf)
        og_gpu = torch.cat(outs_g) if len(outs_g) > 1 else outs_g[0]
        of_gpu = torch.cat(outs_full) if outs_full else torch.empty(0, 3, D, device=dev)
        oe_gpu = evl_buf if evl_buf is not None else torch.empty(0, D, device=dev)
        og = torch.empty(og_gpu.shape, pin_memory=True); of = torch.empty(of_gpu.shape, pin_memory=True)
        oe = torch.empty(oe_gpu.shape, pin_memory=True)
        cur = torch.cuda.current_stream(dev); ev_c = torch.cuda.Event(); ev_c.record(cur)
        with torch.cuda.stream(self.d2h):
            self.d2h.wait_event(ev_c)
            for h, g_ in ((og, og_gpu), (of, of_gpu), (oe, oe_gpu)):
                h.copy_(g_, non_blocking=True); g_.record_stream(self.d2h)
            out_ev = torch.cuda.Event(); out_ev.record(self.d2h)
        self.counts['d2h_bytes'] += (og.numel() + of.numel() + oe.numel()) * 4
        self.enqueue_update_ids(s, out_ev, og, of, oe, p.cpu)

    def enqueue_update_ids(self, s, out_ev, og, of, oe, c):
        done = self._done_event(s)
        for old in [k for k in self.upd_done if k < s - 2 * self.depth - 2]: self.upd_done.pop(old, None)

        def task():
            with record_function('U_update'):
                try:
                    t0 = time.perf_counter()
                    while not out_ev.query():
                        time.sleep(2e-5)
                    t1 = time.perf_counter(); self.timers['cpu_wait_gpu'] += t1 - t0
                    st, D = self.store, self.D
                    S_slot, A_ids = c['S_slot'], c['A_ids']
                    nE, n_rep_rows = c['nE'], c['n_rep']
                    evl_ids = c['EVL_ids'][:nE]                                     # packed: valid entries only
                    W_slot = c['W_slot'][:len(c['W_slot']) - n_rep_rows]           # packed: rows going back to L dropped
                    nS = len(S_slot); nA = len(A_ids)
                    nb = (nS + nA + nE) * D * 4 + len(W_slot) * 3 * D * 4
                    self.link.transfer('out', nb, t1)
                    t2 = time.perf_counter()
                    mt, lt, live, _ = c['tops'].tolist()
                    st.set_tops(mt, lt, live)
                    st.fault('mat', mt); st.fault('log', lt)
                    if nS: self.E.cpu_adam(st.mat, S_slot, og[:nS].contiguous(), self.cfg['lr'], self.cfg['b1'], self.cfg['b2'], self.cfg['eps'], s, self.threads)
                    t3 = time.perf_counter(); self.timers['cpu_adam'] += t3 - t2
                    if nA: self.E.cpu_scatter(st.logs, A_ids, og[nS:].view(nA, 1, D), self.threads)
                    if nE: self.E.cpu_scatter(st.logs, evl_ids, oe.view(-1, 1, D), self.threads)
                    if len(W_slot): self.E.cpu_scatter(st.mat, W_slot, of, self.threads)
                    self.timers['cpu_update'] += time.perf_counter() - t2
                    self.counts['cpu_adam_rows'] += nS; self.counts['log_appends'] += nA + nE; self.counts['full_writes'] += len(W_slot)
                    self.counts['ev_to_logs_rows'] = self.counts.get('ev_to_logs_rows', 0) + n_rep_rows
                    self.counts['ev_to_logs_entries'] = self.counts.get('ev_to_logs_entries', 0) + nE
                    if s % 50 == 0:
                        rss, hwm = proc_mem(); self.footprint_trace.append({'step': s, 'rss': rss, **st.footprint()})
                finally:
                    done.set()
        self.q_out.put(task)

    def eval_weights(self, rows):
        pl, D, cfg = self.pl, self.D, self.cfg
        W = torch.empty(rows.numel(), D, device=self.d)
        sl = pl.slot_of[rows]; res = sl >= 0
        if res.any(): W[res] = self.pool[sl[res].long(), 0]
        nr = (~res).nonzero().flatten(); r = rows[nr]; c = pl.code[r]
        mat = c == pl.MAT
        if mat.any():
            slots = pl.loc[r[mat]].long().cpu(); buf = torch.empty(slots.numel(), D, pin_memory=True)
            self.E.cpu_gather(self.store.mat, slots, buf, 1, self.threads); W[nr[mat]] = buf.to(self.d)
        lz = ~mat
        if lz.any():
            rl = r[lz]; cnt = c[lz].to(I32); K = pl.Kmax; kk = torch.arange(K, device=self.d)
            e = pl.ent[rl[:, None] * K + kk[None, :]]
            ids = torch.where(kk[None, :] < cnt[:, None], e, torch.full_like(e, -1))
            steps = torch.where(e >= 0, pl.lstep[e.clamp(min=0).long()], torch.zeros_like(e))
            hid = ids.reshape(-1).cpu(); hid = hid[hid >= 0].long()
            lg = torch.empty(hid.numel(), D, pin_memory=True)
            if hid.numel(): self.E.cpu_gather(self.store.logs, hid, lg, 1, self.threads)
            off = torch.cumsum(cnt.long(), 0) - cnt.long()
            full = replay_rows(rl, lg.to(self.d), off, steps, cnt, D, K, cfg['seed'], cfg['bound'], self.table)
            W[nr[lz]] = full[:, 0]
        return W

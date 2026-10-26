"""Fused Triton kernels for the ELARA state directory (fast path of the planner).

The directory maintenance of one plan step is a handful of kernels instead of ~230 small torch operations
(kernel-launch / Python-dispatch overhead was the planner's cost; the GPU work itself is < 1 ms). Semantics are
identical to the reference torch path in elara.py (ELARAPlanner, dir_impl='ref'); ids are storage addresses only,
so allocation order never affects numerics. Allocation discipline per step: every log-id allocation (resident
histories of evicted rows, then appends) pops the free stack / bumps log_top before any release of the step pushes.
"""
import torch
import triton
import triton.language as tl

BLOCK = 1024


@triton.jit
def _hits(hs, n, n0, Cdyn, K: tl.constexpr, flag, adm_c0, rj, rok, pstep, step, out_dyn, out_j, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK); m = i < n
    slot = tl.load(hs + i, mask=m, other=0)
    dyn = slot >= n0
    c0 = tl.load(adm_c0 + slot, mask=m, other=-1).to(tl.int32)
    j = tl.load(rj + slot, mask=m, other=0).to(tl.int32)
    ok = tl.load(rok + slot, mask=m, other=0) != 0
    rec = m & dyn & ok & ((c0 + j) < K) & (flag != 0)
    di = slot - n0
    tl.store(out_dyn + i, tl.where(rec, di, Cdyn), mask=m)
    tl.store(out_j + i, tl.where(rec, j, 0).to(tl.int64), mask=m)
    tl.store(pstep + di * K + j, step, mask=rec)
    tl.store(rj + slot, (j + 1).to(tl.int8), mask=rec)
    ovf = m & dyn & ok & (~rec)
    tl.store(rok + slot, tl.zeros([BLOCK], tl.int8), mask=ovf)


def hits(pl, hs, s):
    n = hs.numel(); d = hs.device
    out_dyn = torch.empty(n, dtype=torch.int64, device=d); out_j = torch.empty(n, dtype=torch.int64, device=d)
    if n:
        _hits[(triton.cdiv(n, BLOCK),)](hs, n, pl.n0, pl.Cdyn, pl.Kmax, int(pl.evict_logs), pl.adm_c0, pl.rj, pl.rok, pl.pstep, s,
                                         out_dyn, out_j, BLOCK=BLOCK)
    return out_dyn, out_j


@triton.jit
def _lsteps(rows, cnt, n, K: tl.constexpr, ent, lstep, out_steps, out_ids, BLOCK: tl.constexpr):
    t = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK); m = t < n * K
    i = t // K; k = t % K
    r = tl.load(rows + i, mask=m, other=0)
    c = tl.load(cnt + i, mask=m, other=0)
    e = tl.load(ent + r * K + k, mask=m, other=-1)
    st = tl.load(lstep + tl.where(e >= 0, e, 0).to(tl.int64), mask=m & (e >= 0), other=0)
    tl.store(out_steps + t, tl.where(e >= 0, st, 0), mask=m)
    tl.store(out_ids + t, tl.where(k < c, e, -1).to(tl.int64), mask=m)


def lsteps(pl, rows, cnt):
    n = rows.numel(); K = pl.Kmax; d = rows.device
    steps = torch.empty(n, K, dtype=torch.int32, device=d); ids = torch.empty(n, K, dtype=torch.int64, device=d)
    if n:
        _lsteps[(triton.cdiv(n * K, BLOCK),)](rows, cnt, n, K, pl.ent, pl.lstep, steps, ids, BLOCK=BLOCK)
    return steps, ids


@triton.jit
def _evict_prep(ev_slot, ev_row, thr, n, n0, K: tl.constexpr, flag, MAT, adm_c0, rj, rok, loc, counts, c0_out, j_out, BLOCK: tl.constexpr):
    """counts [4, n]: entries to allocate (to-L rows: j), entries to release (to-M rows: c0), needs-MAT-slot, rep flag."""
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK); m = i < n
    sl = tl.load(ev_slot + i, mask=m, other=0); r = tl.load(ev_row + i, mask=m, other=0)
    c0 = tl.load(adm_c0 + sl, mask=m, other=-1).to(tl.int64); j = tl.load(rj + sl, mask=m, other=0).to(tl.int64)
    th = tl.load(thr + i, mask=m, other=0).to(tl.int64)
    ok = tl.load(rok + sl, mask=m, other=0) != 0
    rep = (flag != 0) & ok & ((c0 + j) <= th)
    lc = tl.load(loc + r, mask=m, other=0)
    tl.store(counts + i, tl.where(rep, j, 0), mask=m)
    tl.store(counts + n + i, tl.where(rep, 0, tl.maximum(c0, 0)), mask=m)
    tl.store(counts + 2 * n + i, tl.where((~rep) & (lc < 0), 1, 0).to(tl.int64), mask=m)
    tl.store(counts + 3 * n + i, rep.to(tl.int64), mask=m)
    tl.store(c0_out + i, c0, mask=m); tl.store(j_out + i, j, mask=m)


@triton.jit
def _alloc_entries(rows, kstart, nent, off, base_ptr, n, K: tl.constexpr, lfree, nfree_ptr, ltop_ptr, ent, lstep,
                   steps_src, dyn, step_const, use_src: tl.constexpr, out_ids, BLOCK: tl.constexpr):
    """Entry k < nent[i] of row i gets log id from rank = base + off[i] + k (free stack first, then bump); directory:
    ent[row, kstart[i] + k] = id, lstep[id] = (steps_src[dyn[i], k] if use_src else step_const). out_ids [n, K]."""
    t = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK); m = t < n * K
    i = t // K; k = t % K
    ne = tl.load(nent + i, mask=m, other=0)
    v = m & (k < ne)
    base = tl.load(base_ptr); nf = tl.load(nfree_ptr); top = tl.load(ltop_ptr)
    rank = base + tl.load(off + i, mask=m, other=0) + k
    fid = tl.load(lfree + tl.maximum(nf - 1 - rank, 0), mask=v & (rank < nf), other=0).to(tl.int64)
    lid = tl.where(rank < nf, fid, top + rank - nf)
    r = tl.load(rows + i, mask=m, other=0); ks = tl.load(kstart + i, mask=m, other=0)
    tl.store(ent + r * K + ks + k, lid.to(tl.int32), mask=v)
    if use_src:
        dv = tl.load(dyn + i, mask=m, other=0)
        stv = tl.load(steps_src + dv * K + k, mask=v, other=0)
    else:
        stv = tl.zeros([BLOCK], tl.int32) + step_const
    tl.store(lstep + lid, stv, mask=v)
    tl.store(out_ids + t, tl.where(v, lid, -1), mask=m)


@triton.jit
def _release(rows, nrel, off, n, K: tl.constexpr, lfree, nfree_ptr, ent, BLOCK: tl.constexpr):
    """Push the first nrel[i] entries of row i onto the free stack at nfree + off[i] + k and clear them."""
    t = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK); m = t < n * K
    i = t // K; k = t % K
    v = m & (k < tl.load(nrel + i, mask=m, other=0))
    r = tl.load(rows + i, mask=m, other=0)
    e = tl.load(ent + r * K + k, mask=v, other=-1)
    v = v & (e >= 0)
    nf = tl.load(nfree_ptr)
    tl.store(lfree + nf + tl.load(off + i, mask=m, other=0) + k, e, mask=v)
    tl.store(ent + r * K + k, tl.full([BLOCK], -1, tl.int32), mask=v)


@triton.jit
def _mat_assign(rows, need, off, rep, n, trash, loc, mtop_ptr, out_slot, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK); m = i < n
    r = tl.load(rows + i, mask=m, other=0)
    nd = tl.load(need + i, mask=m, other=0) != 0
    rp = tl.load(rep + i, mask=m, other=0) != 0
    lc = tl.load(loc + r, mask=m, other=-1).to(tl.int64)
    sl = tl.where(nd, tl.load(mtop_ptr) + tl.load(off + i, mask=m, other=0), lc)
    tl.store(loc + r, sl.to(tl.int32), mask=m & nd)
    tl.store(out_slot + i, tl.where(rp, trash, sl), mask=m)


@triton.jit
def _code_set(rows, rep, c0, j, n, MAT, code, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK); m = i < n
    r = tl.load(rows + i, mask=m, other=0)
    rp = tl.load(rep + i, mask=m, other=0) != 0
    tot = tl.load(c0 + i, mask=m, other=0) + tl.load(j + i, mask=m, other=0)
    tl.store(code + r, tl.where(rp, tot, MAT).to(tl.int8), mask=m)


def _excl(x):
    """exclusive prefix sums along the last dim and totals (one cumsum kernel)."""
    c = torch.cumsum(x, -1)
    return c - x, c[..., -1:] if x.shape[-1] else torch.zeros(x.shape[:-1] + (1,), dtype=x.dtype, device=x.device)


class Directory:
    """Fused directory operations for one plan step (bound to an ELARAPlanner)."""

    def __init__(self, pl):
        self.pl = pl; d = pl.d
        self.zero = torch.zeros(1, dtype=torch.int64, device=d)

    def evict(self, ev_slot, ev_row, thr):
        """Eviction representation (to L if the history fits, else to M). Returns (rep int64[n], evl_ids [n,K],
        evf_slots [n] (trash for to-L rows), release plan (rows, counts) to apply after the step's allocations)."""
        pl = self.pl; n = ev_slot.numel(); K = pl.Kmax; d = pl.d
        if n == 0:
            e = torch.empty(0, dtype=torch.int64, device=d)
            return e, torch.empty(0, K, dtype=torch.int64, device=d), e, (e, e)
        counts = torch.empty(4, n, dtype=torch.int64, device=d); c0 = torch.empty(n, dtype=torch.int64, device=d); j = torch.empty_like(c0)
        g = (triton.cdiv(n, BLOCK),)
        _evict_prep[g](ev_slot, ev_row, thr, n, pl.n0, K, int(pl.evict_logs), pl.MAT, pl.adm_c0, pl.rj, pl.rok, pl.loc, counts, c0, j, BLOCK=BLOCK)
        offs, tot = _excl(counts[:3])
        evl_ids = torch.empty(n, K, dtype=torch.int64, device=d)
        _alloc_entries[(triton.cdiv(n * K, BLOCK),)](ev_row, c0.clamp(min=0), counts[0], offs[0], self.zero, n, K, pl.lfree, pl.n_lfree,
                                                     pl.log_top, pl.ent, pl.lstep, pl.pstep, ev_slot - pl.n0, 0, True, evl_ids, BLOCK=BLOCK)
        self.alloc_total = tot[0]                                          # appends rank after these
        evf = torch.empty(n, dtype=torch.int64, device=d)
        _mat_assign[g](ev_row, counts[2], offs[2], counts[3], n, pl.mat_trash, pl.loc, pl.mat_top, evf, BLOCK=BLOCK)
        pl.mat_top += tot[2]
        _code_set[g](ev_row, counts[3], c0, j, n, pl.MAT, pl.code, BLOCK=BLOCK)
        return counts[3], evl_ids, evf, (ev_row, counts[1])

    def appends(self, rows, ks, s):
        pl = self.pl; n = rows.numel(); K = pl.Kmax; d = pl.d
        base = getattr(self, 'alloc_total', self.zero)
        ids = torch.empty(n, K, dtype=torch.int64, device=d)
        if n:
            one = torch.ones(n, dtype=torch.int64, device=d); off = torch.arange(n, dtype=torch.int64, device=d)
            _alloc_entries[(triton.cdiv(n * K, BLOCK),)](rows, ks, one, off, base, n, K, pl.lfree, pl.n_lfree, pl.log_top, pl.ent,
                                                         pl.lstep, pl.pstep, off, s, False, ids, BLOCK=BLOCK)
            pl.code[rows] = (ks + 1).to(torch.int8)
        total = base + n
        used = torch.minimum(total, pl.n_lfree)
        pl.log_top += total - used; pl.n_lfree -= used; pl.live += total
        self.alloc_total = self.zero
        return ids[:, 0]

    def compact(self, rows, cnt):
        """MAT slots for compacted rows; returns slots; their logs are released by release()."""
        pl = self.pl; n = rows.numel(); d = pl.d
        out = torch.empty(n, dtype=torch.int64, device=d)
        if n:
            need = (pl.loc[rows] < 0).to(torch.int64); off, tot = _excl(need)
            _mat_assign[(triton.cdiv(n, BLOCK),)](rows, need, off, torch.zeros_like(need), n, pl.mat_trash, pl.loc, pl.mat_top, out, BLOCK=BLOCK)
            pl.mat_top += tot
            pl.code[rows] = pl.MAT
        return out

    def release(self, parts):
        """parts: list of (rows, counts) -> push all their entries (after this step's allocations)."""
        pl = self.pl; K = pl.Kmax
        parts = [(r, c) for r, c in parts if r.numel()]
        if not parts:
            return
        rows = torch.cat([r for r, _ in parts]); cnt = torch.cat([c.to(torch.int64) for _, c in parts]); n = rows.numel()
        off, tot = _excl(cnt)
        _release[(triton.cdiv(n * K, BLOCK),)](rows, cnt, off, n, K, pl.lfree, pl.n_lfree, pl.ent, BLOCK=BLOCK)
        # entries pushed = only those that existed (e >= 0); every counted entry exists by construction
        pl.n_lfree += tot; pl.live -= tot

"""Triton kernels: deterministic row initialization and exact Adam replay of short gradient logs.

init: w0[row, j] = (u(row, j) * 2 - 1) * bound, u from a 32-bit integer hash of (seed, row, j);
      m = v = 0. Every method obtains initial weights from this kernel (host tables for eager
      methods are filled by it at setup), so lazy and eager paths start from identical bits.
replay: state = init(row); for k < cnt: Adam(state, g_k, step_k)  -- same arithmetic as gpu_adam /
      cpu_adam (separately rounded ops, div_rn/sqrt_rn, FP fusion disabled), hence bitwise exact.
"""
import torch
import triton
import triton.language as tl
from .ops import ext


@triton.jit
def _hash32(x):
    # murmur3 fmix32 on uint32
    x = x ^ (x >> 16)
    x = x * 0x85EBCA6B
    x = x ^ (x >> 13)
    x = x * 0xC2B2AE35
    x = x ^ (x >> 16)
    return x


@triton.jit
def _init_rows(out, rows, seed, bound, D: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0)
    r = tl.load(rows + i).to(tl.int64)
    offs = tl.arange(0, BLOCK); mask = offs < D
    key = (r * D + offs).to(tl.uint32) ^ (_hash32((r >> 32).to(tl.uint32) + seed.to(tl.uint32)))
    h = _hash32(key ^ seed.to(tl.uint32))
    u = (h >> 8).to(tl.float32) * (1.0 / 16777216.0)
    w = (u * 2.0 - 1.0) * bound
    base = out + i.to(tl.int64) * (3 * D)
    tl.store(base + offs, w, mask=mask)
    tl.store(base + D + offs, tl.zeros([BLOCK], tl.float32), mask=mask)
    tl.store(base + 2 * D + offs, tl.zeros([BLOCK], tl.float32), mask=mask)


def init_rows(rows, D, seed, bound, out=None):
    """rows: cuda int64 [n] -> out [n,3,D] (w0, 0, 0)."""
    n = rows.numel()
    if out is None:
        out = torch.empty(n, 3, D, device=rows.device)
    if n:
        _init_rows[(n,)](out, rows, seed, bound, D, BLOCK=triton.next_power_of_2(D), enable_fp_fusion=False)
    return out


@triton.jit
def _replay(out, rows, logs, loff, steps, cnt, seed, bound, sc, D: tl.constexpr, K: tl.constexpr, BLOCK: tl.constexpr):
    """sc: float32 [maxstep+1, 8] table of Adam scalars per logical step (lr,b1,omb1,b2,omb2,bc1,bc2,eps)."""
    i = tl.program_id(0)
    i64 = i.to(tl.int64)
    r = tl.load(rows + i).to(tl.int64)
    offs = tl.arange(0, BLOCK); mask = offs < D
    key = (r * D + offs).to(tl.uint32) ^ (_hash32((r >> 32).to(tl.uint32) + seed.to(tl.uint32)))
    h = _hash32(key ^ seed.to(tl.uint32))
    u = (h >> 8).to(tl.float32) * (1.0 / 16777216.0)
    w = (u * 2.0 - 1.0) * bound
    m = tl.zeros([BLOCK], tl.float32); v = tl.zeros([BLOCK], tl.float32)
    c = tl.load(cnt + i)
    o = tl.load(loff + i).to(tl.int64)
    for k in tl.static_range(K):
        if k < c:
            st = tl.load(steps + i64 * K + k).to(tl.int64)
            lr = tl.load(sc + st * 8 + 0); b1 = tl.load(sc + st * 8 + 1); omb1 = tl.load(sc + st * 8 + 2)
            b2 = tl.load(sc + st * 8 + 3); omb2 = tl.load(sc + st * 8 + 4); bc1 = tl.load(sc + st * 8 + 5)
            bc2 = tl.load(sc + st * 8 + 6); eps = tl.load(sc + st * 8 + 7)
            g = tl.load(logs + (o + k) * D + offs, mask=mask)
            m = b1 * m + omb1 * g
            t2 = omb2 * g
            v = b2 * v + t2 * g
            mh = tl.math.div_rn(m, bc1); vh = tl.math.div_rn(v, bc2)
            den = tl.math.sqrt_rn(vh) + eps
            w = w - tl.math.div_rn(lr * mh, den)
    base = out + i64 * (3 * D)
    tl.store(base + offs, w, mask=mask); tl.store(base + D + offs, m, mask=mask); tl.store(base + 2 * D + offs, v, mask=mask)


class ScalarTable:
    """Per-logical-step Adam scalars computed by the C++ helper (identical to cpu_adam / gpu_adam)."""

    def __init__(self, lr, b1, b2, eps, max_step, device):
        e = ext()
        t = torch.tensor([e.adam_scalars(lr, b1, b2, eps, s) if s > 0 else [0.0] * 8 for s in range(max_step + 1)], dtype=torch.float32)
        self.t = t.to(device)


def replay_rows(rows, logs, offs, steps, cnt, D, K, seed, bound, table, out=None):
    """rows int64 [n]; logs flat [sum(cnt),D] with row i at offs[i]; steps int32 [n,K]; cnt int32 [n] -> [n,3,D]."""
    n = rows.numel()
    if out is None:
        out = torch.empty(n, 3, D, device=rows.device)
    if n:
        _replay[(n,)](out, rows, logs, offs, steps, cnt, seed, bound, table.t, D, K, BLOCK=triton.next_power_of_2(D),
                      enable_fp_fusion=False)
    return out

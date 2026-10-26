"""srm2 primitives.

CPU (C++ extension, JIT-built): fused row-sparse Adam and row gather/scatter on interleaved
[R,3,D] (w,m,v) host state; multi-threaded, GIL released.
GPU (Triton): row-sparse Adam on a [C,3,D] pool, written with correctly rounded division/sqrt
and FP fusion disabled so that it is bitwise identical to the CPU kernel.
"""
import os
from pathlib import Path
import torch
import triton
import triton.language as tl
from torch.utils.cpp_extension import load

_HERE = Path(__file__).resolve().parent
_ext = None


def ext():
    global _ext
    if _ext is None:
        bd = os.environ.get('SRM2_BUILD', str(_HERE / 'build')); os.makedirs(bd, exist_ok=True)
        _ext = load(name='srm2_cpu_ext', sources=[str(_HERE / 'csrc/srm2_cpu.cpp')],
                    extra_cflags=['-O3', '-ffp-contract=off', '-march=native', '-fno-fast-math'],
                    build_directory=bd, verbose=False)
    return _ext


_dir = None


def dir_ext():
    """ELARA state-directory operations (C++ ATen port of the reference path; GIL released)."""
    global _dir
    if _dir is None:
        bd = os.environ.get('SRM2_BUILD', str(_HERE / 'build')) + '_dir'; os.makedirs(bd, exist_ok=True)
        _dir = load(name='srm2_elara_dir', sources=[str(_HERE / 'csrc/elara_dir.cpp')], extra_cflags=['-O2'],
                    build_directory=bd, verbose=False)
    return _dir


@triton.jit
def _adam_rows(pool, slots, grad, D: tl.constexpr, lr, b1, omb1, b2, omb2, bc1, bc2, eps, BLOCK: tl.constexpr):
    i = tl.program_id(0)
    slot = tl.load(slots + i).to(tl.int64)
    offs = tl.arange(0, BLOCK); mask = offs < D
    base = pool + slot * (3 * D)
    w = tl.load(base + offs, mask=mask); m = tl.load(base + D + offs, mask=mask); v = tl.load(base + 2 * D + offs, mask=mask)
    g = tl.load(grad + i.to(tl.int64) * D + offs, mask=mask)
    m = b1 * m + omb1 * g
    t2 = omb2 * g
    v = b2 * v + t2 * g
    mh = tl.math.div_rn(m, bc1); vh = tl.math.div_rn(v, bc2)
    den = tl.math.sqrt_rn(vh) + eps
    up = tl.math.div_rn(lr * mh, den)
    tl.store(base + offs, w - up, mask=mask); tl.store(base + D + offs, m, mask=mask); tl.store(base + 2 * D + offs, v, mask=mask)


def gpu_adam(pool, slots, grad, lr, b1, b2, eps, step):
    """pool [C,3,D] cuda float32; slots int32/int64 [n] (rows of pool); grad [n,D] (row i -> slots[i])."""
    n = slots.numel()
    if n == 0:
        return
    s = ext().adam_scalars(lr, b1, b2, eps, int(step))
    D = pool.shape[2]
    _adam_rows[(n,)](pool, slots, grad, D, *s, BLOCK=triton.next_power_of_2(D), num_warps=4 if D >= 128 else 2,
                     enable_fp_fusion=False)

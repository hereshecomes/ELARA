"""Summarize a torch.profiler window: GPU kernel time per step by category, and the named phase ranges
(M_* main thread, G_* inbound worker, U_* outbound worker, P_* planner) as CPU wall time per step."""
import re

CATS = [('dense_gemm', r'gemm|cutlass|sm\d\d_|ampere|cublas|xmma|wgmma'), ('adam', r'adam'), ('replay_init', r'replay|init_rows|_init|murmur'),
        ('sort_unique', r'sort|unique|radix|cub::|scan'), ('memcpy', r'memcpy|Memcpy'), ('index_gather_scatter', r'index|gather|scatter|embedding'),
        ('elementwise_other', r'.')]


def summarize(prof, nsteps):
    ka = prof.key_averages(); kern = {}; ranges = {}; cats = {c: 0.0 for c, _ in CATS}
    for e in ka:
        dt = getattr(e, 'self_device_time_total', None)
        if dt is None: dt = getattr(e, 'self_cuda_time_total', 0)
        name = e.key
        if name[:2] in ('M_', 'G_', 'U_', 'P_'):
            ranges[name] = {'cpu_ms_per_step': e.cpu_time_total / 1e3 / nsteps, 'count': e.count}
            continue
        if dt and dt > 0 and getattr(e, 'device_type', None) is not None and 'CUDA' in str(e.device_type):
            kern[name] = dt / 1e3 / nsteps
            for c, pat in CATS:
                if re.search(pat, name, re.I): cats[c] += dt / 1e3 / nsteps; break
    top = sorted(kern.items(), key=lambda kv: -kv[1])[:25]
    return {'steps': nsteps, 'gpu_ms_per_step_total': sum(kern.values()), 'gpu_ms_per_step_by_category': cats,
            'top_kernels_ms_per_step': top, 'ranges': ranges}

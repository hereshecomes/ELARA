"""Recommendation-model training on a chronological CTR trace with tiered sparse state (srm2).

All methods share: data order, model, dense Adam, sparse row-Adam arithmetic (bitwise identical across
paths), executor and pipeline. Methods differ only in the plan (residency / service path per row).
"""
import argparse, json, os, queue, threading, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from srm2.planner import GPUPlanner
from srm2.executor import TieredSparse, LinkModel
from srm2.store import HostBudgetExceeded, proc_mem
from srm2.models import MODELS, MULTI_TASK
import subprocess

METHODS = {   # name: (base, dyn, lazy, st_mode)
    'fixed_S': ('static', 'carry', False, 'S'),     # FixedHot / FAE-style static hot set, cold rows CPU-updated
    'fixed_T': ('static', 'carry', False, 'T'),     # static hot set, cold rows full-state GPU-updated (Hotline-style)
    'lru_T': ('none', 'lru', False, 'T'),           # full-state LRU cache, every miss admitted (DynamicEmb-style)
    'lfu_T': ('none', 'lfu', False, 'T'),           # full-state LFU cache
    'ttl_T': ('none', 'ttl', False, 'T'),           # BagPipe lookahead TTL cache, others transient
    'fixed_lazy_S': ('static', 'carry', True, 'S'),  # static hot set + exact lazy-log tail (host-side log index)
    'fixed_lazy_mix': ('static', 'carry', True, 'mix'),
    'fixed_lazy_T': ('static', 'carry', True, 'T'),
    'ttl_lazy_S': ('none', 'ttl', True, 'S'),
    'ttl_lazy_T': ('none', 'ttl', True, 'T'),        # lookahead residency only + lazy tail
    'fixed_ttl_T': ('static', 'ttl', False, 'T'),     # strong eager hybrid: static hot set + BagPipe TTL region
    'fixed_ttl_S': ('static', 'ttl', False, 'S'),     # same, cold rows CPU-updated (FAE + BagPipe)
    'fixed_ttl_lazy_T': ('static', 'ttl', True, 'T'), # static + TTL residency + exact lazy tail (host-side log index)
    'fixed_ttl_lazy_S': ('static', 'ttl', True, 'S'),
    'window_T': ('none', 'window', False, 'T'),      # ScratchPipe-style scratchpad (past window + lookahead), adapted
    'sparserm': ('static', 'ttl', True, 'T'),        # SparseRM-K3 (predecessor, host-side log index) = fixed_ttl_lazy_T, cost-model K
    'sparserm_S': ('static', 'ttl', True, 'S'),      # same on a tier with a CPU = fixed_ttl_lazy_S, cost-model K
    # ELARA engine (srm2/elara.py): GPU-resident state directory, residency-aware representation, adaptive thresholds
    'elara': ('static', 'ttl', True, 'T'),            # passive tier (transient full-state updates on GPU)
    'elara_S': ('static', 'ttl', True, 'S'),          # tier with a CPU (materialized cold rows updated near memory)
    'elara_k0': ('static', 'ttl', True, 'T'),         # optimized on-demand eager (K=0) on the same engine (baseline)
    'elara_k0_S': ('static', 'ttl', True, 'S'),
    'elara_carry': ('static', 'carry', True, 'T'),    # ablation: ELARA without lookahead (TTL) admission, carry only
}


def k_auto(st_mode):
    """Cost-model materialization threshold. Per touch of a logged row with k entries the tier moves
    k*G in + G out (G = one vector); a materialized touch moves 3G in + 3G out on a passive tier (T) or
    G in + G out plus near-memory Adam on a tier with CPU (S). With a full-duplex link the cost is
    max(in, out): logging beats a T touch while k < 3 (K* = 3) and an S touch while k <= 1 (we use K* = 2
    because the logged path also removes the near-memory Adam work)."""
    return 3 if st_mode == 'T' else 2


class GPUMonitor:
    """Samples utilization / memory of the GPU this process runs on (physical id = first CUDA_VISIBLE_DEVICES entry)
    and estimates foreign use (contention metadata; a foreign CUDA context alone is ~500 MiB, so > 300 MiB = contended). Container PID namespaces hide or remap compute-app PIDs, so a
    foreign process is inferred from the process count (> 1) or from memory not explained by this process
    (used - own reserved - ~0.6 GiB context)."""
    def __init__(self, period=5.0):
        self.samples = []; self.stop = threading.Event(); self.me = os.getpid(); self.period = period
        self.gpu = os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(',')[0] or '0'
        self.base = float(os.environ.get('SRM2_FOREIGN_BASE_MIB', 0) or 0)   # idle foreign memory present at launch (accepted)
        self.max_load = float(os.environ.get('SRM2_MAX_LOAD', 0) or 0)   # > 0: a run whose mean host load exceeds it is contended
        self.t = threading.Thread(target=self.run, daemon=True); self.t.start()

    def snap(self):
        try:
            q = subprocess.run(['nvidia-smi', '-i', self.gpu, '--query-gpu=utilization.gpu,memory.used', '--format=csv,noheader,nounits'],
                               capture_output=True, text=True, timeout=10).stdout.strip().split(',')
            apps = [l for l in subprocess.run(['nvidia-smi', '-i', self.gpu, '--query-compute-apps=pid,used_memory', '--format=csv,noheader,nounits'],
                                              capture_output=True, text=True, timeout=10).stdout.strip().splitlines() if l.strip()]
            own = torch.cuda.memory_reserved(0) / 2**20 + 600 if torch.cuda.is_initialized() else 0
            self.samples.append({'t': time.time(), 'util': float(q[0]), 'mem_mib': float(q[1]), 'foreign_procs': max(0, len(apps) - 1),
                                 'foreign_mem_mib': max(0.0, float(q[1]) - own), 'loadavg1': os.getloadavg()[0]})
        except Exception:
            pass

    def run(self):
        while not self.stop.wait(self.period):
            self.snap()

    def summary(self):
        self.stop.set()
        if not self.samples: self.snap()
        S = self.samples; m = lambda k: float(np.mean([x[k] for x in S])) if S else None
        return {'n': len(S), 'gpu': self.gpu, 'foreign_base_mib': self.base, 'gpu_util_mean': m('util'), 'foreign_procs_mean': m('foreign_procs'),
                'foreign_mem_mib_mean': m('foreign_mem_mib'), 'foreign_mem_mib_max': max((x['foreign_mem_mib'] for x in S), default=None),
                'loadavg1_mean': m('loadavg1'),
                'contended': bool(S and max(max(x['foreign_procs'] for x in S[1:] or S),
                                            int(max(x['foreign_mem_mib'] for x in S[1:] or S) > self.base + 300),
                                            int(self.max_load > 0 and float(np.mean([x['loadavg1'] for x in S])) > self.max_load)) > 0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data/criteo_kaggle')
    ap.add_argument('--method', required=True, choices=sorted(METHODS) + ['all_gpu'])
    ap.add_argument('--batch', type=int, default=16384); ap.add_argument('--dim', type=int, default=128)
    ap.add_argument('--start', type=int, default=0, help='first training sample'); ap.add_argument('--steps', type=int, default=200)
    ap.add_argument('--budget-rows', type=int, default=337626); ap.add_argument('--carry-reserve', type=int, default=16384)
    ap.add_argument('--profile', default='day0', help='static profile: dayK or a:b sample range')
    ap.add_argument('--K', default='auto', help="int, or 'auto' (tier cost model)"); ap.add_argument('--t-frac', type=float, default=0.3)
    ap.add_argument('--depth', type=int, default=3, help='pipeline depth d: gathers for step s wait for updates of s-d');
    ap.add_argument('--L', type=int, default=8); ap.add_argument('--threads', type=int, default=16)
    ap.add_argument('--link-gbps', type=float, default=0); ap.add_argument('--link-lat-us', type=float, default=0)
    ap.add_argument('--eval-start', type=int, default=-1); ap.add_argument('--eval-samples', type=int, default=0)
    ap.add_argument('--gpu-cap-gb', type=float, default=0); ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--seed', type=int, default=42); ap.add_argument('--out', required=True)
    ap.add_argument('--save-pred', type=int, default=1, help='1: save evaluation predictions to <out>.pred.npy '
                    '(tools/compare_preds.py checks bitwise identity across exact methods)')
    ap.add_argument('--cross', default='', help="'' or 'N_B': N hashed pairwise cross features into 2^B buckets each")
    ap.add_argument('--model', default='dlrm', choices=sorted(MODELS)); ap.add_argument('--amp', default='none', choices=['none', 'bf16'])
    ap.add_argument('--side-prio', type=int, default=-1, help='CUDA priority of the loader / planner streams (-1 = high, the '
                    'default so far; 0 = same as the training stream: planner work runs ahead and need not preempt training)')
    ap.add_argument('--trace-states', type=int, default=0, help='ELARA: every N planned steps record the row-state histogram '
                    '(I, L_1..L_K, M) and residency at plan time (dedicated runs; adds a planner sync)')
    ap.add_argument('--tasks', default='is_click,long_view,is_like', help='multi-task models: label columns of labels_mt.npy (equal-weight BCE)')
    ap.add_argument('--store', default='auto', choices=['auto', 'dense', 'compact'])
    ap.add_argument('--host-budget-gb', type=float, default=0, help='enforced capacity-tier store budget (0 = none)')
    ap.add_argument('--admit-threshold', type=int, default=1, help='>1: inexact CounterFilter-style admission baseline')
    ap.add_argument('--past-window', type=int, default=3)
    ap.add_argument('--diag', type=int, default=0, help='1: planner diagnostics (update-count histograms at eviction/compaction, per-table hazards)')
    ap.add_argument('--prof-steps', default='', help="'a:b': torch.profiler window over steps [a,b) -> <out>.trace.json + kernel summary")
    ap.add_argument('--emb-lr', type=float, default=0, help='sparse (embedding) Adam learning rate (0 = --lr)')
    ap.add_argument('--elara-kmax', type=int, default=3, help='ELARA: maximum gradient-log entries per row')
    ap.add_argument('--elara-evict-logs', type=int, default=2, help='ELARA: residency-aware representation (evicted rows may '
                    'return to L): 0 off, 1 always, 2 gated by memory pressure (on once the store reaches --elara-evl-gate of the host budget)')
    ap.add_argument('--elara-evl-gate', type=float, default=0.75, help='ELARA: store/budget fraction that switches the gated representation on')
    ap.add_argument('--elara-adaptive', type=int, default=0, help='ELARA: critical-path-aware per-table thresholds (0 = fixed K)')
    ap.add_argument('--elara-fused-dir', type=int, default=1, help='(legacy) 1 = triton, 0 = ref when --elara-dir-impl is unset')
    ap.add_argument('--elara-dir-impl', default='cpp', choices=['cpp', 'triton', 'ref'], help='ELARA directory implementation')
    ap.add_argument('--elara-ctl-every', type=int, default=50, help='ELARA controller interval (steps)')
    ap.add_argument('--fast-store', type=int, default=0, help='1: fused C++ compact-store operations (ELARA fast path)')
    ap.add_argument('--prefault-ahead-mb', type=int, default=0, help='compact store: background allocate-ahead (0 = off)')
    ap.add_argument('--hash-cap', type=int, default=0, help='>0: inexact hashing-trick baseline, every table folded to <= N rows')
    ap.add_argument('--multihot', default='', help="'' (one-hot) or 'mlperf' (DLRMv2 pooling sizes, deterministic expansion)")
    a = ap.parse_args()
    t_setup = time.perf_counter()
    dev = torch.device('cuda:0'); torch.cuda.set_device(dev)
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    if a.gpu_cap_gb:
        torch.cuda.set_per_process_memory_fraction(a.gpu_cap_gb * 2**30 / torch.cuda.get_device_properties(dev).total_memory, dev)
    meta = json.load(open(f'{a.data}/meta.json')); cards = meta['cardinalities']; cards0 = list(cards)
    if a.hash_cap: cards = [min(c, a.hash_cap) for c in cards0]   # folded vocabulary (collisions), same init bound
    R = int(sum(cards)); NF = len(cards)
    fold_t = torch.tensor([a.hash_cap if c > a.hash_cap else 0 for c in cards0] if a.hash_cap else [0] * len(cards0), dtype=torch.int64, device=dev)

    def fold(loc, t):
        """hashing trick: local id -> mix(id) mod cap for tables larger than the cap (identity otherwise)."""
        if not a.hash_cap:
            return loc
        h = (loc * 0x9E3779B1) & 0x7FFFFFFF; h = (h ^ (h >> 15)) * 0x2C1B3C6D & 0x7FFFFFFF
        cap = fold_t[t]
        return torch.where(cap > 0, h % cap.clamp(min=1), loc)
    off_np = np.cumsum([0] + cards[:-1]).astype(np.int64); off = torch.from_numpy(off_np).to(dev)
    BAGS = meta.get('bags')   # data-defined multi-hot bags over (possibly shared) tables, -1 = padding entry
    if BAGS:
        col_table = torch.tensor(meta['col_table'], device=dev)
        flat_cols = torch.tensor([c for bag in BAGS for c in bag], device=dev)
        bag_of_flat = torch.tensor([k for k, bag in enumerate(BAGS) for _ in bag], device=dev)
        NF = len(BAGS)
    sp = np.load(f'{a.data}/sparse.npy', mmap_mode='r'); dn = np.load(f'{a.data}/dense.npy', mmap_mode='r'); lb = np.load(f'{a.data}/labels.npy', mmap_mode='r')
    b, D, steps = a.batch, a.dim, a.steps
    NCOL = sp.shape[1]   # raw sparse columns
    ND = dn.shape[1]   # dense features (13 for Criteo, 2 for Avazu)
    MLPERF = [3, 2, 1, 2, 6, 1, 1, 1, 1, 7, 3, 8, 1, 6, 9, 5, 1, 1, 1, 12, 100, 27, 10, 3, 1, 1]
    P = MLPERF if a.multihot == 'mlperf' else [1] * NF; NF0, NL0 = NF, sum(P)
    if BAGS: NF0 = NCOL; NL0 = NCOL; P = [1] * NCOL   # (P/NL/f_of only used by the fixed-layout paths)
    NCROSS, HCROSS = (int(a.cross.split('_')[0]), 1 << int(a.cross.split('_')[1])) if a.cross else (0, 0)
    cpairs = [(2 * i, 2 * i + 1) for i in range(13)][:NCROSS] if NCROSS <= 13 else [(i, j) for i in range(NF0) for j in range(i + 1, NF0)][:NCROSS]
    cpa = torch.tensor([q[0] for q in cpairs], device=dev, dtype=torch.long); cpb = torch.tensor([q[1] for q in cpairs], device=dev, dtype=torch.long)
    coff = R + torch.arange(NCROSS, device=dev) * HCROSS
    R = R + NCROSS * HCROSS; NF = NF0 + NCROSS; NL = NL0 + NCROSS; P = P + [1] * NCROSS
    if BAGS: NF = len(BAGS); NL = NCOL
    f_of = torch.tensor([f for f in range(NF0) for _ in range(P[f])], device=dev)   # base fields only
    k_of = torch.tensor([k for f in range(NF0) for k in range(P[f])], device=dev)
    card_t = torch.tensor(cards, dtype=torch.int64, device=dev)
    bag_off1 = torch.tensor(np.cumsum([0] + P[:-1]), dtype=torch.int64, device=dev)

    def expand_base(raw):
        """raw int32 [n,NF0] local IDs -> global IDs [n,NL0]; slot k>0 of field f is a fixed hash of (f,k,id),
        i.e. every ID always expands to the same P_f IDs (MLPerf DLRMv2 synthetic multi-hot, uniform)."""
        loc = fold(raw.long()[:, f_of], f_of)
        if NL0 > NF0:
            h = (loc * 0x9E3779B1 + f_of * 0x85EBCA77 + k_of * 0xC2B2AE3D) & 0x7FFFFFFF
            h = (h ^ (h >> 15)) * 0x2C1B3C6D & 0x7FFFFFFF
            loc = torch.where(k_of == 0, loc, h % card_t[f_of])
        return loc + off[f_of]

    def expand(raw):
        g = expand_base(raw)
        if not NCROSS:
            return g
        a_ = raw.long()[:, cpa]; b_ = raw.long()[:, cpb]   # hashed pairwise crosses of the real field values
        h = (a_ * 0x9E3779B1 + b_ * 0x85EBCA77 + torch.arange(NCROSS, device=dev) * 0xC2B2AE3D) & 0x7FFFFFFFFFFF
        h = (h ^ (h >> 17)) * 0x2C1B3C6D & 0x7FFFFFFF
        return torch.cat([g, h % HCROSS + coff], 1)

    def bag_offsets(n):
        return (torch.arange(n, device=dev)[:, None] * NL + bag_off1[None, :]).reshape(-1)

    def lookup(raw):
        """raw int32 [n, NCOL] -> (global row ids 1-D, bag offsets [n*NF] or None for the one-hot fast path)."""
        if BAGS:
            g = raw.long()[:, flat_cols]; valid = g >= 0; g = fold(g, col_table[flat_cols])
            gid = (g + off[col_table[flat_cols]]).masked_select(valid)
            sizes = torch.zeros(raw.shape[0], NF, dtype=torch.int64, device=dev).index_add_(1, bag_of_flat, valid.long())
            offs = torch.cumsum(sizes.reshape(-1), 0) - sizes.reshape(-1)
            return gid, offs
        g = expand(raw)
        return g.reshape(-1), (bag_offsets(raw.shape[0]) if NL != NF else None)
    MULTI = bool(BAGS) or NL != NF
    assert a.start + steps * b <= meta['rows']
    if a.method == 'all_gpu':
        base, dyn, lazy, st = 'static', 'carry', False, 'S'
    else:
        base, dyn, lazy, st = METHODS[a.method]
    # ---- static profile (offline, FAE-style; identical for every static method) ----
    static_rows = None; C = a.budget_rows
    if base == 'static':
        if a.method == 'all_gpu':   # every row touched by the run is resident: reference execution
            seen = torch.zeros(R, dtype=torch.bool, device=dev)
            for c0 in range(a.start, a.start + steps * b, 1 << 18):
                raw = torch.from_numpy(np.asarray(sp[c0:min(a.start + steps * b, c0 + (1 << 18))], dtype=np.int32)).to(dev)
                seen[lookup(raw)[0]] = True
            static_rows = seen.nonzero().flatten().cpu(); C = len(static_rows) + a.carry_reserve; del seen
        else:
            if a.profile.startswith('day'):
                k = int(a.profile[3:]); lo, hi = meta['day_boundaries'][k], meta['day_boundaries'][k + 1]
            else:
                lo, hi = map(int, a.profile.split(':'))
            cnt = torch.zeros(R, dtype=torch.int64, device=dev)
            for c0 in range(lo, hi, 1 << 18):
                raw = torch.from_numpy(np.asarray(sp[c0:min(hi, c0 + (1 << 18))], dtype=np.int32)).to(dev)
                cnt += torch.bincount(lookup(raw)[0], minlength=R)
            ns = C - a.carry_reserve
            top = torch.topk(cnt, ns, sorted=True)   # ties broken deterministically by topk on equal keys
            static_rows = torch.sort(top.indices[top.values > 0]).values.cpu(); del cnt
    bound = float(np.sqrt(1.0 / max(cards0)))   # deterministic uniform init bound (shared by all methods)
    cfg = {'lr': a.emb_lr or a.lr, 'b1': 0.9, 'b2': 0.999, 'eps': 1e-8, 'seed': a.seed, 'bound': bound, 'max_step': steps + 2}
    K = k_auto(st) if a.K == 'auto' else int(a.K); a.K_used = K
    mon = GPUMonitor(); mon.snap()
    out_p = Path(a.out); out_p.parent.mkdir(parents=True, exist_ok=True)
    ELARA = a.method.startswith('elara')
    if ELARA:
        from srm2.elara import ELARAPlanner, ELARAExecutor
        k0 = a.method.startswith('elara_k0'); Kfix = 0 if k0 else min(K, max(1, a.elara_kmax))
        kthr0 = torch.full((len(off_np),), Kfix, dtype=torch.int8)
        pl = ELARAPlanner(R, C, max(1, a.elara_kmax), base, dyn, st, static_rows, a.L, dev, torch.from_numpy(off_np), kthr0,
                         evict_logs=0 if k0 else a.elara_evict_logs, fused_dir=bool(a.elara_fused_dir), dir_impl=a.elara_dir_impl,
                         evl_gate_frac=a.elara_evl_gate)
    else:
        pl = GPUPlanner(R, C, K, lazy, base, dyn, st, a.t_frac, static_rows, a.L, dev,
                        admit_threshold=a.admit_threshold, past_window=a.past_window)
    if a.diag and not ELARA: pl.enable_diag(torch.from_numpy(off_np))
    link = LinkModel(a.link_gbps or None, a.link_lat_us)
    try:
        if ELARA:
            ts = ELARAExecutor(R, D, cfg, pl, dev, threads=a.threads, link=link, depth=a.depth,
                              host_budget=int(a.host_budget_gb * 2**30), ahead_bytes=a.prefault_ahead_mb << 20)
        else:
            ts = TieredSparse(R, D, cfg, pl, dev, threads=a.threads, link=link, depth=a.depth, store=a.store,
                              host_budget=int(a.host_budget_gb * 2**30), ahead_bytes=a.prefault_ahead_mb << 20, fused=bool(a.fast_store))
    except HostBudgetExceeded as e:   # eager store cannot even be allocated within the host budget
        json.dump({'args': vars(a), 'method': a.method, 'status': 'host_budget_exceeded_at_setup', 'error': str(e),
                   'R': R, 'store_bytes_required': R * 3 * D * 4, 'monitor': mon.summary()}, open(a.out, 'w'), indent=1, default=float)
        print(json.dumps({'method': a.method, 'status': 'host_budget_exceeded_at_setup', 'error': str(e)}), flush=True); return
    ctl = None
    if ELARA and a.elara_adaptive:
        from srm2.elara_ctl import ELARAController
        ctl = ELARAController(pl, ts, link, D, pl.Kmax, int(kthr0[0]), every=a.elara_ctl_every, budget_bytes=int(a.host_budget_gb * 2**30))
    t0 = time.perf_counter()
    if not lazy: ts.init_host()
    t_init_host = time.perf_counter() - t0
    ts.load_static()
    MT = a.model in MULTI_TASK
    if MT:
        mt_all = json.load(open(f'{a.data}/meta_mt.json'))['tasks']; TASKS = a.tasks.split(','); tix = [mt_all.index(t) for t in TASKS]
    T = len(TASKS) if MT else 1
    model = MODELS[a.model](ND, NF, D, **({'n_tasks': T} if MT else {})).to(dev); opt = torch.optim.Adam(model.parameters(), lr=a.lr, capturable=True)
    amp = torch.autocast('cuda', dtype=torch.bfloat16, enabled=a.amp == 'bf16')
    # ---- preload the run's samples into pinned host memory (shared infrastructure for every method) ----
    n_s = steps * b
    def preload(fn, out):
        """rows [a.start, a.start+n_s) of an .npy file into a pinned tensor with plain chunked reads, dropping each
        chunk's page cache as it goes (no mapped pages, no cache build-up). Same bytes as slicing the memmap."""
        path = f'{a.data}/{fn}'
        with open(path, 'rb') as f:
            ver = np.lib.format.read_magic(f); shape, _, dt = np.lib.format._read_array_header(f, ver); hdr = f.tell()
            row = int(np.prod(shape[1:])) * np.dtype(dt).itemsize if len(shape) > 1 else np.dtype(dt).itemsize
            same = np.dtype(dt) == out.numpy().dtype   # else read in the file dtype and cast (as the old slice assignment did)
            tmp = out.numpy() if same else np.empty(out.numpy().shape, dtype=dt)
            dst = tmp.reshape(-1).view(np.uint8); pos = hdr + a.start * row; total = n_s * row; done = 0
            assert dst.nbytes == total, (fn, dst.nbytes, total)
            while done < total:
                k = min(1 << 28, total - done); f.seek(pos + done); f.readinto(memoryview(dst[done:done + k]))
                os.posix_fadvise(f.fileno(), pos + done, k, os.POSIX_FADV_DONTNEED); done += k
            if not same: out.numpy()[:] = tmp
    h_sp = torch.empty(n_s, NCOL, dtype=torch.int32, pin_memory=True); preload('sparse.npy', h_sp)
    h_dn = torch.empty(n_s, ND, dtype=torch.float32, pin_memory=True); preload('dense.npy', h_dn)
    if MT:   # multi-task labels (uint8 [N, 6] on disk) -> the selected task columns, float32 pinned
        h_all = torch.empty(n_s, len(mt_all), dtype=torch.float32); preload('labels_mt.npy', h_all)
        h_lb = torch.empty(n_s, T, dtype=torch.float32, pin_memory=True); h_lb.copy_(h_all[:, tix]); del h_all
    else:
        h_lb = torch.empty(n_s, dtype=torch.float32, pin_memory=True); preload('labels.npy', h_lb)
    # the run's samples now live in pinned memory: drop the duplicate file-cache pages of the dataset (same for every
    # method; keeps the container's memory well below its limit). Evaluation re-reads its slice from disk.
    # unmap the dataset first (mapped pages stay resident otherwise); evaluation re-opens it
    import gc; sp = dn = lb = None; gc.collect()
    for fn in ('sparse.npy', 'dense.npy', 'labels.npy'):
        try:
            fd = os.open(f'{a.data}/{fn}', os.O_RDONLY); os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED); os.close(fd)
        except OSError:
            pass
    # ---- dense part as a CUDA graph (fixed shapes); embedding gradient taken from the graph ----
    emb_s = torch.zeros(b, NF, D, device=dev, requires_grad=True); x_s = torch.zeros(b, ND, device=dev); y_s = torch.zeros((b, T) if MT else (b,), device=dev)
    snap = [q.detach().clone() for q in model.parameters()]
    side = torch.cuda.Stream(dev); side.wait_stream(torch.cuda.current_stream(dev))
    with torch.cuda.stream(side):
        for _ in range(3):
            opt.zero_grad(set_to_none=True); emb_s.grad = None
            with amp: lo_ = model(x_s, emb_s)
            F.binary_cross_entropy_with_logits(lo_.float(), y_s).backward(); opt.step()
    torch.cuda.current_stream(dev).wait_stream(side)
    graph = torch.cuda.CUDAGraph(); opt.zero_grad(set_to_none=True); emb_s.grad = None
    with torch.cuda.graph(graph):
        with amp: lo_ = model(x_s, emb_s)
        loss_s = F.binary_cross_entropy_with_logits(lo_.float(), y_s); loss_s.backward(); opt.step()
    with torch.no_grad():   # undo warm-up/capture side effects: restore initial params, zero Adam state
        for q, v in zip(model.parameters(), snap): q.copy_(v)
        for st_ in opt.state.values():
            for v in st_.values():
                if torch.is_tensor(v): v.zero_()
    torch.cuda.synchronize(dev)
    setup_s = time.perf_counter() - t_setup

    # ---- planner thread: load batches, unique on GPU, plan L steps ahead ----
    state_trace = []
    plans = queue.Queue(maxsize=6); pstream = torch.cuda.Stream(dev, priority=a.side_prio); perr = []; ptimes = []; ltimes = []
    lstream = torch.cuda.Stream(dev, priority=a.side_prio); loaded = queue.Queue(maxsize=a.L + a.depth + 4)
    def loader_thread():   # method-independent: H2D of samples, ID expansion, unique on GPU
        try:
            with torch.cuda.stream(lstream):
                for j in range(steps):
                    t0 = time.perf_counter(); lo = j * b
                    raw = h_sp[lo:lo + b].to(dev, non_blocking=True)
                    x = h_dn[lo:lo + b].to(dev, non_blocking=True); y = h_lb[lo:lo + b].to(dev, non_blocking=True)
                    ids, offs = lookup(raw)
                    u, inv = torch.unique(ids, return_inverse=True)
                    e = torch.cuda.Event(); e.record(lstream); loaded.put((u, inv, x, y, offs, e, time.perf_counter() - t0))
        except Exception:
            import traceback; perr.append(traceback.format_exc()); loaded.put(None)

    def planner_thread():
        try:
            with torch.cuda.stream(pstream):
                U = {}; B = {}; nxt = [0]
                def load(j):
                    item = loaded.get()
                    if item is None: raise RuntimeError('loader failed')
                    u, inv, x, y, offs, e, tl = item; pstream.wait_event(e)
                    for t_ in (u, inv, x, y, offs):
                        if t_ is not None: t_.record_stream(pstream)
                    U[j] = u; B[j] = (inv, x, y, offs); pl.observe(j + 1, u); ltimes.append(tl)   # 1-based logical steps
                for j in range(min(steps, a.L + 1)): load(j)
                for s in range(steps):
                    tp0 = time.perf_counter()
                    if s + a.L + 1 < steps: load(s + a.L + 1)
                    tp1 = time.perf_counter()
                    with torch.profiler.record_function('P_plan'):
                        p = pl.plan(s + 1, U[s], [U[s + k] for k in range(1, a.depth) if s + k < steps])
                    ptimes.append((tp1 - tp0, time.perf_counter() - tp1))
                    if a.trace_states and ELARA and (s + 1) % a.trace_states == 0:
                        h = torch.bincount(pl.code.long(), minlength=pl.MAT + 1).tolist()
                        state_trace.append({'step': s + 1, 'I': h[0], 'L': h[1:pl.MAT], 'M': h[pl.MAT],
                                            'resident': int((pl.slot_of >= 0).sum()), 'mat_top': int(pl.mat_top), 'live_logs': int(pl.live),
                                            'evl_on': bool(pl.evict_logs)})
                    p.inv, p.x, p.y, p.offs = B.pop(s); U.pop(s)
                    e = torch.cuda.Event(); e.record(pstream); p.ev = e
                    plans.put(p)
        except Exception:
            import traceback; perr.append(traceback.format_exc()); plans.put(None)
    threading.Thread(target=loader_thread, daemon=True).start()
    th = threading.Thread(target=planner_thread, daemon=True); th.start()

    def get_plan():
        p = plans.get()
        if p is None: raise RuntimeError('planner failed:\n' + perr[0])
        if p.barrier: ts.mark_barrier(p.s)   # registered before any later gather is queued
        return p

    def adopt(p):   # make planner-stream tensors safe to use on the compute stream
        cur = torch.cuda.current_stream(dev); cur.wait_event(p.ev)
        for v in list(vars(p).values()) + list(p.ix.values()):
            if isinstance(v, torch.Tensor) and v.is_cuda: v.record_stream(cur)

    losses = []; step_t = []; phase = []; gev = []
    torch.cuda.synchronize(dev); torch.cuda.reset_peak_memory_stats(dev)
    wall0 = time.perf_counter(); cpu0 = time.process_time()
    import collections
    ahead = collections.deque(); s = -1
    prof = None; pa, pb = (map(int, a.prof_steps.split(':')) if a.prof_steps else (-1, -1))
    try:
        for _ in range(min(steps, a.depth - 1)):
            ahead.append(get_plan()); ts.enqueue_gather(ahead[-1])
        for s in range(steps):
            if s == pa:
                prof = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA])
                prof.__enter__()
            if s == pb and prof is not None:
                torch.cuda.synchronize(dev); prof.__exit__(None, None, None)
            ts0 = time.perf_counter()
            p_cur = ahead.popleft()
            if s + a.depth - 1 < steps:
                ahead.append(get_plan()); ts.enqueue_gather(ahead[-1])   # G(s+d-1) queued after U(s-1)
            ts1 = time.perf_counter()
            adopt(p_cur)
            ge0 = torch.cuda.Event(enable_timing=True); ge0.record()
            with torch.profiler.record_function('M_assemble'):
                W, bufs = ts.assemble(p_cur)
            ts2 = time.perf_counter()
            if not MULTI:
                with torch.no_grad():
                    torch.index_select(W, 0, p_cur.inv, out=emb_s.detach().view(-1, D)); x_s.copy_(p_cur.x); y_s.copy_(p_cur.y)
                graph.replay()
                gW = torch.ops.aten.embedding_dense_backward(emb_s.grad.view(-1, D), p_cur.inv, W.shape[0], -1, False)
            else:
                W.requires_grad_(True)
                pooled = F.embedding_bag(p_cur.inv, W, p_cur.offs, mode='sum')
                with torch.no_grad():
                    emb_s.detach().view(-1, D).copy_(pooled); x_s.copy_(p_cur.x); y_s.copy_(p_cur.y)
                graph.replay()
                pooled.backward(emb_s.grad.view(-1, D)); gW = W.grad
            loss = loss_s.detach().clone()
            ts3 = time.perf_counter()
            with torch.profiler.record_function('M_update'):
                ts.update(p_cur, gW, bufs)
            if ctl is not None: ctl.step(s + 1)
            ge1 = torch.cuda.Event(enable_timing=True); ge1.record(); gev.append((ge0, ge1))
            losses.append(loss.detach())
            ts4 = time.perf_counter(); step_t.append(ts4 - ts0); phase.append((ts1 - ts0, ts2 - ts1, ts3 - ts2, ts4 - ts3))
        ts.drain(); torch.cuda.synchronize(dev)
    except HostBudgetExceeded as e:   # the store outgrew the enforced host budget during training
        rss, hwm = proc_mem()
        json.dump({'args': vars(a), 'method': a.method, 'status': 'host_budget_exceeded_in_training', 'error': str(e),
                   'failed_at_step': s + 1, 'steps': steps, 'elapsed_s': time.perf_counter() - wall0, 'vm_hwm': hwm,
                   'store_footprint': ts.store.footprint(), 'footprint_trace': ts.footprint_trace, 'monitor': mon.summary()},
                  open(a.out, 'w'), indent=1, default=float)
        print(json.dumps({'method': a.method, 'status': 'host_budget_exceeded_in_training', 'failed_at_step': s + 1}), flush=True)
        os._exit(0)
    train_s = time.perf_counter() - wall0; cpu_s = time.process_time() - cpu0
    prof_summary = None
    if prof is not None:
        from srm2.profsum import summarize
        prof.export_chrome_trace(str(Path(a.out).with_suffix('.trace.json')))
        prof_summary = summarize(prof, pb - pa)
    th.join()
    loss_vals = torch.stack(losses).float().cpu().numpy()
    assert np.isfinite(loss_vals).all()
    res = {'args': vars(a), 'method': a.method, 'R': R, 'C': C, 'static_rows': 0 if static_rows is None else int(len(static_rows)),
           'setup_s': setup_s, 'init_host_s': t_init_host, 'train_s': train_s, 'process_cpu_s': cpu_s,
           'samples_per_s': steps * b / train_s, 'step_ms_median': float(np.median(step_t) * 1e3),
           'loss_first': float(loss_vals[0]), 'loss_last_mean100': float(loss_vals[-100:].mean()),
           'planner': pl.stats, 'steps': steps, 'timers': ts.timers, 'counts': ts.counts,
           'link': {'bytes': link.bytes, 'busy_s': link.busy, 'waited_s': link.waited},
           'peak_gpu_alloc_gb': torch.cuda.max_memory_allocated(dev) / 2**30, 'peak_gpu_reserved_gb': torch.cuda.max_memory_reserved(dev) / 2**30,
           'pool_gb': ts.pool.numel() * 4 / 2**30, 'status': 'complete', 'K_used': K,
           'store_footprint': ts.store.footprint(), 'footprint_trace': ts.footprint_trace,
           'vm_rss_end': proc_mem()[0], 'vm_hwm': proc_mem()[1], 'monitor': mon.summary(), 'profile': prof_summary,
           'elara_ctl': ctl.history if ctl is not None else None, 'state_trace': state_trace or None}
    if ELARA:   # directory invariants (leak / consistency check): live == valid entries; live + free == arena top
        ev_valid = int((pl.ent[:R * pl.Kmax] >= 0).sum()); live = int(pl.live); nfree = int(pl.n_lfree); top = int(pl.log_top)
        res['dir_check'] = {'live': live, 'ent_valid': ev_valid, 'n_free': nfree, 'log_top': top, 'mat_top': int(pl.mat_top),
                            'ok': live == ev_valid and live + nfree == top}
    if pl.diag is not None:   # per-table distribution of final update counts (rows touched >= 1)
        dg = pl.diag; tc = dg['tc'].long(); toff = dg['toff']; tid = torch.searchsorted(toff, torch.arange(R, device=dev), right=True) - 1
        touched = tc > 0; per_table = []
        for t in range(len(off_np)):
            m = touched & (tid == t)
            per_table.append(torch.bincount(torch.clamp(tc[m], max=16), minlength=17)[:17].tolist())
        res['diag'] = {k: dg[k].tolist() for k in ('h_ev_from_lazy', 'h_ev_from_mat', 'h_cmp')} | {'per_table_count_hist': per_table}
    w0, w1 = steps // 4, steps * 3 // 4   # steady-state window
    ph = np.array(phase[w0:w1]) * 1e3; pt = np.array(ptimes[w0:w1]) * 1e3
    gpu_ms = np.array([e0.elapsed_time(e1) for e0, e1 in gev[w0:w1]])
    res['steady'] = {'host_step_ms': float(np.mean(step_t[w0:w1]) * 1e3), 'host_get_plan_ms': float(ph[:, 0].mean()),
                     'host_assemble_ms': float(ph[:, 1].mean()), 'host_fwdbwd_ms': float(ph[:, 2].mean()), 'host_update_ms': float(ph[:, 3].mean()),
                     'gpu_step_ms': float(gpu_ms.mean()), 'planner_load_ms': float(pt[:, 0].mean()), 'loader_ms': float(np.mean(ltimes[w0:w1]) * 1e3), 'planner_plan_ms': float(pt[:, 1].mean()),
                     'train_ms_per_step': train_s / steps * 1e3}
    # ---- evaluation (read-only) ----
    if a.eval_samples:
        sp = np.load(f'{a.data}/sparse.npy', mmap_mode='r'); dn = np.load(f'{a.data}/dense.npy', mmap_mode='r'); lb = np.load(f'{a.data}/labels.npy', mmap_mode='r')
        es = a.eval_start if a.eval_start >= 0 else meta['day_boundaries'][6]
        model.eval(); preds = []; t0 = time.perf_counter()
        with torch.no_grad():
            for lo in range(es, es + a.eval_samples, b):
                hi = min(lo + b, es + a.eval_samples)
                ids, offs = lookup(torch.from_numpy(np.asarray(sp[lo:hi], dtype=np.int32)).to(dev))
                u, inv = torch.unique(ids, return_inverse=True)
                if offs is None: offs = torch.arange(len(inv), device=dev)
                emb = F.embedding_bag(inv, ts.eval_weights(u), offs, mode='sum').view(hi - lo, NF, D)
                with amp: lo_ = model(torch.from_numpy(np.asarray(dn[lo:hi])).to(dev), emb)
                preds.append(torch.sigmoid(lo_.float()).cpu())
        pr = torch.cat(preds).numpy()
        from sklearn.metrics import roc_auc_score, log_loss
        if MT:   # per-task AUC / LogLoss; 'auc' = mean over tasks
            ymt = np.asarray(np.load(f'{a.data}/labels_mt.npy', mmap_mode='r')[es:es + a.eval_samples])[:, tix].astype(np.float64)
            aucs = {t: float(roc_auc_score(ymt[:, i], pr[:, i])) for i, t in enumerate(TASKS)}
            lls = {t: float(log_loss(ymt[:, i], pr[:, i].astype(np.float64).clip(1e-7, 1 - 1e-7))) for i, t in enumerate(TASKS)}
            res['eval'] = {'auc': float(np.mean(list(aucs.values()))), 'auc_per_task': aucs, 'logloss_per_task': lls,
                           'eval_s': time.perf_counter() - t0}
        else:
            yv = np.asarray(lb[es:es + a.eval_samples]).astype(np.float64)
            res['eval'] = {'auc': float(roc_auc_score(yv, pr)), 'logloss': float(log_loss(yv, pr.astype(np.float64).clip(1e-7, 1 - 1e-7))),
                           'eval_s': time.perf_counter() - t0}
        if a.save_pred: np.save(Path(a.out).with_suffix('.pred.npy'), pr)
    ts.close()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out, 'w'), indent=1, default=float)
    print(json.dumps({k: res[k] for k in ('method', 'train_s', 'samples_per_s', 'step_ms_median', 'loss_last_mean100', 'steady')} | ({'auc': res['eval']['auc']} if 'eval' in res else {})), flush=True)


if __name__ == '__main__':
    main()

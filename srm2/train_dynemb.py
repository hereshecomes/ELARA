"""Faithful external baseline: NVIDIA DynamicEmb (recsys-examples corelib/dynamicemb, built from source),
BatchedDynamicEmbeddingTablesV2 with fused Adam. Default 'hybrid' storage: each table's hash table holds `budget_rows`
HBM rows in total (split over tables in proportion to their per-step working set, profiled on the first 5 steps of the
window, like the static hot-set profile of srm2; small tables entirely) and the rest of its rows in host memory.
'caching' (HBM cache in front of a full host table) produced non-finite embeddings at batch >= 16K with this stack
(fbgemm 1.6.0 / torch 2.11) and is not used; 'device' keeps everything in HBM. Rows are created on first access with the same uniform bound as srm2.
Same chronological data, dense backbone, dense Adam, batch size and evaluation protocol as srm2/train.py and
train_fbgemm.py. REAL hardware only (host table over PCIe; no link emulation). Not bitwise comparable to srm2.
Needs a separate environment with DynamicEmb installed (see README)."""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from srm2.models import MODELS
from srm2.store import proc_mem
from fbgemm_gpu.split_table_batched_embeddings_ops_common import BoundsCheckMode
from dynamicemb import (DynamicEmbTableOptions, DynamicEmbInitializerArgs, DynamicEmbInitializerMode, DynamicEmbScoreStrategy,
                        DynamicEmbPoolingMode, EmbOptimType)
from dynamicemb.batched_dynamicemb_tables import BatchedDynamicEmbeddingTablesV2
from dynamicemb.optimizer import get_optimizer_state_dim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data/criteo_kaggle'); ap.add_argument('--model', default='dlrm', choices=sorted(MODELS))
    ap.add_argument('--batch', type=int, default=16384); ap.add_argument('--dim', type=int, default=128)
    ap.add_argument('--start', type=int, default=0); ap.add_argument('--steps', type=int, default=200)
    ap.add_argument('--budget-rows', type=int, default=337626); ap.add_argument('--location', default='hybrid', choices=['hybrid', 'caching', 'device'])
    ap.add_argument('--cache-algo', default='lru', choices=['lru', 'lfu']); ap.add_argument('--prefetch', type=int, default=1)
    ap.add_argument('--lr', type=float, default=1e-3); ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--eval-start', type=int, default=-1); ap.add_argument('--eval-samples', type=int, default=0)
    ap.add_argument('--amp', default='none', choices=['none', 'bf16']); ap.add_argument('--out', required=True)
    ap.add_argument('--optim', default='adam', choices=['adam'], help='sparse optimizer (dense stays Adam)')
    ap.add_argument('--dedup', type=int, default=1, help='use_index_dedup')
    ap.add_argument('--emb-lr', type=float, default=0, help='sparse learning rate (0 = --lr)')
    a = ap.parse_args(); t_setup = time.perf_counter()
    dev = torch.device('cuda:0'); torch.cuda.set_device(dev); torch.manual_seed(a.seed); np.random.seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    meta = json.load(open(f'{a.data}/meta.json')); cards = meta['cardinalities']; R = int(sum(cards)); NF = len(cards)
    sp = np.load(f'{a.data}/sparse.npy', mmap_mode='r'); dn = np.load(f'{a.data}/dense.npy', mmap_mode='r'); lb = np.load(f'{a.data}/labels.npy', mmap_mode='r')
    NCOL = sp.shape[1]; BAGS = meta.get('bags')   # data-defined multi-hot bags over shared tables (KuaiRand), -1 = padding
    ftm = [meta['col_table'][bag[0]] for bag in BAGS] if BAGS else None
    # DynamicEmb requires the features of a shared table to be adjacent (KuaiRand: the history bag shares the video
    # table with the target video feature): feed the features sorted by table, restore the model's field order after
    perm = sorted(range(len(ftm)), key=lambda f: (ftm[f], f)) if ftm else None
    invp = torch.tensor([perm.index(f) for f in range(len(perm))]) if ftm else None
    if ftm: BAGS = [BAGS[f] for f in perm]; ftm = [ftm[f] for f in perm]
    if BAGS: NF = len(BAGS)

    def make_ids(raw):
        """raw int64 [n, NCOL] on GPU -> (feature-major indices, offsets [NF*n+1]) for the TBE."""
        n = raw.shape[0]
        if not BAGS:
            return raw.t().contiguous().view(-1), torch.arange(NF * n + 1, device=dev, dtype=torch.int64)
        parts, lens = [], []
        for bag in BAGS:
            v = raw[:, bag]; m = v >= 0; parts.append(v[m]); lens.append(m.sum(1))
        L = torch.stack(lens).view(-1)
        return torch.cat(parts), torch.cat([torch.zeros(1, dtype=torch.int64, device=dev), torch.cumsum(L, 0)])
    b, D, steps, ND = a.batch, a.dim, a.steps, dn.shape[1]
    need = R * D * 4 * 3 if a.optim == 'adam' else R * (D + 1) * 4   # weights + Adam moments / row-wise AdaGrad scalar
    avail = int(open('/proc/meminfo').read().split('MemAvailable:')[1].split()[0]) * 1024
    if a.location != 'device' and need > avail - (12 << 30):   # never risk the host OOM killer on a shared machine
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump({'args': vars(a), 'method': f'dynamicemb_{a.location}', 'status': 'infeasible_host_memory', 'R': R,
                   'tables_bytes_weights_plus_adam': need, 'mem_available': avail}, open(a.out, 'w'), indent=1)
        print(json.dumps({'status': 'infeasible_host_memory', 'need_gb': need / 2**30, 'avail_gb': avail / 2**30})); return
    invp_d = invp.to(dev) if invp is not None else None
    t0 = time.perf_counter()
    bound = float(np.sqrt(1.0 / max(cards)))   # same uniform init bound as srm2 (values differ: DynamicEmb's own RNG)
    vdim = D + get_optimizer_state_dim(EmbOptimType.ADAM, D, torch.float32)
    col_table = meta.get('col_table', list(range(NCOL)))
    uq = np.zeros(len(cards))
    for s0 in range(5):   # per-table unique ids per step at the start of the window (working-set profile)
        r0 = np.asarray(sp[a.start + s0 * a.batch:a.start + (s0 + 1) * a.batch], dtype=np.int64)
        for t in range(len(cards)):
            v = r0[:, [j for j in range(NCOL) if col_table[j] == t]].reshape(-1); uq[t] += len(np.unique(v[v >= 0]))
    opts = []
    for t, c in enumerate(cards):
        rows = c if a.location == 'device' else min(c, max(128, int(a.budget_rows * uq[t] / uq.sum())))
        opts.append(DynamicEmbTableOptions(index_type=torch.int64, embedding_dtype=torch.float32, dim=D, max_capacity=c,
                                           local_hbm_for_values=int(rows * vdim * 4), bucket_capacity=128,
                                           initializer_args=DynamicEmbInitializerArgs(mode=DynamicEmbInitializerMode.UNIFORM, lower=-bound, upper=bound),
                                           score_strategy=DynamicEmbScoreStrategy.LFU if a.cache_algo == 'lfu' else DynamicEmbScoreStrategy.TIMESTAMP,
                                           caching=a.location == 'caching'))
    tbe = BatchedDynamicEmbeddingTablesV2(table_options=opts, table_names=[f't{i}' for i in range(len(cards))],
                                          **({'feature_table_map': ftm} if ftm else {}), use_index_dedup=bool(a.dedup),
                                          pooling_mode=DynamicEmbPoolingMode.SUM, output_dtype=torch.float32, device=dev,
                                          optimizer=EmbOptimType.ADAM, learning_rate=a.emb_lr or a.lr, eps=1e-8, weight_decay=0.0,
                                          beta1=0.9, beta2=0.999, stochastic_rounding=False, bounds_check_mode=BoundsCheckMode.NONE)
    torch.cuda.synchronize(dev); init_s = time.perf_counter() - t0
    model = MODELS[a.model](ND, NF, D).to(dev); opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    amp = torch.autocast('cuda', dtype=torch.bfloat16, enabled=a.amp == 'bf16')
    n_s = steps * b
    h_sp = torch.empty(n_s, NCOL, dtype=torch.int32, pin_memory=True); h_sp.numpy()[:] = sp[a.start:a.start + n_s]
    h_dn = torch.empty(n_s, ND, dtype=torch.float32, pin_memory=True); h_dn.numpy()[:] = dn[a.start:a.start + n_s]
    h_lb = torch.empty(n_s, dtype=torch.float32, pin_memory=True); h_lb.numpy()[:] = lb[a.start:a.start + n_s]
    def batch(j):   # feature-major indices (one-hot: pooling factor 1; bags: variable lengths)
        lo = j * b
        ids, offs = make_ids(h_sp[lo:lo + b].to(dev, non_blocking=True).long())
        return ids, offs, h_dn[lo:lo + b].to(dev, non_blocking=True), h_lb[lo:lo + b].to(dev, non_blocking=True)
    pstream = torch.cuda.Stream(dev)
    setup_s = time.perf_counter() - t_setup
    torch.cuda.synchronize(dev); torch.cuda.reset_peak_memory_stats(dev)
    wall0 = time.perf_counter(); losses = []; nxt = batch(0)
    for s in range(steps):
        ids, offsets, x, y = nxt
        if s + 1 < steps:
            nxt = batch(s + 1)
        emb = tbe(ids, offsets).view(b, NF, D)
        if invp is not None: emb = emb[:, invp_d]
        with amp: lo_ = model(x, emb)
        loss = F.binary_cross_entropy_with_logits(lo_.float(), y)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()   # TBE applies fused Adam in backward
        losses.append(loss.detach())
    torch.cuda.synchronize(dev); train_s = time.perf_counter() - wall0
    lv = torch.stack(losses).float().cpu().numpy(); assert np.isfinite(lv).all()
    rss, hwm = proc_mem()
    res = {'args': vars(a), 'method': f'dynamicemb_{a.location}_{a.cache_algo}', 'status': 'complete',
           'R': R, 'setup_s': setup_s, 'init_tables_s': init_s, 'train_s': train_s, 'steps': steps, 'samples_per_s': steps * b / train_s,
           'train_ms_per_step': train_s / steps * 1e3, 'loss_last_mean100': float(lv[-100:].mean()),
           'peak_gpu_alloc_gb': torch.cuda.max_memory_allocated(dev) / 2**30, 'vm_rss_end': rss, 'vm_hwm': hwm,
           'tables_bytes_weights_plus_adam': need, 'optim': a.optim, 'hardware': 'real (host table over PCIe)', 'hbm_rows_total': int(sum(o.local_hbm_for_values // (vdim * 4) for o in opts)), 'storage': a.location}
    if a.eval_samples:
        es = a.eval_start if a.eval_start >= 0 else meta['day_boundaries'][6]; preds = []; t0 = time.perf_counter()
        model.eval()
        with torch.no_grad():
            for lo in range(es, es + a.eval_samples, b):
                hi = min(lo + b, es + a.eval_samples); n = hi - lo
                ids, off = make_ids(torch.from_numpy(np.asarray(sp[lo:hi], dtype=np.int64)).to(dev))
                emb = tbe(ids, off).view(n, NF, D)
                if invp is not None: emb = emb[:, invp_d]
                with amp: lo_ = model(torch.from_numpy(np.asarray(dn[lo:hi])).to(dev), emb)
                preds.append(torch.sigmoid(lo_.float()).cpu())
        pr = torch.cat(preds).numpy(); yv = np.asarray(lb[es:es + a.eval_samples]).astype(np.float64)
        from sklearn.metrics import roc_auc_score, log_loss
        res['eval'] = {'auc': float(roc_auc_score(yv, pr)), 'logloss': float(log_loss(yv, pr.astype(np.float64).clip(1e-7, 1 - 1e-7))),
                       'eval_s': time.perf_counter() - t0}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True); json.dump(res, open(a.out, 'w'), indent=1, default=float)
    print(json.dumps({k: res[k] for k in ('method', 'train_s', 'train_ms_per_step', 'init_tables_s', 'vm_hwm')} | ({'auc': res['eval']['auc']} if 'eval' in res else {})), flush=True)


if __name__ == '__main__':
    main()

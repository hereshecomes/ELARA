"""Faithful external baseline: FBGEMM TBE (the fused embedding-bag + optimizer kernel used by TorchRec's
FUSED / FUSED_UVM_CACHING compute kernels) with fused row-sparse Adam.

Embedding tables live in UVM host memory (EmbeddingLocation.MANAGED_CACHING) with an HBM software cache of
`budget_rows` rows (LRU or LFU); Adam moments are allocated by FBGEMM itself (for caching tables FBGEMM keeps
non-row-wise moments in UVM, cacheable=False). Optional `--location device` keeps everything in HBM when it
fits. Same chronological data, dense backbone, dense Adam, batch size and evaluation protocol as srm2/train.py.
Supports data-defined multi-hot bags over shared tables (KuaiRand history bag -> video table) via feature_table_map.
REAL hardware only (UVM traffic is real PCIe; no link emulation). Not bitwise comparable to srm2 (different
kernel arithmetic and FBGEMM's own weight initialization); compare AUC / time / memory.
"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from srm2.models import MODELS
from srm2.store import proc_mem
from fbgemm_gpu.split_table_batched_embeddings_ops_training import SplitTableBatchedEmbeddingBagsCodegen, ComputeDevice
from fbgemm_gpu.split_table_batched_embeddings_ops_common import EmbeddingLocation, CacheAlgorithm, PoolingMode
from fbgemm_gpu.split_embedding_configs import EmbOptimType as OptimType


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data/criteo_kaggle'); ap.add_argument('--model', default='dlrm', choices=sorted(MODELS))
    ap.add_argument('--batch', type=int, default=16384); ap.add_argument('--dim', type=int, default=128)
    ap.add_argument('--start', type=int, default=0); ap.add_argument('--steps', type=int, default=200)
    ap.add_argument('--budget-rows', type=int, default=337626); ap.add_argument('--location', default='caching', choices=['caching', 'managed', 'device'])
    ap.add_argument('--cache-algo', default='lru', choices=['lru', 'lfu']); ap.add_argument('--prefetch', type=int, default=1)
    ap.add_argument('--lr', type=float, default=1e-3); ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--eval-start', type=int, default=-1); ap.add_argument('--eval-samples', type=int, default=0)
    ap.add_argument('--amp', default='none', choices=['none', 'bf16']); ap.add_argument('--out', required=True)
    ap.add_argument('--optim', default='adam', choices=['adam', 'rowwise_adagrad'], help='sparse optimizer (dense stays Adam)')
    ap.add_argument('--emb-lr', type=float, default=0, help='sparse learning rate (0 = --lr)')
    a = ap.parse_args(); t_setup = time.perf_counter()
    dev = torch.device('cuda:0'); torch.cuda.set_device(dev); torch.manual_seed(a.seed); np.random.seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    meta = json.load(open(f'{a.data}/meta.json')); cards = meta['cardinalities']; R = int(sum(cards)); NF = len(cards)
    sp = np.load(f'{a.data}/sparse.npy', mmap_mode='r'); dn = np.load(f'{a.data}/dense.npy', mmap_mode='r'); lb = np.load(f'{a.data}/labels.npy', mmap_mode='r')
    NCOL = sp.shape[1]; BAGS = meta.get('bags')   # data-defined multi-hot bags over shared tables (KuaiRand), -1 = padding
    ftm = [meta['col_table'][bag[0]] for bag in BAGS] if BAGS else None
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
        json.dump({'args': vars(a), 'method': f'fbgemm_{a.location}', 'status': 'infeasible_host_memory', 'R': R,
                   'tables_bytes_weights_plus_adam': need, 'mem_available': avail}, open(a.out, 'w'), indent=1)
        print(json.dumps({'status': 'infeasible_host_memory', 'need_gb': need / 2**30, 'avail_gb': avail / 2**30})); return
    loc = {'caching': EmbeddingLocation.MANAGED_CACHING, 'managed': EmbeddingLocation.MANAGED, 'device': EmbeddingLocation.DEVICE}[a.location]
    t0 = time.perf_counter()
    tbe = SplitTableBatchedEmbeddingBagsCodegen(
        embedding_specs=[(c, D, loc, ComputeDevice.CUDA) for c in cards],
        optimizer=OptimType.ADAM if a.optim == 'adam' else OptimType.EXACT_ROWWISE_ADAGRAD,
        learning_rate=a.emb_lr or a.lr, eps=1e-8, beta1=0.9, beta2=0.999, weight_decay=0.0,
        cache_load_factor=a.budget_rows / R, cache_algorithm=CacheAlgorithm.LRU if a.cache_algo == 'lru' else CacheAlgorithm.LFU,
        pooling_mode=PoolingMode.SUM, prefetch_pipeline=False, device=dev,
        **({'feature_table_map': ftm} if ftm else {}))
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
        if a.prefetch and a.location == 'caching':   # TorchRec TrainPipelinePrefetch-style: cache prefetch for this batch
            tbe.prefetch(ids, offsets)
        if s + 1 < steps:
            nxt = batch(s + 1)
        emb = tbe(ids, offsets).view(b, NF, D)
        with amp: lo_ = model(x, emb)
        loss = F.binary_cross_entropy_with_logits(lo_.float(), y)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()   # TBE applies fused Adam in backward
        losses.append(loss.detach())
    torch.cuda.synchronize(dev); train_s = time.perf_counter() - wall0
    lv = torch.stack(losses).float().cpu().numpy(); assert np.isfinite(lv).all()
    rss, hwm = proc_mem()
    res = {'args': vars(a), 'method': f'fbgemm_{a.location}_{a.cache_algo}' + ('_pf' if a.prefetch else '') + ('' if a.optim == 'adam' else '_' + a.optim), 'status': 'complete',
           'R': R, 'setup_s': setup_s, 'init_tables_s': init_s, 'train_s': train_s, 'steps': steps, 'samples_per_s': steps * b / train_s,
           'train_ms_per_step': train_s / steps * 1e3, 'loss_last_mean100': float(lv[-100:].mean()),
           'peak_gpu_alloc_gb': torch.cuda.max_memory_allocated(dev) / 2**30, 'vm_rss_end': rss, 'vm_hwm': hwm,
           'tables_bytes_weights_plus_adam': need, 'optim': a.optim, 'hardware': 'real (UVM over PCIe)'}
    if a.eval_samples:
        es = a.eval_start if a.eval_start >= 0 else meta['day_boundaries'][6]; preds = []; t0 = time.perf_counter()
        model.eval()
        with torch.no_grad():
            for lo in range(es, es + a.eval_samples, b):
                hi = min(lo + b, es + a.eval_samples); n = hi - lo
                ids, off = make_ids(torch.from_numpy(np.asarray(sp[lo:hi], dtype=np.int64)).to(dev))
                emb = tbe(ids, off).view(n, NF, D)
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

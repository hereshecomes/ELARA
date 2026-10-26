"""KuaiRand-27K (Kuaishou; salmon1802/KuaiRand block layout) -> chronological srm2 dataset with a natural
multi-hot feature.

Per training row (one user-video impression): user_id, video_id, 4 video attributes, user static profile
features, tab / hour / day-of-week context, and the user's previous H videos (natural multi-hot history bag,
sharing the video embedding table), label is_click. Timestamps come from user_info.full_timestamp_seq at the
row's sequence position; rows from the selected user blocks are globally sorted by timestamp (stable), so the
stream is chronological. IDs are dense, assigned in first-occurrence order of the chronological stream.
meta.json adds: col_table (table of each sparse column), bags (columns pooled into each model field; -1 =
padding entry, skipped), cardinalities per table, day boundaries (UTC+8 calendar days).
Usage: prep_kuairand.py SRC_DIR OUT_DIR H BLOCK [BLOCK ...]
"""
import json, os, sys, time
import numpy as np, pyarrow.parquet as pq

src, out, H = sys.argv[1], sys.argv[2], int(sys.argv[3]); blocks = sys.argv[4:]
os.makedirs(out, exist_ok=True); t0 = time.time()
SCALAR = None; parts = []
for b in blocks:
    d = pq.read_table(f'{src}/train/data/part-{b}.parquet')
    ui = pq.read_table(f'{src}/train/user_info/part-{b}.parquet')
    ii = pq.read_table(f'{src}/train/item_info/part-{b}.parquet')
    if SCALAR is None:
        LABELS = {'is_click', 'is_like', 'is_follow', 'is_comment', 'is_forward', 'long_view'}
        SCALAR = [c for c in d.column_names if c not in ('user_index', 'item_index', 'seq_len') and c not in LABELS]
    imap = np.zeros(ii.column('item_index').to_numpy().max() + 1, np.int64)
    imap[ii.column('item_index').to_numpy()] = ii.column('item_id').to_numpy()
    feats = {c: np.zeros(len(imap), np.int64) for c in ('video_type', 'primary_tag', 'music_type', 'duration_bucket')}
    for c in feats: feats[c][ii.column('item_index').to_numpy()] = ii.column(c).to_numpy()
    uidx = ui.column('user_index').to_numpy()
    seq_items = ui.column('full_item_seq').combine_chunks(); seq_ts = ui.column('full_timestamp_seq').combine_chunks()
    flat_items = seq_items.values.to_numpy(); flat_ts = seq_ts.values.to_numpy(); offs = seq_items.offsets.to_numpy()
    start = np.zeros(uidx.max() + 1, np.int64); start[uidx] = offs[:-1]
    u = d.column('user_index').to_numpy().astype(np.int64); sl = d.column('seq_len').to_numpy().astype(np.int64)
    pos = start[u] + sl
    ts = flat_ts[pos]
    item_local = d.column('item_index').to_numpy()
    assert (flat_items[pos] == item_local).mean() > 0.999, 'seq_len does not index the user sequence'
    cols = {'video_id': imap[item_local].astype(np.int32)}
    for c in feats: cols[c] = feats[c][item_local].astype(np.int32)
    for c in SCALAR: cols[c] = d.column(c).to_numpy().astype(np.int32)
    for k in range(1, H + 1):
        p = sl - k
        cols[f'hist{k}'] = np.where(p >= 0, imap[flat_items[np.maximum(start[u] + p, 0)]], -1).astype(np.int32)
    parts.append({'ts': ts, 'label': d.column('is_click').to_numpy().astype(np.uint8), **cols})
    print('block', b, len(ts), round(time.time() - t0, 1), flush=True)
    del d, ui, ii
n = sum(len(p['ts']) for p in parts)
ts = np.concatenate([p['ts'] for p in parts]); order = np.argsort(ts, kind='stable'); ts = ts[order]
names = ['user_id', 'video_id', 'video_type', 'primary_tag', 'music_type', 'duration_bucket'] + [c for c in SCALAR if c != 'user_id']
hist = [f'hist{k}' for k in range(1, H + 1)]
# remapped columns go to a column-major temp file (contiguous writes), then are transposed into the row-major output
# in row chunks: ~2x the output size in writes instead of one full rewrite of the output per column
NC = len(names) + H
colm = np.lib.format.open_memmap(f'{out}/sparse_colmajor.tmp.npy', mode='w+', dtype=np.int32, shape=(NC, n))
cards = []; col_table = []
for j, c in enumerate(names):
    v = np.concatenate([p[c] for p in parts])[order]
    if c == 'video_id':   # dense video IDs in first-occurrence order of the chronological stream (targets, then histories)
        uq, first = np.unique(v, return_index=True); rank = np.empty(len(uq), np.int64); rank[np.argsort(first, kind='stable')] = np.arange(len(uq))
        vid_uq, vid_rank = uq, rank
        colm[j] = rank[np.searchsorted(uq, v)]; cards.append(len(uq)); vid_table = len(cards) - 1
    else:
        uq, first = np.unique(v, return_index=True); rank = np.empty(len(uq), np.int64); rank[np.argsort(first, kind='stable')] = np.arange(len(uq))
        colm[j] = rank[np.searchsorted(uq, v)]; cards.append(len(uq))
    col_table.append(len(cards) - 1); print(c, cards[-1], round(time.time() - t0, 1), flush=True)
extra = []
for k, c in enumerate(hist):
    v = np.concatenate([p[c] for p in parts])[order]
    m = v >= 0; idx = np.searchsorted(vid_uq, v[m]); ok = (idx < len(vid_uq)) & (vid_uq[np.minimum(idx, len(vid_uq) - 1)] == v[m])
    if (~ok).any():   # history videos never seen as a target in the selected window get new IDs after the targets
        extra.append(np.unique(v[m][~ok]))
    out_col = np.full(n, -1, np.int64); vv = np.full(m.sum(), -1, np.int64); vv[ok] = vid_rank[idx[ok]]; out_col[m] = vv
    colm[len(names) + k] = out_col; col_table.append(vid_table)
if extra:   # assign IDs to history-only videos and fill them in
    ex = np.unique(np.concatenate(extra)); base = cards[vid_table]; cards[vid_table] += len(ex)
    for k, c in enumerate(hist):
        v = np.concatenate([p[c] for p in parts])[order]; col = np.asarray(colm[len(names) + k]).astype(np.int64)
        miss = (v >= 0) & (col < 0); col[miss] = base + np.searchsorted(ex, v[miss]); colm[len(names) + k] = col
colm.flush()
sparse = np.lib.format.open_memmap(f'{out}/sparse.npy', mode='w+', dtype=np.int32, shape=(n, NC))
for lo in range(0, n, 1 << 22):
    hi = min(n, lo + (1 << 22)); sparse[lo:hi] = np.asarray(colm[:, lo:hi]).T
del colm; os.remove(f'{out}/sparse_colmajor.tmp.npy')
labels = np.concatenate([p['label'] for p in parts])[order]
day = ((ts // 1000 + 8 * 3600) // 86400).astype(np.int64); day -= day.min()
hourv = np.asarray(sparse[:, names.index('hour')]).astype(np.float32); dowv = np.asarray(sparse[:, names.index('day_of_week')]).astype(np.float32)   # (hourv unused)
dense = np.stack([((ts // 1000 + 8 * 3600) % 86400 / 86400.0).astype(np.float32), dowv / max(1.0, dowv.max())], 1)
np.save(f'{out}/dense.npy', dense); np.save(f'{out}/labels.npy', labels); np.save(f'{out}/timestamps.npy', ts); sparse.flush()
bags = [[j] for j in range(len(names))] + [list(range(len(names), len(names) + H))]
bounds = [0] + [int(np.searchsorted(day, k)) for k in range(1, int(day.max()) + 1)] + [n]
json.dump({'source': src, 'blocks': blocks, 'rows': int(n), 'columns': names + hist, 'tables': names, 'cardinalities': cards,
           'col_table': col_table, 'bags': bags, 'bag_names': names + ['video_history'], 'history_len': H,
           'total_rows': int(sum(cards)), 'day_boundaries': bounds, 'ctr': float(labels.mean()),
           'order': 'global stable sort by impression timestamp (ms)', 'label': 'is_click', 'elapsed_s': time.time() - t0},
          open(f'{out}/meta.json', 'w'), indent=1)
print('done rows', n, 'tables', len(cards), 'video vocab', cards[vid_table], 'days', len(bounds) - 1, round(time.time() - t0, 1), flush=True)

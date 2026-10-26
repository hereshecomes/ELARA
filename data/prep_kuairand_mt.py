"""Multi-task labels for an existing KuaiRand srm2 dataset (prep_kuairand.py output), in exactly its row order.

Recomputes the per-row impression timestamps of the same blocks the same way, applies the same global stable sort, and
writes OUT/labels_mt.npy (uint8 [N, 6]; columns in TASKS order) + OUT/meta_mt.json. Alignment is verified against the
existing dataset: the reordered timestamps must equal timestamps.npy and the reordered is_click must equal labels.npy.
Usage: prep_kuairand_mt.py SRC_DIR OUT_DIR BLOCK [BLOCK ...]
"""
import json, sys, time
import numpy as np, pyarrow.parquet as pq

TASKS = ['is_click', 'long_view', 'is_like', 'is_follow', 'is_comment', 'is_forward']
src, out, blocks = sys.argv[1], sys.argv[2], sys.argv[3:]
meta = json.load(open(f'{out}/meta.json')); assert meta['blocks'] == blocks, (meta['blocks'], blocks)
t0 = time.time(); ts_parts, lab_parts = [], []
for b in blocks:
    d = pq.read_table(f'{src}/train/data/part-{b}.parquet', columns=['user_index', 'item_index', 'seq_len'] + TASKS)
    ui = pq.read_table(f'{src}/train/user_info/part-{b}.parquet', columns=['user_index', 'full_timestamp_seq'])
    uidx = ui.column('user_index').to_numpy(); seq_ts = ui.column('full_timestamp_seq').combine_chunks()
    flat_ts = seq_ts.values.to_numpy(); offs = seq_ts.offsets.to_numpy()
    start = np.zeros(uidx.max() + 1, np.int64); start[uidx] = offs[:-1]
    u = d.column('user_index').to_numpy().astype(np.int64); sl = d.column('seq_len').to_numpy().astype(np.int64)
    ts_parts.append(flat_ts[start[u] + sl])
    lab_parts.append(np.stack([d.column(c).to_numpy().astype(np.uint8) for c in TASKS], 1))
    print('block', b, len(u), round(time.time() - t0, 1), flush=True)
ts = np.concatenate(ts_parts); order = np.argsort(ts, kind='stable')
lab = np.concatenate(lab_parts)[order]; ts = ts[order]
ref_ts = np.load(f'{out}/timestamps.npy', mmap_mode='r'); ref_lb = np.load(f'{out}/labels.npy', mmap_mode='r')
assert len(ts) == len(ref_ts) and np.array_equal(ts, ref_ts), 'timestamp order differs from the dataset'
assert np.array_equal(lab[:, 0], ref_lb), 'is_click differs from labels.npy'
np.save(f'{out}/labels_mt.npy', lab)
json.dump({'tasks': TASKS, 'rows': int(len(lab)), 'positive_rate': {t: float(lab[:, i].mean()) for i, t in enumerate(TASKS)},
           'verified': 'timestamps == timestamps.npy and is_click == labels.npy', 'elapsed_s': time.time() - t0},
          open(f'{out}/meta_mt.json', 'w'), indent=1)
print('done', lab.shape, {t: round(float(lab[:, i].mean()), 4) for i, t in enumerate(TASKS)}, round(time.time() - t0, 1), flush=True)

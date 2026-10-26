"""Preprocess the raw, chronologically ordered Criteo Kaggle DAC train.txt (7 days, 45.84M rows).

Output (no shuffling, original file order preserved):
  sparse.npy  int32 [N,26]  per-field IDs; 0 = missing; IDs assigned in order of first
                            occurrence in the file (identical to streaming online assignment)
  dense.npy   float32 [N,13] log1p(max(x,0)), missing=0  (DLRM reference preprocessing)
  labels.npy  uint8 [N]
  meta.json   cardinalities (incl. the reserved 0), day boundaries (7 equal parts, as in the
              DLRM reference Kaggle split), source size/mtime.
No frequency thresholding or hashing: the full vocabulary is kept so table sizes are realistic.
"""
import json, os, sys, time
import numpy as np
import pyarrow as pa, pyarrow.csv as pcsv

src, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
t0 = time.time()
names = ['label'] + [f'I{i}' for i in range(1, 14)] + [f'C{i}' for i in range(1, 27)]
types = {'label': pa.int8()}
types.update({f'I{i}': pa.int64() for i in range(1, 14)})
types.update({f'C{i}': pa.dictionary(pa.int32(), pa.string()) for i in range(1, 27)})
tbl = pcsv.read_csv(src, read_options=pcsv.ReadOptions(column_names=names, block_size=1 << 26),
                    parse_options=pcsv.ParseOptions(delimiter='\t'),
                    convert_options=pcsv.ConvertOptions(column_types=types, null_values=[''], strings_can_be_null=True))
n = tbl.num_rows
print('read', n, 'rows', round(time.time() - t0, 1), 's', flush=True)
tbl = tbl.unify_dictionaries()
sparse = np.empty((n, 26), dtype=np.int32); cards = []
for j in range(26):
    col = tbl.column(f'C{j + 1}'); pos = 0
    for ch in col.chunks:
        idx = ch.indices.to_numpy(zero_copy_only=False)  # nulls -> arbitrary; fix below
        valid = ch.is_valid().to_numpy(zero_copy_only=False)
        sparse[pos:pos + len(ch), j] = np.where(valid, idx.astype(np.int64) + 1, 0)
        pos += len(ch)
    nd = len(col.chunks[0].dictionary) if col.num_chunks else 0
    cards.append(int(nd) + 1)
    # verify first-occurrence ordering of IDs (required for online-equivalent assignment)
    ids = sparse[:, j]; u, first = np.unique(ids, return_index=True)
    nz = u > 0
    assert len(u[nz]) == nd, (j, len(u[nz]), nd)
    assert np.all(np.diff(first[nz]) > 0), f'field {j}: IDs not in first-occurrence order'
    print('field', j, 'card', cards[-1], round(time.time() - t0, 1), flush=True)
dense = np.empty((n, 13), dtype=np.float32)
for i in range(13):
    c = tbl.column(f'I{i + 1}').to_numpy(zero_copy_only=False)
    c = np.nan_to_num(c.astype(np.float64), nan=0.0)
    dense[:, i] = np.log1p(np.maximum(c, 0)).astype(np.float32)
labels = tbl.column('label').to_numpy(zero_copy_only=False).astype(np.uint8)
np.save(f'{out}/sparse.npy', sparse); np.save(f'{out}/dense.npy', dense); np.save(f'{out}/labels.npy', labels)
days = [int(round(n * k / 7)) for k in range(8)]
st = os.stat(src)
json.dump({'source': src, 'source_bytes': st.st_size, 'source_mtime_ns': st.st_mtime_ns, 'rows': n,
           'cardinalities': cards, 'total_rows_incl_missing': int(sum(cards)), 'day_boundaries': days,
           'id_assignment': 'per-field first occurrence in file order; 0 = missing',
           'dense': 'log1p(max(x,0)), missing=0', 'order': 'original file order (chronological per Criteo readme)',
           'ctr': float(labels.mean()), 'elapsed_s': time.time() - t0}, open(f'{out}/meta.json', 'w'), indent=1)
print('done', round(time.time() - t0, 1), 's; total rows', sum(cards), flush=True)

"""Memory-bounded preprocessing (column groups) for chronological CTR datasets -> srm2 format.

criteo: raw Kaggle DAC train.txt (TSV, no header; label, I1..I13, C1..C26), file order (chronological).
avazu : reczoo Avazu_x4 train.csv; the file concatenates 8 folds each spanning all 10 days, so rows are
        stably sorted by 'hour' (chronological at hour granularity).
IDs: per field, first-occurrence order in the (chronological) row order; 0 reserved for missing (criteo).
Identical ID assignment to data/prep_criteo_kaggle.py / data/prep_avazu.py, but each pass reads only a
few columns, bounding peak memory to a few GB.
"""
import datetime, json, os, sys, time
import numpy as np, pyarrow as pa, pyarrow.csv as pcsv

kind, src, out = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(out, exist_ok=True); t0 = time.time()


def read_cols(cols, names=None, types=None, delim=','):
    ro = pcsv.ReadOptions(column_names=names, block_size=1 << 26) if names else pcsv.ReadOptions(block_size=1 << 26)
    co = pcsv.ConvertOptions(include_columns=cols, column_types=types or {}, null_values=[''], strings_can_be_null=True)
    return pcsv.read_csv(src, read_options=ro, parse_options=pcsv.ParseOptions(delimiter=delim), convert_options=co)


def encode(col, order=None, missing_zero=False):
    """First-occurrence dictionary encoding of a (possibly reordered) column -> int32 codes, cardinality."""
    if order is not None:
        col = col.take(order)
    arr = col.cast(pa.string()) if not pa.types.is_dictionary(col.type) else col
    arr = arr.combine_chunks() if hasattr(arr, 'combine_chunks') else arr
    enc = arr.dictionary_encode() if not pa.types.is_dictionary(arr.type) else arr
    idx = enc.indices.to_numpy(zero_copy_only=False)
    valid = enc.is_valid().to_numpy(zero_copy_only=False)
    nd = len(enc.dictionary)
    if missing_zero:
        codes = np.where(valid, np.nan_to_num(idx, nan=-1).astype(np.int64) + 1, 0).astype(np.int32); card = nd + 1
    else:
        codes = idx.astype(np.int32); card = nd
    return codes, card


if kind == 'criteo':
    names = ['label'] + [f'I{i}' for i in range(1, 14)] + [f'C{i}' for i in range(1, 27)]
    lab = read_cols(['label'], names, {'label': pa.int8()}, '\t').column('label').to_numpy().astype(np.uint8)
    n = len(lab); print('rows', n, round(time.time() - t0, 1), flush=True)
    sparse = np.lib.format.open_memmap(f'{out}/sparse.npy', mode='w+', dtype=np.int32, shape=(n, 26)); cards = []
    for g in range(0, 26, 4):
        cols = [f'C{i + 1}' for i in range(g, min(26, g + 4))]
        t = read_cols(cols, names, {c: pa.string() for c in cols}, '\t')
        for c in cols:
            codes, card = encode(t.column(c), missing_zero=True); j = int(c[1:]) - 1
            sparse[:, j] = codes; cards.append((j, card)); print(c, card, round(time.time() - t0, 1), flush=True)
        del t
    cards = [c for _, c in sorted(cards)]
    dense = np.lib.format.open_memmap(f'{out}/dense.npy', mode='w+', dtype=np.float32, shape=(n, 13))
    t = read_cols([f'I{i}' for i in range(1, 14)], names, {f'I{i}': pa.int64() for i in range(1, 14)}, '\t')
    for i in range(13):
        c = np.nan_to_num(t.column(f'I{i + 1}').to_numpy(zero_copy_only=False).astype(np.float64), nan=0.0)
        dense[:, i] = np.log1p(np.maximum(c, 0)).astype(np.float32)
    del t
    days = [int(round(n * k / 7)) for k in range(8)]
    meta = {'order': 'original file order (chronological per Criteo readme)', 'dense': 'log1p(max(x,0)), missing=0',
            'id_assignment': 'per-field first occurrence in file order; 0 = missing'}
elif kind == 'avazu':
    hour = read_cols(['hour'], types={'hour': pa.int64()}).column('hour').to_numpy()
    order = np.argsort(hour, kind='stable'); hour = hour[order]; n = len(hour); assert (np.diff(hour) >= 0).all()
    lab = read_cols(['click'], types={'click': pa.int8()}).column('click').take(order).to_numpy().astype(np.uint8)
    header = open(src).readline().strip().split(',')
    cols_all = [c for c in header if c not in ('id', 'click', 'hour')]
    sparse = np.lib.format.open_memmap(f'{out}/sparse.npy', mode='w+', dtype=np.int32, shape=(n, len(cols_all))); cards = []
    for g in range(0, len(cols_all), 6):
        cols = cols_all[g:g + 6]
        t = read_cols(cols, types={c: pa.string() for c in cols})
        for c in cols:
            codes, card = encode(t.column(c), order=order); sparse[:, cols_all.index(c)] = codes; cards.append(card)
            print(c, card, round(time.time() - t0, 1), flush=True)
        del t
    hh = (hour % 100).astype(np.float32); day = (hour // 100) % 100
    wd = np.array([datetime.date(2014, 10, int(d)).weekday() for d in range(21, 31)])[day - 21].astype(np.float32)
    dense = np.lib.format.open_memmap(f'{out}/dense.npy', mode='w+', dtype=np.float32, shape=(n, 2))
    dense[:, 0] = hh / 23.0; dense[:, 1] = wd / 6.0
    days = [0] + [int(np.searchsorted(day, d)) for d in range(22, 31)] + [n]
    meta = {'fields': cols_all, 'order': 'stable sort by hour of the 8 concatenated folds (chronological at hour granularity)',
            'dense': 'hour-of-day/23, weekday/6'}
else:
    raise ValueError(kind)
np.save(f'{out}/labels.npy', lab); sparse.flush(); dense.flush()
meta.update({'source': src, 'source_bytes': os.path.getsize(src), 'rows': int(n), 'cardinalities': cards,
             'total_rows': int(sum(cards)), 'day_boundaries': days, 'ctr': float(lab.mean()),
             'elapsed_s': time.time() - t0})
json.dump(meta, open(f'{out}/meta.json', 'w'), indent=1)
print('done', n, sum(cards), round(time.time() - t0, 1), flush=True)

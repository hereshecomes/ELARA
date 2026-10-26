"""Avazu (reczoo Avazu_x4 train.csv) -> srm2 format, re-sorted chronologically.
The x4 train file concatenates 8 random folds, each spanning all 10 days in hour order; a stable sort by
'hour' restores global chronological order (hour granularity).
sparse.npy int32 [N,21]: categorical fields C1..C21 incl. site/app/device IDs; IDs by first occurrence; no filtering.
dense.npy float32 [N,2]: hour-of-day/23 and weekday/6 (derived from the 'hour' field YYMMDDHH).
labels.npy uint8; meta.json with cardinalities and day boundaries (by calendar day of 'hour')."""
import json, os, sys, time
import numpy as np, pyarrow.csv as pcsv
src, out = sys.argv[1], sys.argv[2]; os.makedirs(out, exist_ok=True); t0 = time.time()
tbl = pcsv.read_csv(src); n = tbl.num_rows
order = np.argsort(tbl.column('hour').to_numpy(), kind='stable'); tbl = tbl.take(order)
hour = tbl.column('hour').to_numpy(); assert (np.diff(hour) >= 0).all()
cols = [c for c in tbl.column_names if c not in ('id', 'click', 'hour')]
sparse = np.empty((n, len(cols)), np.int32); cards = []
for j, c in enumerate(cols):
    arr = tbl.column(c).cast('string').combine_chunks().dictionary_encode()
    sparse[:, j] = arr.indices.to_numpy(zero_copy_only=False); cards.append(len(arr.dictionary))
hh = (hour % 100).astype(np.float32); day = (hour // 100) % 100
import datetime
wd = np.array([datetime.date(2014, 10, int(d)).weekday() for d in range(21, 31)])[day - 21].astype(np.float32)
dense = np.stack([hh / 23.0, wd / 6.0], 1).astype(np.float32)
labels = tbl.column('click').to_numpy().astype(np.uint8)
np.save(f'{out}/sparse.npy', sparse); np.save(f'{out}/dense.npy', dense); np.save(f'{out}/labels.npy', labels)
bounds = [0] + [int(np.searchsorted(day, d)) for d in range(22, 31)] + [n]
json.dump({'source': src, 'rows': n, 'fields': cols, 'cardinalities': cards, 'total_rows': int(sum(cards)),
           'day_boundaries': bounds, 'order': 'stable sort by hour of the 8 concatenated folds (chronological at hour granularity)', 'ctr': float(labels.mean()), 'secs': time.time() - t0},
          open(f'{out}/meta.json', 'w'), indent=1)
print('done', n, sum(cards), bounds, round(time.time() - t0, 1))

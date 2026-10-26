"""Bitwise comparison of evaluation predictions written by srm2/train.py (--save-pred 1 -> <out>.pred.npy).

All exact methods (ELARA, K=0, static+TTL, FixedHot, LRU, LFU, TTL, window, SparseRM-K3, and their host-tier
variants) run on the same workload, seed and model train the same sparse trajectory, so their held-out
predictions must be identical bit for bit.

usage: python tools/compare_preds.py REFERENCE.pred.npy OTHER.pred.npy [OTHER.pred.npy ...]
Exit code 0 if every file is identical to the reference, 1 otherwise.
"""
import sys

import numpy as np


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    ref_path, others = sys.argv[1], sys.argv[2:]
    ref = np.load(ref_path)
    ok = True
    for path in others:
        x = np.load(path)
        same_layout = x.shape == ref.shape and x.dtype == ref.dtype
        identical = same_layout and np.array_equal(x.view(np.uint8), ref.view(np.uint8))
        diff = float(np.abs(x.astype(np.float64) - ref.astype(np.float64)).max()) if x.shape == ref.shape else float('nan')
        print(f"{'IDENTICAL' if identical else 'DIFFERENT'}  {path}  (max |diff| = {diff:.3g})")
        ok &= identical
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()

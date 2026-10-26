# ELARA: Exact Optimizer-State Replay for Tiered Recommendation Training

Anonymous code for the WWW '27 submission. ELARA (Exact Lazy Auxiliary-State Replay and Allocation) keeps
untouched embedding rows implicit, stores short update histories as their original gradients, and materializes
the full optimizer state only after a bounded, optimizer- and tier-aware history threshold `K`. A GPU-resident
state directory, physical-ID-addressed host arenas and a pipelined executor make every representation exact:
all exact methods in this repository train bitwise-identical models.

The package is called `srm2`. It contains ELARA together with every same-engine baseline of the paper, so that
all exact policies share the data path, dense model, sparse update arithmetic and pipeline.

## Repository layout

| Path | Contents | Paper |
|---|---|---|
| `srm2/elara.py` | ELARA planner (GPU state directory: row codes `I`/`L_c`/`M`, log ids and steps, sync-free allocators, residency, resident histories, memory-pressure gate), executor and host arenas (`ArenaStore`) | §3.1–3.4, §4.1 |
| `srm2/csrc/elara_dir.cpp` | Default C++/ATen implementation of the directory updates (GIL released) | §4.1 |
| `srm2/elara_kernels.py` | Triton implementation of the same directory updates (ablation "Triton directory") | Table 3 |
| `srm2/elara_ctl.py` | Optional critical-path-aware threshold controller (ablation "Adaptive controller"; off by default) | Table 3 |
| `srm2/gpu_ops.py` | Triton kernels: counter-hash row initialization and exact replay of logged gradients | §3.1 |
| `srm2/ops.py`, `srm2/csrc/srm2_cpu.cpp` | Row-sparse Adam on GPU (Triton) and CPU (C++), bitwise identical; host gather/scatter, page pre-faulting | §3.4, §4.1 |
| `srm2/executor.py` | Pipelined executor (inbound/outbound host workers, copy streams, ordering barriers) and the passive-tier link model `LinkModel` | §3.4, §4.2 |
| `srm2/planner.py` | Placement planner of the eager baselines (FixedHot, LRU, LFU, TTL, static+TTL, window), the SparseRM-K3 predecessor and the inexact admission filter | §5.1 |
| `srm2/store.py` | Capacity-tier stores of the baselines (preallocated eager table, on-demand compact store) with an enforced host budget | §5.1 |
| `srm2/models.py` | DLRM, DeepFM, DCN-v2, DLRMv2 (DCN-v2 interaction), shared-bottom multi-task DLRM, MMoE | §5.1, Table 6 |
| `srm2/train.py` | Training driver for all exact methods: chronological loader, planner thread, CUDA-graph dense step, evaluation, result record | §5 |
| `srm2/train_fbgemm.py`, `srm2/train_dynemb.py` | External libraries: FBGEMM TBE (UVM caching / HBM) and NVIDIA DynamicEmb | Table 5 |
| `srm2/profsum.py` | Summary of a `torch.profiler` window (`--prof-steps`) | Table 9 |
| `data/` | Chronological preprocessing of Criteo, Avazu and KuaiRand | §5.1 |
| `tests/` | Exactness tests (replay = eager Adam, CPU = GPU Adam, fused store = reference store) | §3.4 |
| `scripts/run_experiments.sh` | Command lines of all experiments | §5, App. 8 |
| `tools/compare_preds.py` | Bitwise comparison of saved predictions across methods | §3.4 |

## Methods

`--method` of `srm2/train.py` selects the policy. Passive-tier runs add `--link-gbps <b> --link-lat-us 2`; host-DRAM
runs (`-S` variants) use no link model.

| Paper | `--method` and flags |
|---|---|
| **ELARA** (passive tier, `K=3`) | `elara --K auto --elara-dir-impl cpp --elara-evict-logs 2 --prefault-ahead-mb 512` |
| **ELARA-S** (host DRAM, `K=2`) | `elara_S` with the same flags |
| K=0 (on-demand eager, same engine) | `elara_k0 --elara-dir-impl cpp --prefault-ahead-mb 512` (host: `elara_k0_S`) |
| static+TTL | `fixed_ttl_T` (host: `fixed_ttl_S`) |
| FixedHot | `fixed_T` (host: `fixed_S`) |
| LRU / LFU / TTL / window | `lru_T` / `lfu_T` / `ttl_T` / `window_T` |
| SparseRM-K3 (predecessor, host-side log index) | `sparserm --K 3` |
| Filter t2 / t3 (inexact) | `fixed_ttl_lazy_T --K 0 --admit-threshold 2` / `3` |
| Hash (inexact) | `fixed_ttl_T --hash-cap <rows>` |
| Reduced dimension (inexact) | `fixed_ttl_T --dim <d>` |

Main ELARA options: `--K` materialization threshold (`auto` = 3 on the passive tier, 2 on host DRAM, §3.2),
`--elara-evict-logs` history retention across eviction (0 off, 1 always, 2 gated: switched on once the logical
store reaches `--elara-evl-gate` = 75 % of `--host-budget-gb`, Eq. 12), `--L` lookahead (8), `--depth` pipeline
depth (3), `--budget-rows` HBM pool rows, `--host-budget-gb` enforced capacity-tier budget (charged on faulted
arena pages), `--elara-dir-impl {cpp,triton,ref}` directory implementation.

## Setup

```bash
pip install -r requirements.txt
```

Tested with Python 3.12, PyTorch 2.11 (CUDA 12.8), Triton 3.6 and GCC 11.4 on Linux. The two C++ extensions are
built automatically on first use (`srm2/build*`; override with `SRM2_BUILD=<dir>`). The paper's numbers come from
one NVIDIA RTX 5090 (32 GB) with an Intel Xeon Platinum 8470Q host, 8 host-worker threads and 16 intra-op threads.
The eager baselines need host memory for the full optimizer state (for example 48.3 GiB for Criteo d=128 with Adam).

Exactness tests:

```bash
python tests/test_replay.py            # replay(init, log) == eager Adam on GPU and CPU, bitwise (GPU)
python tests/test_ops.py ops.json      # CPU Adam == GPU Adam, bitwise, plus primitive throughput (GPU)
python tests/test_store_fused.py       # fused C++ compact store == reference store (CPU only)
```

## Data

All datasets keep the chronological sample order, assign IDs by first occurrence and keep the full vocabulary
(no hashing or frequency filtering). Each output directory holds `sparse.npy`, `dense.npy`, `labels.npy` and
`meta.json`.

| Workload | Source | Command |
|---|---|---|
| Criteo (Kaggle Display Advertising Challenge, 7 days) | `train.txt` | `python data/prep_lowmem.py criteo train.txt data/criteo_kaggle` (or `data/prep_criteo_kaggle.py train.txt data/criteo_kaggle` with more memory) |
| Avazu | Avazu_x4 `train.csv` (BARS / reczoo) | `python data/prep_avazu.py train.csv data/avazu` (or `data/prep_lowmem.py avazu ...`) |
| KuaiRand-27K | parquet release with `train/{data,user_info,item_info}/part-<block>.parquet` | `python data/prep_kuairand.py <src> data/kuairand8 10 00000 00001 00002 00003 00004 00005 00006 00007` |
| KuaiRand multi-task labels | same source | `python data/prep_kuairand_mt.py <src> data/kuairand8 00000 00001 00002 00003 00004 00005 00006 00007` |

Criteo trains on the first six days and evaluates on the first 1M samples of the seventh; Avazu and KuaiRand
evaluate on 1M samples starting at the first day boundary after the training pass (`--eval-start`). KuaiRand uses a 10-video history bag that shares the video table.

## Running the experiments

```bash
DATA=data OUT=results bash scripts/run_experiments.sh smoke          # 40-step end-to-end check + bitwise comparison
DATA=data OUT=results bash scripts/run_experiments.sh main 42        # one seed of the main matrix
```

| Experiment | Paper |
|---|---|
| `main` | Table 1, Fig. 3 (upper), Table 7 (passive tier) |
| `host` | Table 7 (host DRAM) |
| `ablation` | Table 3, Table 9 (SparseRM column) |
| `capacity` | Table 2, Fig. 3 (lower), Fig. 4 |
| `policy` | §5.4 retention-gate comparison, App. 8.3 host-DRAM `K=2` vs. `K=3` |
| `quality` | Table 8, Fig. 7 |
| `models` | Table 6, Table 10 (multi-task) |
| `sensitivity` | Fig. 5 |
| `profile` | Table 9 (phase breakdown; `torch.profiler` window over steps 250–260) |
| `libraries` | Table 5 (needs FBGEMM-GPU; DynamicEmb via `DYNEMB_PY=<python of its environment>`) |

The main matrix uses seeds 42–46, the other experiments 42–44. Run jobs one at a time on an otherwise idle GPU.
A single job can also be started directly, for example ELARA on Criteo d=128 at 12.5 Gb/s:

```bash
python srm2/train.py --data data/criteo_kaggle --dim 128 --batch 65536 --steps 599 --start 0 --profile day0 \
  --eval-samples 1000000 --budget-rows 337626 --carry-reserve 131072 --threads 8 --depth 3 --L 8 \
  --lr 1e-3 --emb-lr 3e-3 --method elara --K auto --elara-dir-impl cpp --elara-evict-logs 2 \
  --prefault-ahead-mb 512 --link-gbps 12.5 --link-lat-us 2 --seed 42 --out results/criteo_d128.elara.json
```

### Result record

Each job writes one JSON file (`--out`) and, with `--save-pred 1` (default), its held-out predictions
(`<out>.pred.npy`).

| Field | Meaning |
|---|---|
| `steady.host_step_ms` | Steady-state step time: mean over steps `[n/4, 3n/4)` (Tables 1, 3, 6, 7, Figs. 3, 5) |
| `steady.train_ms_per_step` | Synchronized training wall time / steps (the cross-library metric of Table 5) |
| `steady.host_*_ms` | Phase breakdown of the host step (Table 9) |
| `store_footprint` | Logical capacity-tier state (`store_bytes`), materialized / log payload, charged pages (Table 2, Fig. 4) |
| `link.bytes` | Inbound / outbound bytes over the capacity tier (divide by `steps` for MB/step, Table 7) |
| `eval` | Held-out AUC and logloss (per task for multi-task models) |
| `dir_check` | Directory consistency: live log entries = valid directory entries, live + free = arena top |
| `state_trace`, `footprint_trace` | Row-state histogram (`--trace-states N`) and store occupancy over training (Fig. 4) |
| `status` | `complete`, or `host_budget_exceeded_*` with the failing step for budgeted runs |

### Checking exactness

Exact methods trained on the same workload, seed and model produce the same predictions bit for bit:

```bash
python tools/compare_preds.py results/A.pred.npy results/B.pred.npy [...]
```

## Capacity-tier models

* **Passive tier** (`--link-gbps`): the capacity tier stores row payloads and the GPU runs all optimizer updates.
  Each link direction is an independent FIFO queue that charges a `Q`-byte transfer `Q/b + 2 µs · ceil(Q / 128 KiB)`
  at rate `b` (`LinkModel` in `srm2/executor.py`); payloads are published to the training step only after the
  modeled completion time.
* **Host DRAM** (`-S` methods, no link model): materialized rows are updated by the outbound host workers with the
  CPU Adam kernel, so only weights and gradients cross PCIe.

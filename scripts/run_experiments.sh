#!/usr/bin/env bash
# Command lines behind the paper's experiments. Every call of `run` is one training job (one JSON result).
#
#   DATA=/path/to/datasets OUT=results bash scripts/run_experiments.sh <experiment> [seed]
#   DRY_RUN=1 bash scripts/run_experiments.sh <experiment>     # print the commands only
#
# experiments: smoke | main | host | ablation | capacity | policy | quality | models | sensitivity | profile | libraries
# The paper uses seeds 42-46 for `main` and 42-44 for the other experiments. Jobs are meant to run one at a
# time on an otherwise idle GPU (timings are reported from such runs).
set -euo pipefail
cd "$(dirname "$0")/.."

EXP=${1:?usage: run_experiments.sh <experiment> [seed]}
SEED=${2:-42}
DATA=${DATA:-data}
OUT=${OUT:-results}
PY=${PY:-python3}

WORKLOADS="criteo_d64 criteo_d128 avazu_d128 kuairand_d128"
BANDWIDTHS="12.5 25 50 100"     # passive-tier link rate (Gb/s)

# workload flags shared by every driver: dataset, embedding dimension, one chronological pass, HBM pool rows
wl() {
  case $1 in
    criteo_d64)    echo "--data $DATA/criteo_kaggle --dim 64 --steps 599 --budget-rows 337626" ;;
    criteo_d128)   echo "--data $DATA/criteo_kaggle --dim 128 --steps 599 --budget-rows 337626" ;;
    avazu_d128)    echo "--data $DATA/avazu --dim 128 --steps 395 --eval-start 25901338 --budget-rows 167468" ;;
    kuairand_d128) echo "--data $DATA/kuairand8 --dim 128 --steps 897 --eval-start 58835778 --budget-rows 243347" ;;
    *) echo "unknown workload $1" >&2; exit 2 ;;
  esac
}
# engine-only workload flags: static hot-set profiling window and carry reserve of the HBM pool
eng() {
  case $1 in
    criteo_d64|criteo_d128) echo "--profile day0 --carry-reserve 131072" ;;
    avazu_d128)             echo "--profile day0 --carry-reserve 65536" ;;
    kuairand_d128)          echo "--profile day1 --carry-reserve 131072" ;;
  esac
}
COMMON="--batch 65536 --start 0 --eval-samples 1000000 --lr 1e-3 --emb-lr 3e-3"
ENGINE="--threads 8 --depth 3 --L 8"
link() { echo "--link-gbps $1 --link-lat-us 2"; }

# methods (paper name -> flags)
ELARA="--method elara --K auto --elara-dir-impl cpp --elara-evict-logs 2 --prefault-ahead-mb 512"
ELARA_S="--method elara_S --K auto --elara-dir-impl cpp --elara-evict-logs 2 --prefault-ahead-mb 512"
K0="--method elara_k0 --elara-dir-impl cpp --prefault-ahead-mb 512"
K0_S="--method elara_k0_S --elara-dir-impl cpp --prefault-ahead-mb 512"
STATIC_TTL="--method fixed_ttl_T"
EAGER_PASSIVE="fixed_T lru_T lfu_T ttl_T fixed_ttl_T window_T"   # FixedHot LRU LFU TTL static+TTL window

ex() {    # execute a command, or only print it with DRY_RUN=1
  if [ -n "${DRY_RUN:-}" ]; then echo "$*"; else mkdir -p "$OUT"; "$@"; fi
}
run() {   # run <name> <train.py flags...>
  local name=$1; shift
  [ -n "${DRY_RUN:-}" ] || echo ">> $name (seed $SEED)"
  ex $PY -u srm2/train.py "$@" --seed "$SEED" --out "$OUT/$name.s$SEED.json"
}
base() { echo "$(wl "$1") $(eng "$1") $COMMON $ENGINE"; }

case $EXP in
  smoke)   # a short end-to-end check (Criteo d64, 40 steps, 12.5 Gb/s): ELARA vs. its matched eager control
    for m in ELARA STATIC_TTL; do
      run "smoke.$m" $(base criteo_d64) ${!m} $(link 12.5) --steps 40 --eval-samples 200000
    done
    ex $PY tools/compare_preds.py "$OUT/smoke.ELARA.s$SEED.pred.npy" "$OUT/smoke.STATIC_TTL.s$SEED.pred.npy"
    ;;

  main)    # Table 1, Fig. 3 (upper), Table 7 (passive tier): all exact methods x workloads x bandwidths
    for w in $WORKLOADS; do for bw in $BANDWIDTHS; do
      run "$w.g$bw.ELARA" $(base $w) $ELARA $(link $bw)
      run "$w.g$bw.K0" $(base $w) $K0 $(link $bw)
      for m in $EAGER_PASSIVE; do run "$w.g$bw.$m" $(base $w) --method $m $(link $bw); done
    done; done
    ;;

  host)    # Table 7 (real host DRAM; materialized rows updated by CPU Adam near memory)
    for w in $WORKLOADS; do
      run "$w.host.ELARA-S" $(base $w) $ELARA_S
      run "$w.host.K0-S" $(base $w) $K0_S
      run "$w.host.FixedHot-S" $(base $w) --method fixed_S
      run "$w.host.static+TTL-S" $(base $w) --method fixed_ttl_S
    done
    ;;

  ablation)   # Table 3 (and the SparseRM column of Table 9)
    for w in $WORKLOADS; do for bw in $BANDWIDTHS; do
      b="$(base $w) $(link $bw)"; p="$w.g$bw"
      run "$p.full" $b $ELARA
      run "$p.K0" $b $K0
      run "$p.K1" $b --method elara --K 1 --elara-dir-impl cpp --prefault-ahead-mb 512
      run "$p.K2" $b --method elara --K 2 --elara-dir-impl cpp --prefault-ahead-mb 512
      run "$p.K4" $b --method elara --K 4 --elara-kmax 4 --elara-dir-impl cpp --prefault-ahead-mb 512
      run "$p.no-lookahead" $b --method elara_carry --K auto --elara-dir-impl cpp --prefault-ahead-mb 512
      run "$p.no-allocation-ahead" $b --method elara --K auto --elara-dir-impl cpp
      run "$p.reference-directory" $b --method elara --K auto --elara-dir-impl ref --prefault-ahead-mb 512
      run "$p.triton-directory" $b --method elara --K auto --elara-dir-impl triton --prefault-ahead-mb 512
      run "$p.adaptive-controller" $b $ELARA --elara-adaptive 1
      run "$p.retention-always" $b --method elara --K auto --elara-dir-impl cpp --elara-evict-logs 1 --prefault-ahead-mb 512
      run "$p.retention-off" $b --method elara --K auto --elara-dir-impl cpp --elara-evict-logs 0 --prefault-ahead-mb 512
      run "$p.SparseRM-K3" $b --method sparserm --K 3
    done; done
    ;;

  capacity)   # Table 2 boundaries, Fig. 3 (lower), Fig. 4: enforced capacity-tier budgets (GiB) at 25 Gb/s
    declare -A BUDGETS=(
      [criteo_d64]="6 8 9 9.4 10 12 16 24 28"
      [criteo_d128]="12 16 18 19 20 24 32 48 56"
      [avazu_d128]="4 5 5.5 6 8 10 12 16"
      [kuairand_d128]="8 9 9.4 9.8 10.5 11 12 16 18")
    for w in $WORKLOADS; do for B in ${BUDGETS[$w]}; do
      for m in ELARA K0 STATIC_TTL; do run "$w.budget$B.$m" $(base $w) ${!m} $(link 25) --host-budget-gb $B; done
    done; done
    # row-state trajectories of Fig. 4 (state histogram every 10 steps)
    for w in $WORKLOADS; do run "$w.states" $(base $w) $ELARA $(link 25) --trace-states 10; done
    for B in 10.5 9.8; do run "kuairand_d128.states.budget$B" $(base kuairand_d128) $ELARA $(link 25) --trace-states 10 --host-budget-gb $B; done
    ;;

  policy)     # Sec. 5.4 gate comparison: history retention off / gated / always near the capacity boundary (25 Gb/s),
              # and App. 8.3: host-DRAM ELARA-S with K=2 vs. K=3
    declare -A GATE_BUDGETS=([criteo_d64]="8.6 9 9.4 10" [kuairand_d128]="9 9.4 9.8 10.5")
    for w in criteo_d64 kuairand_d128; do for B in ${GATE_BUDGETS[$w]}; do
      for r in "off 0" "gated 2" "always 1"; do
        set -- $r; run "$w.budget$B.retention-$1" $(base $w) $ELARA --elara-evict-logs $2 $(link 25) --host-budget-gb $B
      done
    done; done
    for w in $WORKLOADS; do for k in 2 3; do run "$w.host.ELARA-S-K$k" $(base $w) $ELARA_S --K $k; done; done
    ;;

  quality)    # Table 8, Fig. 7: ELARA vs. inexact budget-fitting alternatives under one store budget, 25 Gb/s
    q() {  # q <workload> <budget GiB> <dim-low> <dim-high> <hash capacity>
      local w=$1 B=$2; local b="$(base $w) $(link 25) --host-budget-gb $B"
      run "$w.q.ELARA" $b $ELARA
      run "$w.q.eager-dim$3" $b $STATIC_TTL --dim $3
      run "$w.q.eager-dim$4" $b $STATIC_TTL --dim $4
      run "$w.q.filter-t2" $b --method fixed_ttl_lazy_T --K 0 --admit-threshold 2
      run "$w.q.filter-t3" $b --method fixed_ttl_lazy_T --K 0 --admit-threshold 3
      run "$w.q.hash-cap$5" $b $STATIC_TTL --hash-cap $5
    }
    q criteo_d64 10 16 24 2522653
    q criteo_d128 19 32 40 2361871
    q avazu_d128 6 48 56 1913287
    q kuairand_d128 10.5 56 64 6744378
    ;;

  models)     # Table 6 and Table 10 (multi-task): other dense backbones
    for w in criteo_d64 avazu_d128 kuairand_d128; do for bw in 12.5 100; do
      for spec in "deepfm none" "dcnv2 bf16" "dlrmv2 bf16"; do
        set -- $spec; mf="--model $1 --amp $2"
        for m in ELARA K0 STATIC_TTL; do run "$w.g$bw.$1.$m" $(base $w) ${!m} $(link $bw) $mf; done
      done
    done; done
    MT="--tasks is_click,long_view,is_like"
    for model in dlrm_mt mmoe; do
      for bw in 12.5 100; do
        for m in ELARA K0 STATIC_TTL; do run "kuairand_d128.g$bw.$model.$m" $(base kuairand_d128) ${!m} $(link $bw) --model $model $MT; done
      done
      for m in ELARA_S K0_S; do run "kuairand_d128.host.$model.$m" $(base kuairand_d128) ${!m} --model $model $MT; done
      run "kuairand_d128.host.$model.FixedHot-S" $(base kuairand_d128) --method fixed_S --model $model $MT
    done
    ;;

  sensitivity)   # Fig. 5: Criteo d64; one axis varied at a time (25 Gb/s unless noted)
    b="$(base criteo_d64)"
    for m in ELARA K0 STATIC_TTL; do
      for r in 168813 337626 675252 1350504; do run "sens.pool$r.$m" $b ${!m} $(link 25) --budget-rows $r; done
      for bs in "16384 2396" "32768 1198" "65536 599" "131072 299"; do
        set -- $bs; run "sens.batch$1.$m" $b ${!m} $(link 25) --batch $1 --steps $2
      done
      for d in 32 64 128 256; do run "sens.dim$d.$m" $b ${!m} $(link 25) --dim $d; done
      for L in 2 4 8 16; do run "sens.L$L.$m" $b ${!m} $(link 25) --L $L; done
      for dp in 2 3 4; do run "sens.depth$dp.$m" $b ${!m} $(link 25) --depth $dp; done   # depth 1 is unsupported
      for bw in $BANDWIDTHS; do run "sens.bf16.g$bw.$m" $b ${!m} $(link $bw) --amp bf16; done
    done
    for bw in 25 100; do run "sens.normal-priority.g$bw.ELARA" $b $ELARA $(link $bw) --side-prio 0; done   # planner/loader streams at normal priority
    ;;

  profile)    # Table 9: host-step phase breakdown, with a torch.profiler window over steps [250, 260)
    for w in $WORKLOADS; do for bw in 12.5 100; do
      run "$w.g$bw.prof.ELARA" $(base $w) $ELARA $(link $bw) --prof-steps 250:260
      run "$w.g$bw.prof.static+TTL" $(base $w) $STATIC_TTL $(link $bw) --prof-steps 250:260
      run "$w.g$bw.prof.SparseRM-K3" $(base $w) --method sparserm --K 3 $(link $bw) --prof-steps 250:260
    done; done
    ;;

  libraries)  # Table 5: FBGEMM TBE (UVM caching / all-HBM) and NVIDIA DynamicEmb on real host memory
    for w in $WORKLOADS; do
      lb="$(wl $w) $COMMON --seed $SEED"
      for algo in lru lfu; do for pf in 0 1; do
        ex $PY -u srm2/train_fbgemm.py $lb --location caching --cache-algo $algo --prefetch $pf --out "$OUT/$w.fbgemm-uvm-$algo-pf$pf.s$SEED.json"
      done; done
      ex $PY -u srm2/train_fbgemm.py $lb --location device --out "$OUT/$w.fbgemm-hbm.s$SEED.json"
      # DynamicEmb lives in its own environment: set DYNEMB_PY to its python
      ex ${DYNEMB_PY:-$PY} -u srm2/train_dynemb.py $lb --location hybrid --out "$OUT/$w.dynemb-hybrid.s$SEED.json"
      ex ${DYNEMB_PY:-$PY} -u srm2/train_dynemb.py $lb --location device --out "$OUT/$w.dynemb-hbm.s$SEED.json"
    done
    ;;

  *) echo "unknown experiment: $EXP" >&2; exit 2 ;;
esac

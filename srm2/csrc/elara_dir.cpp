// ELARA state-directory maintenance in C++ (ATen on CUDA tensors; no Python dispatch, GIL released).
// Line-for-line port of the validated reference torch path in srm2/elara.py (ELARAPlanner, fused_dir=False):
// same operations, same order, hence identical directory states and allocations.
#include <torch/extension.h>
#include <vector>
using at::Tensor; namespace F = torch::indexing;

// state vector layout (all CUDA): 0 code i8[R] 1 loc i32[R+1] 2 ent i32[(R+1)K] 3 lstep i32[Lcap+1] 4 lfree i32[Lcap+1]
// 5 n_lfree i64[1] 6 log_top i64[1] 7 mat_top i64[1] 8 live i64[1] 9 adm_c0 i8[C] 10 rj i8[C] 11 pstep i32[Cdyn+1,K]
// 12 kthr i8[T] 13 toff i64[T] 14 slot_of i32[R] 15 slot_row i64[C] 16 rok i8[C]
struct P { int64_t R, K, Lcap, n0, Cdyn, MAT, trash; bool evl; };

static Tensor alloc_logs(std::vector<Tensor>& st, const Tensor& valid) {
  Tensor v = valid.to(at::kLong); Tensor rank = at::cumsum(v, 0) - 1; Tensor& nf = st[5];
  Tensor from_free = rank < nf;
  Tensor fid = st[4].index({at::clamp_min(nf - 1 - rank, 0)}).to(at::kLong);
  Tensor ids = at::where(from_free, fid, st[6] + rank - nf);
  Tensor k = v.sum().reshape({1}); Tensor used = at::minimum(k, nf);
  st[6].add_(k - used); st[5].sub_(used); st[8].add_(k);
  return ids;
}

static void free_logs(std::vector<Tensor>& st, const P& p, const Tensor& rows, const Tensor& cnt) {
  if (rows.numel() == 0) return;
  Tensor kk = at::arange(p.K, rows.options());
  Tensor pos = rows.unsqueeze(1) * p.K + kk.unsqueeze(0); Tensor e = st[2].index({pos});
  Tensor valid = (kk.unsqueeze(0) < cnt.to(at::kLong).unsqueeze(1)) & (e >= 0);
  Tensor vf = valid.reshape({-1}); Tensor rank = at::cumsum(vf.to(at::kLong), 0) - 1;
  Tensor dst = at::where(vf, st[5] + rank, at::full_like(rank, p.Lcap));
  st[4].scatter_(0, dst, e.reshape({-1}));
  Tensor k = vf.to(at::kLong).sum().reshape({1}); st[5].add_(k); st[8].sub_(k);
  Tensor cl = at::where(valid, pos, at::full_like(pos, p.R * p.K)).reshape({-1});
  st[2].scatter_(0, cl, at::full({cl.numel()}, -1, st[2].options()));
}

static Tensor alloc_mat(std::vector<Tensor>& st, const Tensor& rows, const c10::optional<Tensor>& mask) {
  Tensor cur = st[1].index({rows}).to(at::kLong); Tensor nw = cur < 0;
  if (mask.has_value()) nw = nw & *mask;
  Tensor rank = at::cumsum(nw.to(at::kLong), 0) - 1;
  Tensor sl = at::where(nw, st[7] + rank, cur);
  if (rows.numel()) st[1].index_put_({rows}, sl.to(at::kInt));
  st[7].add_(nw.to(at::kLong).sum().reshape({1}));
  return sl;
}

static Tensor table_of(std::vector<Tensor>& st, const Tensor& rows) {
  return at::searchsorted(st[13], rows, false, /*right=*/true) - 1;
}

// Call A: replay inputs, resident-history recording of hits, eviction representation.
std::vector<Tensor> elara_dir_a(std::vector<Tensor> st, std::vector<int64_t> pv, Tensor hs, Tensor L_row, Tensor L_cnt,
                               Tensor ev_slot, int64_t s) {
  P p{pv[0], pv[1], pv[2], pv[3], pv[4], pv[5], pv[6], pv[7] != 0};
  auto lo = hs.options().dtype(at::kLong); const int64_t K = p.K;
  Tensor kk = at::arange(K, lo);
  // replay inputs
  Tensor Lent = st[2].index({L_row.unsqueeze(1) * K + kk.unsqueeze(0)});
  Tensor L_steps = at::where(Lent >= 0, st[3].index({at::clamp_min(Lent, 0).to(at::kLong)}), at::zeros_like(Lent));
  Tensor L_ids = at::where(kk.unsqueeze(0) < L_cnt.unsqueeze(1), Lent, at::full_like(Lent, -1));
  // hits
  Tensor dynh = hs >= p.n0;
  Tensor c0h = st[9].index({hs}).to(at::kLong), jh = st[10].index({hs}).to(at::kLong);
  Tensor okh = st[16].index({hs}) != 0;
  Tensor rec = p.evl ? (dynh & okh & ((c0h + jh) < K)) : at::zeros_like(dynh);
  Tensor hrec_dyn = at::where(rec, hs - p.n0, at::full_like(hs, p.Cdyn));
  Tensor hrec_j = at::where(rec, jh, at::zeros_like(jh));
  if (hs.numel()) {
    st[11].index_put_({hrec_dyn, hrec_j}, at::full_like(hrec_j, s, st[11].options()));
    st[10].index_put_({hs}, (jh + rec.to(at::kLong)).to(at::kChar));
    st[16].index_put_({hs}, (okh & ~(dynh & ~rec)).to(at::kChar));
  }
  // evictions
  Tensor ev_row = st[15].index({ev_slot}); const int64_t nev = ev_slot.numel();
  Tensor ev_c0 = st[9].index({ev_slot}).to(at::kLong), ev_j = st[10].index({ev_slot}).to(at::kLong);
  Tensor rep = (p.evl && nev) ? ((st[16].index({ev_slot}) != 0) & ((ev_c0 + ev_j) <= st[12].index({table_of(st, ev_row)}).to(at::kLong)))
                              : at::zeros({nev}, lo.dtype(at::kBool));
  Tensor ev_dyn = ev_slot - p.n0;
  if (nev) { st[14].index_put_({ev_row}, -1); st[15].index_put_({ev_slot}, -1); }
  Tensor kg = kk.unsqueeze(0).expand({nev, K});
  Tensor evl_valid = rep.unsqueeze(1) & (kg < ev_j.unsqueeze(1));
  Tensor evl_ids = alloc_logs(st, evl_valid.reshape({-1})).reshape({nev, K});
  evl_ids = at::where(evl_valid, evl_ids, at::full_like(evl_ids, -1));
  if (nev) {
    Tensor pos = at::where(evl_valid, ev_row.unsqueeze(1) * K + at::clamp_min(ev_c0, 0).unsqueeze(1) + kg, at::full_like(kg, p.R * K));
    st[2].scatter_(0, pos.reshape({-1}), evl_ids.reshape({-1}).to(at::kInt));
    Tensor dst = at::where(evl_valid, evl_ids, at::full_like(evl_ids, p.Lcap)).reshape({-1});
    st[3].scatter_(0, dst, st[11].index({ev_dyn}).reshape({-1}));
  }
  Tensor ev_slot_mat = alloc_mat(st, ev_row, ~rep);
  Tensor evf_slots = at::where(rep, at::full_like(ev_slot_mat, p.trash), ev_slot_mat);
  if (nev) {
    free_logs(st, p, ev_row, at::where(rep, at::zeros_like(ev_c0), at::clamp_min(ev_c0, 0)));
    st[0].index_put_({ev_row}, at::where(rep, at::clamp_max(ev_c0 + ev_j, 127).to(at::kChar), at::full_like(ev_row, p.MAT).to(at::kChar)));
  }
  return {L_steps, L_ids, hrec_dyn, hrec_j, ev_row, rep, evl_ids, evf_slots, ev_dyn};
}

// Call B: appends (stay L) and compactions (to M).
std::vector<Tensor> elara_dir_b(std::vector<Tensor> st, std::vector<int64_t> pv, Tensor a_rows, Tensor a_k, Tensor c_rows,
                               Tensor c_cnt, int64_t s) {
  P p{pv[0], pv[1], pv[2], pv[3], pv[4], pv[5], pv[6], pv[7] != 0}; const int64_t K = p.K;
  Tensor a_ids = alloc_logs(st, at::ones({a_rows.numel()}, a_rows.options().dtype(at::kBool)));
  if (a_rows.numel()) {
    st[2].index_put_({a_rows * K + a_k}, a_ids.to(at::kInt)); st[3].index_put_({a_ids}, (int)s);
    st[0].index_put_({a_rows}, (a_k + 1).to(at::kChar));
  }
  Tensor c_slots = alloc_mat(st, c_rows, c10::nullopt);
  if (c_rows.numel()) { free_logs(st, p, c_rows, c_cnt); st[0].index_put_({c_rows}, (int)p.MAT); }
  return {a_ids, c_slots};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("elara_dir_a", &elara_dir_a, py::call_guard<py::gil_scoped_release>());
  m.def("elara_dir_b", &elara_dir_b, py::call_guard<py::gil_scoped_release>());
}

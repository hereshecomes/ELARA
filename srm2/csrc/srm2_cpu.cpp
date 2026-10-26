// srm2 host-side kernels: fused row-sparse Adam, row gather/scatter over interleaved [R,3,D] state.
// Arithmetic is written as separately rounded FP32 operations; compiled with -ffp-contract=off so
// results are bitwise identical to the Triton kernel in gpu_ops.py (div_rn/sqrt_rn, FP fusion disabled).
#include <torch/extension.h>
#include <ATen/Parallel.h>
#include <cmath>
#include <cstring>
#include <thread>
#include <vector>

namespace py = pybind11;

struct AdamScalars { float lr, b1, omb1, b2, omb2, bc1, bc2, eps; };

AdamScalars make_scalars(double lr, double b1, double b2, double eps, int64_t step) {
  AdamScalars s;
  s.lr = (float)lr; s.b1 = (float)b1; s.omb1 = (float)(1.0 - b1); s.b2 = (float)b2; s.omb2 = (float)(1.0 - b2);
  s.bc1 = (float)(1.0 - std::pow(b1, (double)step)); s.bc2 = (float)(1.0 - std::pow(b2, (double)step)); s.eps = (float)eps;
  return s;
}

static inline void adam_row(float* __restrict w, float* __restrict m, float* __restrict v,
                            const float* __restrict g, int64_t D, const AdamScalars& s) {
#pragma omp simd
  for (int64_t j = 0; j < D; ++j) {
    float gj = g[j];
    float mj = s.b1 * m[j];  float t1 = s.omb1 * gj;  mj = mj + t1;
    float vj = s.b2 * v[j];  float t2 = s.omb2 * gj;  t2 = t2 * gj;  vj = vj + t2;
    float mh = mj / s.bc1;   float vh = vj / s.bc2;
    float den = std::sqrt(vh); den = den + s.eps;
    float num = s.lr * mh;   float up = num / den;
    w[j] = w[j] - up; m[j] = mj; v[j] = vj;
  }
}

// state: [R,3,D] float32 (w,m,v interleaved per row), rows: int64 [n], grad: float32 [n,D]
void cpu_adam(torch::Tensor state, torch::Tensor rows, torch::Tensor grad, double lr, double b1, double b2,
              double eps, int64_t step, int64_t nthreads) {
  TORCH_CHECK(state.is_contiguous() && grad.is_contiguous() && rows.is_contiguous());
  const int64_t n = rows.numel(), D = state.size(2);
  AdamScalars s = make_scalars(lr, b1, b2, eps, step);
  float* S = state.data_ptr<float>(); const float* G = grad.data_ptr<float>(); const int64_t* r = rows.data_ptr<int64_t>();
  auto work = [&](int64_t lo, int64_t hi) {
    for (int64_t i = lo; i < hi; ++i) {
      float* base = S + r[i] * 3 * D;
      adam_row(base, base + D, base + 2 * D, G + i * D, D, s);
    }
  };
  if (nthreads <= 1 || n < 256) { work(0, n); return; }
  std::vector<std::thread> th; int64_t chunk = (n + nthreads - 1) / nthreads;
  for (int64_t t = 0; t < nthreads; ++t) {
    int64_t lo = t * chunk, hi = std::min(n, lo + chunk); if (lo >= hi) break;
    th.emplace_back(work, lo, hi);
  }
  for (auto& x : th) x.join();
}

// Gather the first ncomp D-vectors of records rows[i] of state [N,W,D] into out [n,ncomp,D].
// (W=3: interleaved w,m,v records; W=1: log-entry arena.)
void cpu_gather(torch::Tensor state, torch::Tensor rows, torch::Tensor out, int64_t ncomp, int64_t nthreads) {
  const int64_t n = rows.numel(), D = state.size(2), W = state.size(1);
  const float* S = state.data_ptr<float>(); float* O = out.data_ptr<float>(); const int64_t* r = rows.data_ptr<int64_t>();
  const size_t bytes = (size_t)ncomp * D * sizeof(float);
  auto work = [&](int64_t lo, int64_t hi) { for (int64_t i = lo; i < hi; ++i) std::memcpy(O + i * ncomp * D, S + r[i] * W * D, bytes); };
  if (nthreads <= 1 || n < 1024) { work(0, n); return; }
  std::vector<std::thread> th; int64_t chunk = (n + nthreads - 1) / nthreads;
  for (int64_t t = 0; t < nthreads; ++t) { int64_t lo = t * chunk, hi = std::min(n, lo + chunk); if (lo >= hi) break; th.emplace_back(work, lo, hi); }
  for (auto& x : th) x.join();
}

// Scatter records src [n,Wc,D] into the first Wc D-vectors of records rows[i] of state [N,W,D].
void cpu_scatter(torch::Tensor state, torch::Tensor rows, torch::Tensor src, int64_t nthreads) {
  const int64_t n = rows.numel(), D = state.size(2), W = state.size(1), Wc = src.dim() == 3 ? src.size(1) : 1;
  float* S = state.data_ptr<float>(); const float* I = src.data_ptr<float>(); const int64_t* r = rows.data_ptr<int64_t>();
  const size_t bytes = (size_t)Wc * D * sizeof(float);
  auto work = [&](int64_t lo, int64_t hi) { for (int64_t i = lo; i < hi; ++i) std::memcpy(S + r[i] * W * D, I + i * Wc * D, bytes); };
  if (nthreads <= 1 || n < 1024) { work(0, n); return; }
  std::vector<std::thread> th; int64_t chunk = (n + nthreads - 1) / nthreads;
  for (int64_t t = 0; t < nthreads; ++t) { int64_t lo = t * chunk, hi = std::min(n, lo + chunk); if (lo >= hi) break; th.emplace_back(work, lo, hi); }
  for (auto& x : th) x.join();
}



// Generic parallel-for over [0,n) with nthreads std::threads.
template <class F> static void pfor(int64_t n, int64_t nthreads, int64_t minpar, F f) {
  if (nthreads <= 1 || n < minpar) { f(0, n); return; }
  std::vector<std::thread> th; int64_t chunk = (n + nthreads - 1) / nthreads;
  for (int64_t t = 0; t < nthreads; ++t) { int64_t lo = t * chunk, hi = std::min(n, lo + chunk); if (lo >= hi) break; th.emplace_back(f, lo, hi); }
  for (auto& x : th) x.join();
}

// Variable-length gather of the first cnt[i] D-vectors of each row record into a flat buffer.
// offs[i] = prefix sum of cnt (in D-vectors). out_flat: [sum(cnt), D]
void cpu_gather_var(torch::Tensor state, torch::Tensor rows, torch::Tensor cnt, torch::Tensor offs, torch::Tensor out, int64_t nthreads) {
  const int64_t n = rows.numel(), D = state.size(2);
  const float* S = state.data_ptr<float>(); float* O = out.data_ptr<float>();
  const int64_t* r = rows.data_ptr<int64_t>(); const int32_t* c = cnt.data_ptr<int32_t>(); const int64_t* o = offs.data_ptr<int64_t>();
  pfor(n, nthreads, 1024, [&](int64_t lo, int64_t hi) {
    for (int64_t i = lo; i < hi; ++i) std::memcpy(O + o[i] * D, S + r[i] * 3 * D, (size_t)c[i] * D * sizeof(float)); });
}

// Write one D-vector per row into component slot k[i] of the row record (log append) and record the
// logical step of the entry in logstep[row, k] (int32 [R, Kw]).
void cpu_scatter_slot(torch::Tensor state, torch::Tensor rows, torch::Tensor ks, torch::Tensor src,
                      torch::Tensor logstep, int64_t step, int64_t nthreads) {
  const int64_t n = rows.numel(), D = state.size(2), Kw = logstep.size(1);
  float* S = state.data_ptr<float>(); const float* I = src.data_ptr<float>(); int32_t* LS = logstep.data_ptr<int32_t>();
  const int64_t* r = rows.data_ptr<int64_t>(); const int32_t* k = ks.data_ptr<int32_t>();
  pfor(n, nthreads, 1024, [&](int64_t lo, int64_t hi) {
    for (int64_t i = lo; i < hi; ++i) {
      std::memcpy(S + (r[i] * 3 + k[i]) * D, I + i * D, D * sizeof(float));
      LS[r[i] * Kw + k[i]] = (int32_t)step;
    } });
}

// Parallel zero-fill (pre-faults host pages so first-touch faults are not charged to training).
void cpu_prefault(torch::Tensor t, int64_t nthreads) {
  char* p = (char*)t.data_ptr(); const int64_t bytes = t.numel() * t.element_size(); const int64_t chunk = 1 << 24;
  const int64_t nch = (bytes + chunk - 1) / chunk;
  pfor(nch, nthreads, 2, [&](int64_t lo, int64_t hi) {
    for (int64_t c = lo; c < hi; ++c) { int64_t a = c * chunk, b = std::min(bytes, a + chunk); std::memset(p + a, 0, b - a); } });
}


// ---------------- ELARA compact-store fused operations ----------------
// Index translation + allocation bookkeeping + copy in one GIL-free multithreaded call (the fast path of the
// compact store). Storage locations never enter the arithmetic, so these change no numerics.
#include <atomic>

// out[i, :ncomp] = mat[loc[rows[i]], :ncomp]; returns the number of rows without a MAT slot (must be 0).
int64_t cs_gather_full(torch::Tensor mat, torch::Tensor loc, torch::Tensor rows, torch::Tensor out, int64_t ncomp, int64_t nth) {
  const int64_t n = rows.numel(), D = mat.size(2);
  const float* S = mat.data_ptr<float>(); float* O = out.data_ptr<float>();
  const int32_t* L = loc.data_ptr<int32_t>(); const int64_t* r = rows.data_ptr<int64_t>();
  const size_t bytes = (size_t)ncomp * D * sizeof(float); std::atomic<int64_t> bad{0};
  pfor(n, nth, 1024, [&](int64_t lo, int64_t hi) {
    int64_t b = 0;
    for (int64_t i = lo; i < hi; ++i) { int32_t sl = L[r[i]]; if (sl < 0) { ++b; continue; } std::memcpy(O + i * ncomp * D, S + (int64_t)sl * 3 * D, bytes); }
    bad += b; });
  return bad.load();
}

// Gather the first cnt[i] log entries of each row (ids in ent[row, :]) into out_lg[off[i] + k] and the step of
// every entry slot into out_ls[i, k] (0 for empty slots). Returns the number of missing entries (must be 0).
int64_t cs_gather_logs(torch::Tensor logs, torch::Tensor ent, torch::Tensor logstep, torch::Tensor rows, torch::Tensor cnt,
                       torch::Tensor off, torch::Tensor out_lg, torch::Tensor out_ls, int64_t nth) {
  const int64_t n = rows.numel(), D = logs.size(2), Kw = ent.size(1);
  const float* S = logs.data_ptr<float>(); float* O = out_lg.data_ptr<float>(); int32_t* OS = out_ls.data_ptr<int32_t>();
  const int32_t* E = ent.data_ptr<int32_t>(); const int32_t* LS = logstep.data_ptr<int32_t>();
  const int64_t* r = rows.data_ptr<int64_t>(); const int32_t* c = cnt.data_ptr<int32_t>(); const int64_t* o = off.data_ptr<int64_t>();
  std::atomic<int64_t> bad{0};
  pfor(n, nth, 1024, [&](int64_t lo, int64_t hi) {
    int64_t b = 0;
    for (int64_t i = lo; i < hi; ++i) {
      const int32_t* e = E + r[i] * Kw;
      for (int64_t k = 0; k < Kw; ++k) {
        int32_t id = e[k];
        if (k < c[i]) { if (id < 0) { ++b; } else { std::memcpy(O + (o[i] + k) * D, S + (int64_t)id * D, D * sizeof(float)); } }
        OS[i * Kw + k] = id >= 0 ? LS[id] : 0;
      }
    }
    bad += b; });
  return bad.load();
}

// Log append: ent[rows[i], ks[i]] = ids[i]; logstep[ids[i]] = steps[i] (or steps[0] if steps has one element);
// logs[ids[i]] = g[i].
void cs_append(torch::Tensor logs, torch::Tensor ent, torch::Tensor logstep, torch::Tensor rows, torch::Tensor ks,
               torch::Tensor ids, torch::Tensor g, torch::Tensor steps, int64_t nth) {
  const int64_t n = rows.numel(), D = logs.size(2), Kw = ent.size(1); const bool one = steps.numel() == 1;
  float* S = logs.data_ptr<float>(); const float* G = g.data_ptr<float>(); int32_t* E = ent.data_ptr<int32_t>(); int32_t* LS = logstep.data_ptr<int32_t>();
  const int64_t* r = rows.data_ptr<int64_t>(); const int32_t* k = ks.data_ptr<int32_t>(); const int32_t* id = ids.data_ptr<int32_t>();
  const int32_t* st = steps.data_ptr<int32_t>();
  pfor(n, nth, 1024, [&](int64_t lo, int64_t hi) {
    for (int64_t i = lo; i < hi; ++i) {
      E[r[i] * Kw + k[i]] = id[i]; LS[id[i]] = one ? st[0] : st[i];
      std::memcpy(S + (int64_t)id[i] * D, G + i * D, D * sizeof(float));
    } });
}

// Full-state write: rows without a MAT slot get consecutive slots base, base+1, ... in row order; src[i] is copied
// into mat[loc[rows[i]]]; every live log entry of the rows is released into freed (row-major, k order) and
// ent[rows[i], :] = -1. Returns {n_new, n_freed}. Caller guarantees capacity base + n.
std::vector<int64_t> cs_write_full(torch::Tensor mat, torch::Tensor loc, torch::Tensor ent, torch::Tensor rows, torch::Tensor src,
                                   int64_t base, torch::Tensor freed, int64_t nth) {
  const int64_t n = rows.numel(), D = mat.size(2), Kw = ent.size(1);
  float* S = mat.data_ptr<float>(); const float* I = src.data_ptr<float>(); int32_t* L = loc.data_ptr<int32_t>();
  int32_t* E = ent.data_ptr<int32_t>(); int32_t* F = freed.data_ptr<int32_t>(); const int64_t* r = rows.data_ptr<int64_t>();
  const int64_t T = (nth <= 1 || n < 1024) ? 1 : nth, chunk = (n + T - 1) / T;
  std::vector<int64_t> cn(T + 1, 0), cf(T + 1, 0);
  pfor(T, T, 1, [&](int64_t lo, int64_t hi) {          // pass 1: per-chunk counts of new slots and freed entries
    for (int64_t t = lo; t < hi; ++t) {
      int64_t a = t * chunk, b = std::min(n, a + chunk), nn = 0, nf = 0;
      for (int64_t i = a; i < b; ++i) { if (L[r[i]] < 0) ++nn; const int32_t* e = E + r[i] * Kw; for (int64_t k = 0; k < Kw; ++k) nf += e[k] >= 0; }
      cn[t + 1] = nn; cf[t + 1] = nf;
    } });
  for (int64_t t = 0; t < T; ++t) { cn[t + 1] += cn[t]; cf[t + 1] += cf[t]; }
  pfor(T, T, 1, [&](int64_t lo, int64_t hi) {          // pass 2: assign, copy, release logs
    for (int64_t t = lo; t < hi; ++t) {
      int64_t a = t * chunk, b = std::min(n, a + chunk), sn = base + cn[t], fo = cf[t];
      for (int64_t i = a; i < b; ++i) {
        int32_t sl = L[r[i]]; if (sl < 0) { sl = (int32_t)(sn++); L[r[i]] = sl; }
        std::memcpy(S + (int64_t)sl * 3 * D, I + i * 3 * D, 3 * D * sizeof(float));
        int32_t* e = E + r[i] * Kw;
        for (int64_t k = 0; k < Kw; ++k) if (e[k] >= 0) { F[fo++] = e[k]; e[k] = -1; }
      }
    } });
  return {cn[T], cf[T]};
}


// ---------------- host memory policy (applied identically to every store) ----------------
#include <sys/mman.h>
#include <cerrno>
#ifndef MADV_POPULATE_WRITE
#define MADV_POPULATE_WRITE 23
#endif
// Opt a host table into transparent huge pages (THP mode 'madvise'): fewer faults, larger TLB reach.
int64_t mem_hugepage(torch::Tensor t) {
  const uintptr_t a = (uintptr_t)t.data_ptr(), e = a + (uintptr_t)(t.numel() * t.element_size());
  const uintptr_t H = 2u << 20, lo = (a + H - 1) & ~(H - 1), hi = e & ~(H - 1);
  if (hi <= lo) return 0;
  return madvise((void*)lo, hi - lo, MADV_HUGEPAGE) == 0 ? 0 : errno;
}
// Fault-in (populate) a range for writing without writing it from user space (kernel zero-fill only);
// falls back to touching one byte per page. Parallel over nthreads sub-ranges.
int64_t mem_populate(torch::Tensor t, int64_t nthreads) {
  const uintptr_t P = 4096, a = (uintptr_t)t.data_ptr(), e = a + (uintptr_t)(t.numel() * t.element_size());
  const uintptr_t lo = a & ~(P - 1), hi = (e + P - 1) & ~(P - 1);
  if (hi <= lo) return 0;
  const int64_t npages = (int64_t)((hi - lo) / P); std::atomic<int64_t> err{0};
  pfor(npages, nthreads, 256, [&](int64_t p0, int64_t p1) {
    void* s = (void*)(lo + (uintptr_t)p0 * P); size_t len = (size_t)(p1 - p0) * P;
    if (madvise(s, len, MADV_POPULATE_WRITE) != 0) {
      err = errno;
      for (int64_t q = p0; q < p1; ++q) { volatile char* c = (volatile char*)(lo + (uintptr_t)q * P); if ((uintptr_t)c >= a && (uintptr_t)c < e) *c = *c; }
    } });
  return err.load();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("mem_hugepage", &mem_hugepage, py::call_guard<py::gil_scoped_release>());
  m.def("mem_populate", &mem_populate, py::call_guard<py::gil_scoped_release>());
  m.def("cs_gather_full", &cs_gather_full, py::call_guard<py::gil_scoped_release>());
  m.def("cs_gather_logs", &cs_gather_logs, py::call_guard<py::gil_scoped_release>());
  m.def("cs_append", &cs_append, py::call_guard<py::gil_scoped_release>());
  m.def("cs_write_full", &cs_write_full, py::call_guard<py::gil_scoped_release>());
  m.def("cpu_adam", &cpu_adam, py::call_guard<py::gil_scoped_release>());
  m.def("cpu_gather", &cpu_gather, py::call_guard<py::gil_scoped_release>());
  m.def("cpu_scatter", &cpu_scatter, py::call_guard<py::gil_scoped_release>());
  m.def("cpu_gather_var", &cpu_gather_var, py::call_guard<py::gil_scoped_release>());
  m.def("cpu_scatter_slot", &cpu_scatter_slot, py::call_guard<py::gil_scoped_release>());
  m.def("cpu_prefault", &cpu_prefault, py::call_guard<py::gil_scoped_release>());
  m.def("adam_scalars", [](double lr, double b1, double b2, double eps, int64_t step) {
    AdamScalars s = make_scalars(lr, b1, b2, eps, step);
    return std::vector<float>{s.lr, s.b1, s.omb1, s.b2, s.omb2, s.bc1, s.bc2, s.eps}; });
}

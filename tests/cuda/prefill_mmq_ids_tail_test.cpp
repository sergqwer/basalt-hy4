// prefill_mmq_ids_tail_test - an MMQ product's `ids` table that ends where the mapped memory ends (mmq::kIdsTail).
// Each expert's tile reads a whole tile width of ids from its first row, so the last expert's tile reads past its last
// row; the prompt path's identity tables (pair_ident, mmq_ident, heads_ids) are sized kIdsTail entries longer for it.
// The 2026-11-23 bug: hy4flash's pair_ident ended a cudaMalloc range after a decode turn and a 64K prompt's last down
// product read past it (an illegal memory access, reported as "prefill mmq: quantize").
// Here the table sits at the very end of a VMM mapping with an unmapped granule after it; a product over 129 rows (two
// tiles, the second reading 127 entries past the last row) must run clean and equal the same product with a table in
// plain cudaMalloc memory.  Exit 77 without a CUDA device or VMM.
#include "basalt/prefill/moe_mmq.hpp"

#include "basalt/core/vmm.hpp"
#include "basalt/ggml/ggml.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace {
namespace mmq = basalt::prefill::mmq;
namespace vmm = basalt::core::vmm;

constexpr int64_t kCols = 256, kOut = 256, kRows = 129;

bool ok(cudaError_t e, const char* what) {
    if (e == cudaSuccess) return true;
    std::fprintf(stderr, "prefill_mmq_ids_tail_test: %s: %s\n", what, cudaGetErrorString(e));
    return false;
}

// one product of kRows rows (one expert) with `ids` as its row table; the result to `out`
bool product(mmq::Context& ctx, const void* w, const void* xq, const int32_t* bounds, const int32_t* ids, float* dst,
             std::vector<float>& out) {
    mmq::Product p;
    p.w = w;
    p.type = GGML_TYPE_Q8_0;
    p.w_rows = kOut;
    p.w_cols = kCols;
    p.expert_bytes = mmq::matrix_bytes(GGML_TYPE_Q8_0, kOut, kCols);
    p.n = 1;
    p.xq = xq;
    p.bounds = bounds;
    p.ids = ids;
    p.total_rows = kRows;
    p.max_rows = kRows;
    p.dst = dst;
    p.ld_dst = kOut;
    if (!ok(cudaMemset(dst, 0, (size_t) kRows * kOut * 4), "memset")) return false;
    ctx.run(p, nullptr);
    if (!ok(cudaDeviceSynchronize(), "the product")) return false;
    out.resize((size_t) kRows * kOut);
    return ok(cudaMemcpy(out.data(), dst, out.size() * 4, cudaMemcpyDeviceToHost), "copy out");
}
}  // namespace

int main() {
    int n_dev = 0;
    if (cudaGetDeviceCount(&n_dev) != cudaSuccess || n_dev == 0) {
        std::printf("prefill_mmq_ids_tail_test: no CUDA device, skipped\n");
        return 77;
    }
    cudaSetDevice(0);
    cudaFree(nullptr);
    if (!vmm::supported() || vmm::granularity() == 0 || !mmq::built() || !mmq::fits(GGML_TYPE_Q8_0, kOut)) {
        std::printf("prefill_mmq_ids_tail_test: no VMM or no Q8_0 MMQ, skipped\n");
        return 77;
    }
    // Q8_0 weights and fp32 activations, deterministic
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> wf((size_t) kOut * kCols), xf((size_t) kRows * kCols);
    for (float& v : wf) v = nd(rng);
    for (float& v : xf) v = nd(rng);
    const size_t wb = mmq::matrix_bytes(GGML_TYPE_Q8_0, kOut, kCols);
    std::vector<uint8_t> wq(wb + mmq::read_past_bytes(GGML_TYPE_Q8_0, kCols) + 64);
    ggml_get_type_traits(GGML_TYPE_Q8_0)->from_float_ref(wf.data(), wq.data(), (int64_t) wf.size());

    void *w = nullptr, *x = nullptr, *xq = nullptr, *dst = nullptr, *bounds = nullptr, *plain_ids = nullptr;
    const int32_t hb[2] = {0, (int32_t) kRows};
    if (!ok(cudaMalloc(&w, wq.size()), "malloc") || !ok(cudaMemcpy(w, wq.data(), wq.size(), cudaMemcpyHostToDevice), "w") ||
        !ok(cudaMalloc(&x, xf.size() * 4), "malloc") || !ok(cudaMemcpy(x, xf.data(), xf.size() * 4, cudaMemcpyHostToDevice), "x") ||
        !ok(cudaMalloc(&xq, mmq::q8_bytes(kRows, kCols)), "malloc") || !ok(cudaMalloc(&dst, (size_t) kRows * kOut * 4), "malloc") ||
        !ok(cudaMalloc(&bounds, sizeof hb), "malloc") || !ok(cudaMemcpy(bounds, hb, sizeof hb, cudaMemcpyHostToDevice), "bounds") ||
        !ok(cudaMalloc(&plain_ids, (size_t) (kRows + 4096) * 4), "malloc"))
        return 1;
    mmq::load(GGML_TYPE_Q8_0, kOut);
    mmq::quantize((const float*) x, nullptr, xq, GGML_TYPE_Q8_0, kCols, kCols, kRows, nullptr);
    mmq::iota((int32_t*) plain_ids, kRows + 4096, nullptr);

    // the table at the end of a one-granule mapping, the next granule reserved and unmapped
    const uint64_t g = vmm::granularity();
    uint8_t* va = vmm::reserve(2 * g);
    const unsigned long long h = va ? vmm::map(va, g) : 0;
    if (!h) {
        std::printf("prefill_mmq_ids_tail_test: cannot map a granule, skipped\n");
        return 77;
    }
    const int64_t n_ids = kRows + mmq::kIdsTail;   // what the prompt path allocates for kRows rows
    int32_t* edge_ids = (int32_t*) (va + g) - n_ids;
    mmq::iota(edge_ids, n_ids, nullptr);

    mmq::Context ctx;
    std::vector<float> a, b;
    if (!product(ctx, w, xq, (const int32_t*) bounds, (const int32_t*) plain_ids, (float*) dst, a)) return 1;
    if (!product(ctx, w, xq, (const int32_t*) bounds, edge_ids, (float*) dst, b)) return 1;
    if (std::memcmp(a.data(), b.data(), a.size() * 4) != 0) {
        std::fprintf(stderr, "prefill_mmq_ids_tail_test: the product over the edge table differs from the plain one\n");
        return 1;
    }
    double s = 0;
    for (float v : a) s += (double) v * v;
    if (!(s > 0)) {
        std::fprintf(stderr, "prefill_mmq_ids_tail_test: the product is all zero\n");
        return 1;
    }
    vmm::unmap(va, g, h);
    vmm::release_range(va, 2 * g);
    std::printf("prefill_mmq_ids_tail_test: %lld rows over an ids table ending at an unmapped granule: clean, equal to the plain table\n",
                (long long) kRows);
    return 0;
}

// E1-8 microbench: the prompt path's dense Q8_0 products through MMQ at a 16K chunk - the time of the q8_1 quantize and of
// the product for each Hy4 dense shape, and the int8 ops/s they reach (2 R rows cols per product).
#include "basalt/prefill/moe_mmq.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace mmq = basalt::prefill::mmq;

int main(int argc, char** argv) {
    int dev = 0;
    if (cudaGetDeviceCount(&dev) != cudaSuccess || dev == 0 || !mmq::built()) {
        std::printf("no GPU or no MMQ, skipped\n");
        return 77;
    }
    const int kQ8 = 8;   // GGML_TYPE_Q8_0
    const int64_t R = argc > 1 ? std::atoll(argv[1]) : 16384;
    struct Shape { const char* name; int64_t in, out, per_layer; };
    // Hy4-flash: n_embd 3072, q_lora 1024, heads 64 (most layers; the 128-head layers are twice o / gate / q_b)
    const Shape shapes[] = {{"attn_output", 8192, 3072, 1}, {"attn_gate", 3072, 8192, 1}, {"attn_q_b", 1024, 8192, 1},
                            {"attn_q_a", 3072, 1024, 1},    {"kv_a", 3072, 576, 1},       {"shexp gate+up", 3072, 1536, 2},
                            {"shexp down", 1536, 3072, 1},  {"routed_down", 3072, 1536, 1}, {"routed_up", 1536, 3072, 1}};
    std::mt19937 rng(7);
    double total_ms = 0, total_q = 0;
    float* x;
    void* xq;
    float* y;
    int32_t *bounds, *ids;
    const int64_t maxc = 8192, maxr = 8192;
    cudaMalloc(&x, (size_t) R * maxc * 4);
    cudaMalloc(&xq, mmq::q8_bytes(R, maxc));
    cudaMalloc(&y, (size_t) R * maxr * 4);
    cudaMalloc(&bounds, 8);
    cudaMalloc(&ids, (size_t) (R + mmq::kIdsTail) * 4);
    {
        std::vector<float> hx((size_t) R * maxc);
        std::normal_distribution<float> nd(0.f, 1.f);
        for (float& v : hx) v = nd(rng);
        cudaMemcpy(x, hx.data(), hx.size() * 4, cudaMemcpyHostToDevice);
        const int32_t hb[2] = {0, (int32_t) R};
        cudaMemcpy(bounds, hb, 8, cudaMemcpyHostToDevice);
        mmq::iota(ids, R + mmq::kIdsTail, nullptr);
    }
    mmq::Context ctx;
    cudaEvent_t e0, e1, e2;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    cudaEventCreate(&e2);
    std::printf("R = %lld rows\n%-14s %6s %6s %9s %9s %8s\n", (long long) R, "shape", "in", "out", "quant ms", "gemm ms", "TOPS");
    for (const Shape& s : shapes) {
        const size_t wb = mmq::matrix_bytes(kQ8, s.out, s.in) + mmq::read_past_bytes(kQ8, s.in);
        std::vector<uint8_t> hw(wb, 0);
        const size_t blocks = (size_t) s.out * s.in / 32;
        std::uniform_int_distribution<int> code(-127, 127);
        for (size_t b = 0; b < blocks; ++b) {
            const uint16_t d = 0x211F;   // fp16 0.01
            std::memcpy(&hw[b * 34], &d, 2);
            for (int i = 0; i < 32; ++i) hw[b * 34 + 2 + i] = (uint8_t) (int8_t) code(rng);
        }
        void* w;
        cudaMalloc(&w, wb);
        cudaMemcpy(w, hw.data(), wb, cudaMemcpyHostToDevice);
        if (!mmq::fits(kQ8, s.out)) {
            std::printf("%-14s does not fit MMQ\n", s.name);
            cudaFree(w);
            continue;
        }
        mmq::load(kQ8, s.out);
        mmq::Product p;
        p.w = w;
        p.type = kQ8;
        p.w_rows = s.out;
        p.w_cols = s.in;
        p.expert_bytes = mmq::matrix_bytes(kQ8, s.out, s.in);
        p.n = 1;
        p.xq = xq;
        p.bounds = bounds;
        p.ids = ids;
        p.total_rows = R;
        p.max_rows = R;
        p.dst = y;
        p.ld_dst = s.out;
        float qms = 0, gms = 0;
        for (int rep = 0; rep < 3; ++rep) {   // the first pass warms up; the last is kept
            cudaEventRecord(e0);
            mmq::quantize(x, nullptr, xq, kQ8, s.in, s.in, R, nullptr);
            cudaEventRecord(e1);
            ctx.run(p, nullptr);
            cudaEventRecord(e2);
            cudaEventSynchronize(e2);
            cudaEventElapsedTime(&qms, e0, e1);
            cudaEventElapsedTime(&gms, e1, e2);
        }
        if (cudaGetLastError() != cudaSuccess) {
            std::printf("FAIL %s\n", s.name);
            return 1;
        }
        const double tops = 2.0 * R * s.out * s.in / (gms * 1e-3) / 1e12;
        std::printf("%-14s %6lld %6lld %9.3f %9.3f %8.1f\n", s.name, (long long) s.in, (long long) s.out, qms, gms, tops);
        total_ms += (qms + gms) * s.per_layer;
        total_q += qms * s.per_layer;
        cudaFree(w);
    }
    std::printf("per layer (64-head layer, these shapes): %.2f ms, of it quantize %.2f ms; x 42 layers: %.0f ms per chunk\n", total_ms,
                total_q, total_ms * 42);
    return 0;
}

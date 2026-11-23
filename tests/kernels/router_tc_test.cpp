// router_logits_bf16x3 against a double reference and against router()'s fp32 logits: the error bound of the two-term
// split, and how often the chosen experts differ (near ties only).
#include "basalt/kernels/router.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

namespace bk = basalt::kernels;

int main() {
    int dev = 0;
    if (cudaGetDeviceCount(&dev) != cudaSuccess || dev == 0) {
        std::printf("no GPU, skipped\n");
        return 77;
    }
    const int E = 512, K = 3072, k = 8;
    bool ok = true;
    for (const int T : {1, 37, 128, 300, 4096}) {
        std::mt19937 rng(1234 + T);
        std::normal_distribution<float> nd(0.f, 1.f);
        std::vector<float> x((size_t) T * K), w((size_t) E * K), bias(E);
        for (float& v : x) v = nd(rng);
        for (float& v : w) v = nd(rng) * 0.02f;
        for (float& v : bias) v = nd(rng) * 0.01f;
        float *dx, *dw, *db, *l32, *ltc, *w32, *wtc;
        int32_t *i32, *itc;
        cudaMalloc(&dx, x.size() * 4);
        cudaMalloc(&dw, w.size() * 4);
        cudaMalloc(&db, E * 4);
        cudaMalloc(&l32, (size_t) T * E * 4);
        cudaMalloc(&ltc, (size_t) T * E * 4);
        cudaMalloc(&w32, (size_t) T * k * 4);
        cudaMalloc(&wtc, (size_t) T * k * 4);
        cudaMalloc(&i32, (size_t) T * k * 4);
        cudaMalloc(&itc, (size_t) T * k * 4);
        cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(dw, w.data(), w.size() * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(db, bias.data(), E * 4, cudaMemcpyHostToDevice);
        bk::RouterArgs a;
        a.n_expert = E;
        a.n_embd = K;
        a.k = k;
        a.weight_scale = 2.5f;
        a.gate = bk::RouterGate::Sigmoid;
        bk::router(dx, dw, db, T, a, l32, i32, w32, nullptr);
        bk::router_logits_bf16x3(dx, dw, T, E, K, ltc, nullptr);
        bk::router_select(ltc, db, T, a, itc, wtc, nullptr);
        if (cudaDeviceSynchronize() != cudaSuccess) {
            std::printf("FAIL T=%d: %s\n", T, cudaGetErrorString(cudaGetLastError()));
            return 1;
        }
        std::vector<float> h32((size_t) T * E), htc((size_t) T * E);
        std::vector<int32_t> s32((size_t) T * k), stc((size_t) T * k);
        cudaMemcpy(h32.data(), l32, h32.size() * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(htc.data(), ltc, htc.size() * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(s32.data(), i32, s32.size() * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(stc.data(), itc, stc.size() * 4, cudaMemcpyDeviceToHost);
        double e32 = 0, etc = 0;   // the largest error over sum |x w|
        for (int t = 0; t < T; t += (T > 512 ? 7 : 1))
            for (int e = 0; e < E; ++e) {
                double ref = 0, mag = 0;
                for (int i = 0; i < K; ++i) {
                    const double p = (double) x[(size_t) t * K + i] * w[(size_t) e * K + i];
                    ref += p;
                    mag += std::fabs(p);
                }
                e32 = std::fmax(e32, std::fabs(h32[(size_t) t * E + e] - ref) / mag);
                etc = std::fmax(etc, std::fabs(htc[(size_t) t * E + e] - ref) / mag);
            }
        int diff = 0;
        for (int t = 0; t < T; ++t)
            for (int j = 0; j < k; ++j) diff += s32[(size_t) t * k + j] != stc[(size_t) t * k + j];
        const bool pass = etc < 3e-5 && diff <= std::max(2, T * k / 500);
        std::printf("%s T=%d: rel err fp32 %.2e, bf16x3 %.2e; chosen ids differing %d of %d\n", pass ? "ok  " : "FAIL", T, e32, etc, diff,
                    T * k);
        ok = ok && pass;
        cudaFree(dx); cudaFree(dw); cudaFree(db); cudaFree(l32); cudaFree(ltc); cudaFree(w32); cudaFree(wtc); cudaFree(i32); cudaFree(itc);
    }
    {   // a 16K chunk's logits, 41 layers' worth: the fp32 router vs the tensor cores
        const int T = 16384;
        float *dx, *dw, *db, *lg, *wt;
        int32_t* id;
        cudaMalloc(&dx, (size_t) T * K * 4);
        cudaMalloc(&dw, (size_t) E * K * 4);
        cudaMalloc(&db, E * 4);
        cudaMalloc(&lg, (size_t) T * E * 4);
        cudaMalloc(&wt, (size_t) T * k * 4);
        cudaMalloc(&id, (size_t) T * k * 4);
        cudaMemset(dx, 0, (size_t) T * K * 4);
        cudaMemset(dw, 0, (size_t) E * K * 4);
        cudaMemset(db, 0, E * 4);
        bk::RouterArgs a;
        a.n_expert = E;
        a.n_embd = K;
        a.k = k;
        a.gate = bk::RouterGate::Sigmoid;
        cudaEvent_t e0, e1;
        cudaEventCreate(&e0);
        cudaEventCreate(&e1);
        float ms[2] = {};
        for (int v = 0; v < 2; ++v) {
            for (int rep = 0; rep < 2; ++rep) {   // the first pass warms up
                cudaEventRecord(e0);
                for (int l = 0; l < 41; ++l)
                    if (v == 0) {
                        bk::router(dx, dw, db, T, a, lg, id, wt, nullptr);
                    } else {
                        bk::router_logits_bf16x3(dx, dw, T, E, K, lg, nullptr);
                        bk::router_select(lg, db, T, a, id, wt, nullptr);
                    }
                cudaEventRecord(e1);
                cudaEventSynchronize(e1);
                cudaEventElapsedTime(&ms[v], e0, e1);
            }
        }
        std::printf("16K chunk x 41 layers: router fp32 %.1f ms, bf16x3 + select %.1f ms (64K prompt: %.0f -> %.0f ms)\n", ms[0], ms[1],
                    4 * ms[0], 4 * ms[1]);
        cudaFree(dx); cudaFree(dw); cudaFree(db); cudaFree(lg); cudaFree(wt); cudaFree(id);
    }
    std::printf(ok ? "PASS\n" : "FAIL\n");
    return ok ? 0 : 1;
}

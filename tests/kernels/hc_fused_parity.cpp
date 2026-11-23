// hc_post_mix_norm against the three kernels it replaces (hc_post, hc_mixes_plain with 8 mixes, hc_pre_rms_norm): every
// output bit-identical. With --bench N it also times both at N tokens.
#include "basalt/kernels/mhc.hpp"
#include "basalt/kernels/row_norm.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace bk = basalt::kernels;

namespace {

float* up(const std::vector<float>& h) {
    float* d = nullptr;
    cudaMalloc((void**) &d, h.size() * 4);
    cudaMemcpy(d, h.data(), h.size() * 4, cudaMemcpyHostToDevice);
    return d;
}
std::vector<float> down(const float* d, size_t n) {
    std::vector<float> h(n);
    cudaMemcpy(h.data(), d, n * 4, cudaMemcpyDeviceToHost);
    return h;
}
bool same(const char* what, const std::vector<float>& a, const std::vector<float>& b) {
    if (std::memcmp(a.data(), b.data(), a.size() * 4) == 0) return true;
    size_t n = 0, first = 0;
    for (size_t i = 0; i < a.size(); ++i)
        if (std::memcmp(&a[i], &b[i], 4) != 0 && n++ == 0) first = i;
    std::printf("FAIL %s: %zu of %zu differ, first [%zu] %.9g vs %.9g\n", what, n, a.size(), first, a[first], b[first]);
    return false;
}

struct Case {
    int T, E;
    float* f;
    float* x;
    float* fn;
    float* sc;
    float* base;
    float* w;
    std::vector<float> coef0;
};

Case make(int T, int E, bool random_comb, uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::uniform_real_distribution<float> ud(0.f, 1.f);
    const int flat = bk::kHcMult * E;
    std::vector<float> f((size_t) T * E), x((size_t) T * flat), fn((size_t) 8 * flat), sc(2), base(8), w(E);
    for (float& v : f) v = nd(rng);
    for (float& v : x) v = nd(rng) * 3.f;
    for (float& v : fn) v = nd(rng) * 0.02f;
    sc[0] = 0.7f; sc[1] = 1.3f;
    for (float& v : base) v = nd(rng) * 0.5f;
    for (float& v : w) v = 0.5f + ud(rng);
    std::vector<float> coef((size_t) T * bk::kHcCoef);
    for (int t = 0; t < T; ++t)
        for (int k = 0; k < bk::kHcCoef; ++k) {
            const int c = k - 8;
            coef[(size_t) t * bk::kHcCoef + k] = k < 8 ? ud(rng) * 2.f : random_comb ? nd(rng) : (c >> 2) == (c & 3) ? 1.f : 0.f;
        }
    return Case{T, E, up(f), up(x), up(fn), up(sc), up(base), up(w), coef};
}

bool check(int T, int E, bool random_comb, uint32_t seed) {
    Case c = make(T, E, random_comb, seed);
    const int flat = bk::kHcMult * E;
    bk::HcPlainArgs a;
    a.n_embd = E;
    const float eps = 1e-6f;
    float *coef_r = up(c.coef0), *coef_f = up(c.coef0), *xo_r, *xo_f, *cur_r, *cur_f, *out_r, *out_f, *ws;
    cudaMalloc((void**) &xo_r, (size_t) T * flat * 4);
    cudaMalloc((void**) &xo_f, (size_t) T * flat * 4);
    cudaMalloc((void**) &cur_r, (size_t) T * E * 4);
    cudaMalloc((void**) &cur_f, (size_t) T * E * 4);
    cudaMalloc((void**) &out_r, (size_t) T * E * 4);
    cudaMalloc((void**) &out_f, (size_t) T * E * 4);
    cudaMalloc((void**) &ws, bk::hc_plain_workspace_floats(T) * 4);
    bk::hc_post(c.f, c.x, coef_r, bk::kHcCoef, T, E, xo_r, nullptr);
    bk::hc_mixes_plain(xo_r, c.fn, c.sc, c.base, T, 8, a, ws, coef_r, nullptr);
    bk::hc_pre_rms_norm(xo_r, coef_r, bk::kHcCoef, c.w, T, E, eps, cur_r, out_r, nullptr);
    bk::hc_post_mix_norm(c.f, c.x, coef_f, T, E, c.fn, c.sc, c.base, a, c.w, eps, xo_f, cur_f, out_f, nullptr);
    cudaError_t e = cudaGetLastError();
    if (e == cudaSuccess) e = cudaDeviceSynchronize();
    if (e != cudaSuccess) {
        std::printf("FAIL cuda: %s\n", cudaGetErrorString(e));
        return false;
    }
    bool ok = same("streams", down(xo_r, (size_t) T * flat), down(xo_f, (size_t) T * flat));
    ok = same("coef", down(coef_r, (size_t) T * bk::kHcCoef), down(coef_f, (size_t) T * bk::kHcCoef)) && ok;
    ok = same("cur", down(cur_r, (size_t) T * E), down(cur_f, (size_t) T * E)) && ok;
    ok = same("norm", down(out_r, (size_t) T * E), down(out_f, (size_t) T * E)) && ok;
    std::printf("%s T=%d E=%d comb=%s\n", ok ? "ok" : "FAIL", T, E, random_comb ? "random" : "identity");
    for (float* p : {c.f, c.x, c.fn, c.sc, c.base, c.w, coef_r, coef_f, xo_r, xo_f, cur_r, cur_f, out_r, out_f, ws}) cudaFree(p);
    return ok;
}

void bench(int T) {
    const int E = 3072, flat = bk::kHcMult * E;
    Case c = make(T, E, false, 7);
    bk::HcPlainArgs a;
    float *coef = up(c.coef0), *xo, *cur, *out, *ws;
    cudaMalloc((void**) &xo, (size_t) T * flat * 4);
    cudaMalloc((void**) &cur, (size_t) T * E * 4);
    cudaMalloc((void**) &out, (size_t) T * E * 4);
    cudaMalloc((void**) &ws, bk::hc_plain_workspace_floats(T) * 4);
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    cudaStream_t st;
    cudaStreamCreate(&st);
    for (int arm = 0; arm < 2; ++arm) {
        float best = 1e9f;
        for (int rep = 0; rep < 12; ++rep) {
            cudaMemcpyAsync(coef, c.coef0.data(), c.coef0.size() * 4, cudaMemcpyHostToDevice, st);
            cudaEventRecord(e0, st);
            if (arm == 0) {
                bk::hc_post(c.f, c.x, coef, bk::kHcCoef, T, E, xo, st);
                bk::hc_mixes_plain(xo, c.fn, c.sc, c.base, T, 8, a, ws, coef, st);
                bk::hc_pre_rms_norm(xo, coef, bk::kHcCoef, c.w, T, E, 1e-6f, cur, out, st);
            } else {
                bk::hc_post_mix_norm(c.f, c.x, coef, T, E, c.fn, c.sc, c.base, a, c.w, 1e-6f, xo, cur, out, st);
            }
            cudaEventRecord(e1, st);
            cudaEventSynchronize(e1);
            float ms = 0.f;
            cudaEventElapsedTime(&ms, e0, e1);
            if (rep >= 2 && ms < best) best = ms;
        }
        std::printf("%s: %.3f ms at %d tokens\n", arm == 0 ? "three kernels" : "fused", best, T);
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc == 3 && std::strcmp(argv[1], "--bench") == 0) {
        bench(std::atoi(argv[2]));
        return 0;
    }
    bool ok = true;
    ok = check(37, 3072, false, 1) && ok;
    ok = check(5, 3072, true, 2) && ok;
    ok = check(300, 3072, false, 3) && ok;
    ok = check(9, 1024, true, 4) && ok;
    const long long n = bk::hc_post_mix_norm_launches();
    if (n != 4) ok = false;
    std::printf("fused launches %lld (want 4)\n", n);
    std::printf(ok ? "PASS\n" : "FAIL\n");
    return ok ? 0 : 1;
}

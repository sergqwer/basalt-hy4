// tests/prefill/hy4_fp4_moe_test.cpp - Hy4-Flash's prompt experts on prefill::fp4 (moe_fp4, W4A4): blobs of NVFP4 gate/up/down
// at the 1536 latent with the FP32 tail, one launch for a layer's experts (FLAT), the clamped SwiGLU in the gate/up epilogue.
// Gate/up: against a double product of the weights with the token planes the GPU rounded (read back and decoded), so only the
// accumulation order differs. Down: against the product with the unrounded H, within the FP4 (or two-term) rounding error.
// Exit 77 without a CUDA device.
#include "basalt/prefill/moe_fp4.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace f4 = basalt::prefill::fp4;

namespace {

constexpr int L = 1536, F = 1536;
constexpr size_t kRow = (size_t) L / 64 * 36;   // an NVFP4 row of 1536 values
constexpr float kLimit = 10.0f;

float e2m1(unsigned c) {
    static const float v[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
    return (c & 8u) ? -v[c & 7u] : v[c & 7u];
}
float ue4m3(uint8_t x) {
    x &= 0x7F;
    if (x == 0 || x == 0x7F) return 0.f;
    const int e = (x >> 3) & 0xF, m = x & 7;
    return e == 0 ? std::ldexp((float) m, -9) : std::ldexp(1.f + m / 8.f, e - 7);
}
// one weight row of NVFP4 blocks decoded
void row_values(const uint8_t* row, int K, float* out) {
    for (int b = 0; b < K / 64; ++b) {
        const uint8_t* blk = row + 36 * b;
        for (int s = 0; s < 4; ++s) {
            const float d = ue4m3(blk[s]);
            for (int j = 0; j < 8; ++j) {
                const uint8_t q = blk[4 + 8 * s + j];
                out[64 * b + 16 * s + j] = e2m1(q & 15u) * d;
                out[64 * b + 16 * s + j + 8] = e2m1(q >> 4) * d;
            }
        }
    }
}
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::printf("CUDA %s: %s\n", #x, cudaGetErrorString(e_)); return 1; } } while (0)

int run(bool two) {
    std::mt19937 rng(7);
    const int NE = 3;
    const std::vector<int32_t> counts = {70, 0, 130};
    const int T = 150;
    std::vector<int32_t> bounds = {0};
    for (int c : counts) bounds.push_back(bounds.back() + c);
    const int rows = bounds.back();
    std::vector<int32_t> tok(rows);
    for (int& t : tok) t = (int) (rng() % T);
    // blobs: [gate F | up F | down L] rows, then {s_gate, s_up, s_down, 0}
    const size_t down_off = 2 * (size_t) F * kRow, tail_off = down_off + (size_t) L * kRow, stride = tail_off + 16;
    std::vector<uint8_t> blob(stride * NE);
    std::uniform_int_distribution<int> sc(0x28, 0x3c), by(0, 255);
    for (int e = 0; e < NE; ++e) {
        uint8_t* b = blob.data() + e * stride;
        for (size_t r = 0; r < 2 * (size_t) F + L; ++r)
            for (int k = 0; k < L / 64; ++k) {
                uint8_t* blk = b + r * kRow + 36 * k;
                for (int s = 0; s < 4; ++s) blk[s] = (uint8_t) sc(rng);
                for (int q = 0; q < 32; ++q) blk[4 + q] = (uint8_t) by(rng);
            }
        const float tail[4] = {6.0f + e, 3.0f, 0.5f + 0.25f * e, 0.f};
        std::memcpy(b + tail_off, tail, 16);
    }
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> x((size_t) T * L);
    for (float& v : x) v = nd(rng) * (rng() % 16 == 0 ? 4.f : 1.f);

    const int W = two ? 2 : 1;
    uint8_t *d_blob, *d_a4x, *d_a4h;
    float *d_x, *d_gsx, *d_gsh, *d_h, *d_out;
    int32_t *d_bounds, *d_tok, *d_toh, *d_blk;
    CK(cudaMalloc(&d_blob, blob.size()));
    CK(cudaMalloc(&d_x, x.size() * 4));
    CK(cudaMalloc(&d_a4x, f4::token_bytes(T, (int64_t) W * L)));
    CK(cudaMalloc(&d_gsx, T * 4));
    CK(cudaMalloc(&d_a4h, f4::a4_bytes(rows, (int64_t) W * F, NE)));
    CK(cudaMalloc(&d_gsh, f4::scale_count(rows, NE) * 4));
    CK(cudaMalloc(&d_h, (size_t) rows * F * 4));
    CK(cudaMalloc(&d_out, (size_t) rows * L * 4));
    CK(cudaMalloc(&d_bounds, bounds.size() * 4));
    CK(cudaMalloc(&d_tok, tok.size() * 4));
    CK(cudaMalloc(&d_toh, (NE + 1) * 4));
    CK(cudaMalloc(&d_blk, (NE + 1) * 4));
    CK(cudaMemcpy(d_blob, blob.data(), blob.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_x, x.data(), x.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_bounds, bounds.data(), bounds.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_tok, tok.data(), tok.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemset(d_out, 0xff, (size_t) rows * L * 4));

    f4::quantize_tokens(d_x, T, L, L, d_a4x, d_gsx, nullptr, two);
    f4::plan_blocks(d_bounds, NE, d_toh, d_blk, nullptr);
    f4::Product p;
    p.base = d_blob; p.stride = stride; p.n = NE; p.blk_off = d_blk; p.blocks = f4::max_blocks(rows, NE);
    p.bounds = d_bounds; p.tile_off = d_toh; p.w_rows = 2 * F; p.w_cols = L; p.a4 = d_a4x; p.gs = d_gsx; p.ids = d_tok;
    p.tokens = T; p.swiglu = true; p.swiglu_limit = kLimit; p.tail_off = tail_off; p.dst = d_h; p.ld_dst = F; p.two = two;
    f4::run(p, nullptr);
    f4::quantize(d_h, nullptr, d_bounds, d_toh, NE, d_a4h, d_gsh, F, F, rows, nullptr, two);
    f4::Product d = p;
    d.w_off = down_off; d.w_rows = L; d.w_cols = F; d.a4 = d_a4h; d.gs = d_gsh; d.ids = nullptr; d.tokens = 0; d.swiglu = false;
    d.swiglu_limit = 0.f; d.dst = d_out; d.ld_dst = L;
    f4::run(d, nullptr);
    CK(cudaDeviceSynchronize());

    std::vector<uint8_t> a4x(f4::token_bytes(T, (int64_t) W * L));
    std::vector<float> gsx(T), h((size_t) rows * F), out((size_t) rows * L);
    CK(cudaMemcpy(a4x.data(), d_a4x, a4x.size(), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(gsx.data(), d_gsx, T * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(h.data(), d_h, h.size() * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(out.data(), d_out, out.size() * 4, cudaMemcpyDeviceToHost));

    // the token planes decoded: codes [T][W L / 2], then scales [T][W L / 16]; with two terms, logical block (b / 8) * 16 + b % 8
    // and that + 8 hold the two terms of real block b
    std::vector<double> xq((size_t) T * L, 0.0);
    double xerr = 0, xnorm = 0;
    for (int t = 0; t < T; ++t)
        for (int b = 0; b < L / 16; ++b)
            for (int u = 0; u < W; ++u) {
                const int lb = two ? (b / 8) * 16 + b % 8 + 8 * u : b;
                const uint8_t* c = a4x.data() + (size_t) t * (W * L / 2) + 8 * lb;
                const float s = ue4m3(a4x[(size_t) T * (W * L / 2) + (size_t) t * (W * L / 16) + lb]) * gsx[t];
                for (int j = 0; j < 16; ++j) xq[(size_t) t * L + 16 * b + j] += e2m1(j < 8 ? c[j] & 15u : c[j - 8] >> 4) * s;
            }
    for (size_t i = 0; i < xq.size(); ++i) { xerr += (xq[i] - x[i]) * (xq[i] - x[i]); xnorm += (double) x[i] * x[i]; }

    std::vector<float> wr(L);
    double herr = 0, hnorm = 0, derr = 0, dnorm = 0;
    int clamped = 0;
    std::vector<std::vector<float>> Wg(F, std::vector<float>(L)), Wu(F, std::vector<float>(L)), Wd(L, std::vector<float>(F));
    for (int e = 0; e < NE; ++e) {
        if (counts[e] == 0) continue;
        const uint8_t* b = blob.data() + e * stride;
        float tail[4];
        std::memcpy(tail, b + tail_off, 16);
        for (int r = 0; r < F; ++r) { row_values(b + r * kRow, L, Wg[r].data()); row_values(b + (F + r) * kRow, L, Wu[r].data()); }
        for (int r = 0; r < L; ++r) row_values(b + down_off + r * kRow, F, Wd[r].data());
        for (int i = bounds[e]; i < bounds[e + 1]; ++i) {
            const double* xr = xq.data() + (size_t) tok[i] * L;
            for (int r = 0; r < F; ++r) {
                double g = 0, u = 0;
                for (int k = 0; k < L; ++k) { g += Wg[r][k] * xr[k]; u += Wu[r][k] * xr[k]; }
                g *= tail[0]; u *= tail[1];
                clamped += g > kLimit || std::fabs(u) > kLimit;
                g = std::fmin(g, kLimit);
                u = std::fmin(std::fmax(u, -kLimit), kLimit);
                const double ref = g / (1.0 + std::exp(-g)) * u * tail[2], got = h[(size_t) i * F + r];
                herr += (ref - got) * (ref - got);
                hnorm += ref * ref;
            }
            const float* hr = h.data() + (size_t) i * F;
            for (int r = 0; r < L; ++r) {
                double s = 0;
                for (int k = 0; k < F; ++k) s += Wd[r][k] * hr[k];
                const double got = out[(size_t) i * L + r];
                derr += (s - got) * (s - got);
                dnorm += s * s;
            }
        }
    }
    const double xr = std::sqrt(xerr / xnorm), hr = std::sqrt(herr / hnorm), dr = std::sqrt(derr / dnorm);
    const double dmax = two ? 0.02 : 0.15;
    const bool ok = hr < 1e-4 && dr < dmax && clamped > 0 && std::isfinite(dr);
    std::printf("%s: x rounding %.4f, gate/up+swiglu vs exact %.2e (clamped %d), down vs unrounded H %.4f (max %.2f): %s\n",
                two ? "two-term" : "one-term", xr, hr, clamped, dr, dmax, ok ? "ok" : "FAIL");
    cudaFree(d_blob); cudaFree(d_x); cudaFree(d_a4x); cudaFree(d_gsx); cudaFree(d_a4h); cudaFree(d_gsh); cudaFree(d_h);
    cudaFree(d_out); cudaFree(d_bounds); cudaFree(d_tok); cudaFree(d_toh); cudaFree(d_blk);
    return ok ? 0 : 1;
}

}  // namespace

int main() {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) return 77;
    return run(false) | run(true);
}

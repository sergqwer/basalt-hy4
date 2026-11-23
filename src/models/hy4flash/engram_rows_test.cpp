// src/models/hy4flash/engram_rows_test.cpp - engram_rows (the card's decode and norm of the Engram rows) against the host step's
// loop in session.cpp, bit for bit, on random rows, norms and validity: E4M3 codes (type 43), and int8 values with an fp16
// row scale (type 46, I8_ROW_F16).
#include "basalt/models/hy4flash/kernels.hpp"
#include "engram.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace h4 = basalt::models::hy4flash;

static bool run(bool scaled) {
    const int R = 300, NC = 16, W = 96, RB = W + (scaled ? 2 : 0);
    std::mt19937 rng(scaled ? 11 : 9);
    std::vector<float> lut(256);
    for (int i = 0; i < 256; ++i) lut[(size_t) i] = h4::engram_code_to_f32((uint8_t) i, scaled);
    std::vector<uint8_t> raw((size_t) R * NC * RB);
    for (auto& b : raw) b = (uint8_t) (rng() & 0xff);
    if (!scaled) {
        for (int k = 0; k < 64; ++k) raw[(size_t) (rng() % raw.size())] = 0x7f;   // NaN codes too
    } else {
        std::uniform_int_distribution<int> ex(0x1000, 0x2c00);   // fp16 scales from ~6e-5 to ~0.06, and some zero
        for (size_t r = 0; r < (size_t) R * NC; ++r) {
            const uint16_t h = (rng() % 50 == 0) ? 0 : (uint16_t) ex(rng);
            raw[r * RB + W] = (uint8_t) (h & 0xff);
            raw[r * RB + W + 1] = (uint8_t) (h >> 8);
        }
    }
    std::vector<float> nrm((size_t) NC * W), valid(R);
    std::normal_distribution<float> nd(1.f, 0.3f);
    for (auto& v : nrm) v = nd(rng);
    for (int t = 0; t < R; ++t) valid[(size_t) t] = (rng() % 7 == 0) ? 0.f : 1.f;
    std::vector<float> host((size_t) R * NC * W);
    for (int t = 0; t < R; ++t)
        for (int c = 0; c < NC; ++c) {   // session.cpp engram_step, as written there
            float* row = host.data() + ((size_t) t * NC + c) * W;
            const uint8_t* src = raw.data() + ((size_t) t * NC + c) * RB;
            const float sc = scaled ? h4::engram_row_scale(src, W) : 1.f;
            float ss = 0.f;
            for (int d = 0; d < W; ++d) {
                row[d] = lut[src[d]] * sc;
                ss += row[d] * row[d];
            }
            const float r = 1.0f / std::sqrt(ss / (float) W + 1.1920929e-07f);
            for (int d = 0; d < W; ++d) row[d] = row[d] * r * nrm[(size_t) c * W + d] * valid[(size_t) t];
        }
    uint8_t* d_raw;
    float *d_lut, *d_nrm, *d_valid, *d_out;
    cudaMalloc(&d_raw, raw.size());
    cudaMalloc(&d_lut, 1024);
    cudaMalloc(&d_nrm, nrm.size() * 4);
    cudaMalloc(&d_valid, R * 4);
    cudaMalloc(&d_out, host.size() * 4);
    cudaMemcpy(d_raw, raw.data(), raw.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_lut, lut.data(), 1024, cudaMemcpyHostToDevice);
    cudaMemcpy(d_nrm, nrm.data(), nrm.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_valid, valid.data(), R * 4, cudaMemcpyHostToDevice);
    h4::engram_rows(d_raw, d_lut, d_nrm, d_valid, R, NC, W, RB, scaled, d_out, nullptr);
    std::vector<float> dev(host.size());
    const cudaError_t ce = cudaMemcpy(dev.data(), d_out, dev.size() * 4, cudaMemcpyDeviceToHost);
    size_t diff = 0;
    for (size_t i = 0; i < host.size(); ++i)   // NaN: the card's is the canonical one
        diff += (std::isnan(host[i]) && std::isnan(dev[i])) ? 0 : std::memcmp(&host[i], &dev[i], 4) != 0;
    std::printf("engram_rows_test %s: %zu of %zu values differ in their bits%s\n", scaled ? "int8 + fp16 row scale" : "E4M3", diff,
                host.size(), ce == cudaSuccess ? "" : " (CUDA error)");
    cudaFree(d_raw);
    cudaFree(d_lut);
    cudaFree(d_nrm);
    cudaFree(d_valid);
    cudaFree(d_out);
    return diff == 0 && ce == cudaSuccess;
}

int main() {
    const bool a = run(false), b = run(true);
    return a && b ? 0 : 1;
}

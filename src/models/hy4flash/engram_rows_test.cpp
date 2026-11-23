// src/models/hy4flash/engram_rows_test.cpp - engram_rows (the card's decode and norm of the Engram codes) against the host step's
// loop in session.cpp, bit for bit, on random codes, norms and validity.
#include "basalt/models/hy4flash/kernels.hpp"
#include "engram.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace h4 = basalt::models::hy4flash;

int main() {
    const int R = 300, NC = 16, W = 96;
    std::mt19937 rng(9);
    std::vector<float> lut(256);
    for (int i = 0; i < 256; ++i) lut[(size_t) i] = h4::e4m3_to_f32((uint8_t) i);
    std::vector<uint8_t> raw((size_t) R * NC * W);
    for (auto& b : raw) b = (uint8_t) (rng() & 0xff);
    for (int k = 0; k < 64; ++k) raw[(size_t) (rng() % raw.size())] = 0x7f;   // NaN codes too
    std::vector<float> nrm((size_t) NC * W), valid(R);
    std::normal_distribution<float> nd(1.f, 0.3f);
    for (auto& v : nrm) v = nd(rng);
    for (int t = 0; t < R; ++t) valid[(size_t) t] = (rng() % 7 == 0) ? 0.f : 1.f;
    std::vector<float> host((size_t) R * NC * W);
    for (int t = 0; t < R; ++t)
        for (int c = 0; c < NC; ++c) {   // session.cpp engram_step, as written there
            float* row = host.data() + ((size_t) t * NC + c) * W;
            const uint8_t* src = raw.data() + ((size_t) t * NC + c) * W;
            float ss = 0.f;
            for (int d = 0; d < W; ++d) {
                row[d] = lut[src[d]];
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
    h4::engram_rows(d_raw, d_lut, d_nrm, d_valid, R, NC, W, d_out, nullptr);
    std::vector<float> dev(host.size());
    const cudaError_t ce = cudaMemcpy(dev.data(), d_out, dev.size() * 4, cudaMemcpyDeviceToHost);
    size_t diff = 0;
    for (size_t i = 0; i < host.size(); ++i)   // NaN: the card's is the canonical one
        diff += (std::isnan(host[i]) && std::isnan(dev[i])) ? 0 : std::memcmp(&host[i], &dev[i], 4) != 0;
    std::printf("engram_rows_test: %zu of %zu values differ in their bits%s\n", diff, host.size(), ce == cudaSuccess ? "" : " (CUDA error)");
    return diff == 0 && ce == cudaSuccess ? 0 : 1;
}

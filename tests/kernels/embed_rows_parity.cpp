// tests/kernels/embed_rows_parity.cpp - the card's token-embedding rows (kernels/elementwise.hpp embed_rows_from_mapped)
// bit for bit against the host's: Q8_0 blocks through artifact/dequant.hpp's dequantize_q8_0 (every fp16 scale class:
// zero, subnormal, normal, the largest finite, both signs) and BF16 rows widened (subnormals, zeros, both signs).
#include "basalt/artifact/dequant.hpp"
#include "basalt/kernels/elementwise.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

static int run(int type, int rows, int width, std::mt19937& rng) {
    const size_t rb = type == 8 ? (size_t) width / 32 * 34 : (size_t) width * 2;
    uint8_t* src = nullptr;
    float* dst = nullptr;
    if (cudaMallocHost(&src, rb * rows) != cudaSuccess || cudaMalloc(&dst, (size_t) rows * width * 4) != cudaSuccess) {
        std::printf("FAIL: no memory\n");
        return 1;
    }
    const uint16_t scales[] = {0x0000, 0x8000, 0x0001, 0x83ff, 0x0400, 0x3c00, 0xbc00, 0x2e66, 0x7bff, 0xfbff, 0x1234};
    for (int r = 0; r < rows; ++r) {
        uint8_t* row = src + (size_t) r * rb;
        if (type == 8) {
            for (int b = 0; b < width / 32; ++b) {
                uint8_t* blk = row + b * 34;
                const uint16_t d = (r + b) % 3 == 0 ? scales[(r * 7 + b) % (sizeof scales / 2)] : (uint16_t) (rng() & 0x7bff);
                blk[0] = (uint8_t) d;
                blk[1] = (uint8_t) (d >> 8);
                for (int j = 0; j < 32; ++j) blk[2 + j] = (uint8_t) (j == 0 ? 0x80 : j == 1 ? 0x7f : rng());
            }
        } else {
            uint16_t* v = (uint16_t*) row;
            for (int i = 0; i < width; ++i) {
                uint16_t x = (uint16_t) rng();
                if ((x & 0x7f80) == 0x7f80) x &= 0xbfff;   // no inf / NaN in an embedding
                v[i] = i % 97 == 0 ? (uint16_t) (i & 1 ? 0x8000 : 0x0001) : x;
            }
        }
    }
    basalt::kernels::embed_rows_from_mapped(dst, src, rows, width, type, nullptr);
    std::vector<float> got((size_t) rows * width), want((size_t) rows * width);
    if (cudaMemcpy(got.data(), dst, got.size() * 4, cudaMemcpyDeviceToHost) != cudaSuccess) {
        std::printf("FAIL: copy back\n");
        return 1;
    }
    for (int r = 0; r < rows; ++r) {
        const uint8_t* row = src + (size_t) r * rb;
        float* out = want.data() + (size_t) r * width;
        if (type == 8) {
            for (int b = 0; b < width / 32; ++b) basalt::dequantize_q8_0(row + b * 34, out + b * 32);
        } else {
            for (int i = 0; i < width; ++i) {
                const uint32_t u = (uint32_t) ((const uint16_t*) row)[i] << 16;
                std::memcpy(out + i, &u, 4);
            }
        }
    }
    size_t bad = 0;
    for (size_t i = 0; i < got.size(); ++i) bad += std::memcmp(&got[i], &want[i], 4) != 0;
    std::printf("%s %d rows x %d: %zu of %zu values differ\n", type == 8 ? "Q8_0" : "BF16", rows, width, bad, got.size());
    cudaFreeHost(src);
    cudaFree(dst);
    return bad != 0;
}

int main() {
    std::mt19937 rng(7);
    int fails = 0;
    fails += run(8, 300, 3072, rng);   // Hy4's width
    fails += run(8, 1, 32, rng);
    fails += run(30, 300, 3072, rng);
    fails += run(30, 3, 96, rng);
    if (fails) return 1;
    std::printf("embed_rows_parity: ok\n");
    return 0;
}

// tests/kernels/window_plan_test.cpp - window_plan (the card's own plan of a decode group's held experts) against the host's
// plan (session.cpp make_plan): the same groups in the same order, the same entries, for random routings and tables.
#include "basalt/kernels/elementwise.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <random>
#include <vector>

namespace bk = basalt::kernels;

int main() {
    constexpr int NE = 512, K = 16, MAXN = 128;
    unsigned long long *d_tab = nullptr, *d_ptr = nullptr;
    int32_t *d_ids = nullptr, *d_start = nullptr, *d_ng = nullptr, *d_dst = nullptr, *d_tok = nullptr;
    cudaMalloc(&d_tab, NE * 8);
    cudaMalloc(&d_ptr, MAXN * 8);
    cudaMalloc(&d_ids, MAXN * 4);
    cudaMalloc(&d_start, (MAXN + 2) * 4);
    cudaMalloc(&d_ng, 16);
    cudaMalloc(&d_dst, MAXN * 4);
    cudaMalloc(&d_tok, MAXN * 4);
    std::mt19937 rng(5);
    int bad = 0, cases = 0;
    for (int it = 0; it < 2000; ++it) {
        const int rows = 1 + (int) (rng() % 8), n = rows * K;
        std::vector<unsigned long long> tab(NE, 0);
        const int held = (int) (rng() % NE);
        for (int e = 0; e < NE; ++e)
            if ((int) (rng() % NE) < held) tab[e] = 0x100000000ull + (unsigned long long) e * 4096;
        std::vector<int32_t> ids(n);
        const int span = 16 + (int) (rng() % (NE - 16));
        for (int r = 0; r < n; ++r) ids[r] = (rng() % 50 == 0) ? -1 : (int32_t) (rng() % span);
        // the host's plan: distinct experts in first-routed order, entries in routed order
        std::vector<int> uniq;
        std::vector<std::vector<int>> rows_of;
        for (int r = 0; r < n; ++r) {
            const int e = ids[r];
            if (e < 0 || tab[e] == 0) continue;
            size_t u = 0;
            while (u < uniq.size() && uniq[u] != e) ++u;
            if (u == uniq.size()) {
                uniq.push_back(e);
                rows_of.emplace_back();
            }
            rows_of[u].push_back(r);
        }
        std::vector<unsigned long long> hptr;
        std::vector<int32_t> hstart, hdst, htok;
        for (size_t u = 0; u < uniq.size(); ++u) {
            hptr.push_back(tab[uniq[u]]);
            hstart.push_back((int32_t) hdst.size());
            for (int r : rows_of[u]) {
                hdst.push_back(r);
                htok.push_back(r / K);
            }
        }
        hstart.push_back((int32_t) hdst.size());
        cudaMemcpy(d_tab, tab.data(), NE * 8, cudaMemcpyHostToDevice);
        cudaMemcpy(d_ids, ids.data(), n * 4, cudaMemcpyHostToDevice);
        bk::window_plan(d_ids, d_tab, n, K, NE, d_ptr, d_start, d_ng, d_dst, d_tok, nullptr);
        int32_t ng = -1;
        std::vector<unsigned long long> gptr(MAXN);
        std::vector<int32_t> gstart(MAXN + 2), gdst(MAXN), gtok(MAXN);
        cudaMemcpy(&ng, d_ng, 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(gptr.data(), d_ptr, MAXN * 8, cudaMemcpyDeviceToHost);
        cudaMemcpy(gstart.data(), d_start, (MAXN + 2) * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(gdst.data(), d_dst, MAXN * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(gtok.data(), d_tok, MAXN * 4, cudaMemcpyDeviceToHost);
        bool ok = ng == (int) uniq.size();
        for (int g = 0; ok && g < ng; ++g) ok = gptr[g] == hptr[g] && gstart[g] == hstart[g];
        ok = ok && gstart[ng] == hstart[ng];
        for (size_t i = 0; ok && i < hdst.size(); ++i) ok = gdst[i] == hdst[i] && gtok[i] == htok[i];
        ++cases;
        if (!ok && bad++ < 5) std::printf("case %d (rows %d): card %d groups, host %zu\n", it, rows, ng, uniq.size());
    }
    const cudaError_t ce = cudaDeviceSynchronize();
    std::printf("window_plan_test: %d of %d cases differ%s\n", bad, cases, ce == cudaSuccess ? "" : " (CUDA error)");
    return bad == 0 && ce == cudaSuccess ? 0 : 1;
}

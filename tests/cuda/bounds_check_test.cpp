// bounds_check_test - HY4_BOUNDS_CHECK (core/bounds_check.hpp) trips each check on purpose:
//   - check_cells: a span row past the limit (cut to it), a listed cell of -1 and one at the limit (set to 0), rows that are
//     in range left as they were;
//   - check_list: a drafter-style list with one cell past the limit;
//   - host_violation and host_range (the tier's slot_ok: a copy target past the mapped arena);
//   - each one reported once through take() with its site, layer, index and limit, and no CUDA error after any of them.
// `bounds_check_test off` (run with HY4_BOUNDS_CHECK=0): nothing is allocated, no check launches anything, take() finds
// nothing. Exit 77 without a CUDA device.
#include "basalt/core/bounds_check.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {
namespace bd = basalt::core::bounds;

int g_fail = 0;
void expect(bool c, const char* what) {
    if (c) return;
    std::fprintf(stderr, "bounds_check_test: FAIL: %s\n", what);
    ++g_fail;
}
// the one violation take() returns names the site, the layer, the index and the limit
void expect_report(const char* site, int layer, long long index, long long limit) {
    std::string m;
    const bool got = bd::take(m);
    char want[200];
    std::snprintf(want, sizeof want, "%s out of range (site", site);
    char tail[120];
    std::snprintf(tail, sizeof tail, "layer %d): index %lld, limit %lld", layer, index, limit);
    if (!got || m.find(want) == std::string::npos || m.find(tail) == std::string::npos) {
        std::fprintf(stderr, "bounds_check_test: FAIL: wanted '%s ... %s', got '%s'\n", want, tail, got ? m.c_str() : "(none)");
        ++g_fail;
    }
    std::string again;
    expect(!bd::take(again), "take() clears the report");
}
bool sync_ok() { return cudaDeviceSynchronize() == cudaSuccess && cudaGetLastError() == cudaSuccess; }
}  // namespace

int main(int argc, char** argv) {
    int n_dev = 0;
    if (cudaGetDeviceCount(&n_dev) != cudaSuccess || n_dev == 0) {
        std::printf("bounds_check_test: no CUDA device, skipped\n");
        return 77;
    }
    const bool off = argc > 1 && std::strcmp(argv[1], "off") == 0;
    if (!off) {
#ifdef _WIN32
        _putenv_s("HY4_BOUNDS_CHECK", "1");
#else
        setenv("HY4_BOUNDS_CHECK", "1", 1);
#endif
    }
    if (off) {
        expect(!bd::enabled(), "HY4_BOUNDS_CHECK=0 leaves the checks off");
        expect(bd::device_report() == nullptr, "no report while off");
        bd::Limit lim;
        expect(!lim.open() && lim.d == nullptr, "no limit word while off");
        bd::check_list(nullptr, 4, nullptr, bd::kDraftCells, 0, nullptr);
        bd::host_violation(bd::kTierPromote, 1, 2, 3);
        std::string m;
        expect(!bd::take(m), "nothing to take while off");
        expect(sync_ok(), "no CUDA error");
        if (g_fail) return 1;
        std::printf("bounds_check_test off: nothing allocated, launched or reported\n");
        return 0;
    }
    expect(bd::enabled(), "HY4_BOUNDS_CHECK=1 turns the checks on");
    bd::Limit lim;
    expect(lim.open() && lim.d != nullptr, "the limit word opens");
    lim.set(100);

    // check_cells: 4 rows, stride 8. Row 0 a span 90 + 10 (in range), row 1 a span 95 + 10 (past: cut to 5), row 2 listed
    // {3, -1, 7}, row 3 listed {100, 99} with fill[3] = 0; fill[1] > 0 makes row 1 a span although cells are set.
    const int N = 4, S = 8;
    std::vector<int32_t> firsts = {90, 95, 0, 0}, counts = {10, 10, 3, 2}, fill = {1, 10, 0, 0}, cells(N * S, 5);
    cells[2 * S + 0] = 3; cells[2 * S + 1] = -1; cells[2 * S + 2] = 7;
    cells[3 * S + 0] = 100; cells[3 * S + 1] = 99;
    int32_t *d_f = nullptr, *d_c = nullptr, *d_l = nullptr, *d_fill = nullptr;
    cudaMalloc(&d_f, N * 4); cudaMalloc(&d_c, N * 4); cudaMalloc(&d_fill, N * 4); cudaMalloc(&d_l, N * S * 4);
    cudaMemcpy(d_f, firsts.data(), N * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_fill, fill.data(), N * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_l, cells.data(), N * S * 4, cudaMemcpyHostToDevice);

    // one violation at a time: the report keeps the first
    auto run_rows = [&](int t0, int n) {
        cudaMemcpy(d_c, counts.data(), N * 4, cudaMemcpyHostToDevice);
        bd::check_cells(d_f + t0, d_c + t0, d_l + (size_t) t0 * S, S, d_fill + t0, n, lim.d, bd::kAttendCells, 7, nullptr);
        expect(sync_ok(), "no CUDA error after check_cells");
    };
    run_rows(0, 1);
    std::string m;
    expect(!bd::take(m), "a span in range reports nothing");
    run_rows(1, 1);
    expect_report("attention cells", 7, 104, 100);
    std::vector<int32_t> hc(N);
    cudaMemcpy(hc.data(), d_c, N * 4, cudaMemcpyDeviceToHost);
    expect(hc[1] == 5, "a span past the limit is cut to it");
    run_rows(2, 1);
    expect_report("attention cells", 7, -1, 100);
    run_rows(3, 1);
    expect_report("attention cells", 7, 100, 100);
    std::vector<int32_t> hl(N * S);
    cudaMemcpy(hl.data(), d_l, N * S * 4, cudaMemcpyDeviceToHost);
    expect(hl[2 * S + 0] == 3 && hl[2 * S + 1] == 0 && hl[2 * S + 2] == 7, "a listed -1 is set to 0, the others kept");
    expect(hl[3 * S + 0] == 0 && hl[3 * S + 1] == 99, "a listed cell at the limit is set to 0");
    expect(hl[0] == 5, "a span row's cells are not touched");

    // check_list: the drafter's selected cells
    std::vector<int32_t> list = {0, 50, 99, 250, 1};
    int32_t* d_list = nullptr;
    cudaMalloc(&d_list, list.size() * 4);
    cudaMemcpy(d_list, list.data(), list.size() * 4, cudaMemcpyHostToDevice);
    bd::check_list(d_list, (int) list.size(), lim.d, bd::kDraftCells, 2, nullptr);
    expect(sync_ok(), "no CUDA error after check_list");
    expect_report("drafter selected cells", 2, 250, 100);
    cudaMemcpy(list.data(), d_list, list.size() * 4, cudaMemcpyDeviceToHost);
    expect(list[3] == 0 && list[2] == 99, "the cell past the limit is set to 0");
    lim.set(1000);   // a limit the host moves is what the next launch reads
    cudaMemcpy(d_list, std::vector<int32_t>{250, 999}.data(), 8, cudaMemcpyHostToDevice);
    bd::check_list(d_list, 2, lim.d, bd::kDraftCells, 2, nullptr);
    expect(sync_ok(), "no CUDA error");
    expect(!bd::take(m), "in range under the moved limit");

    // the host sites: the indexer's pools, the drafter's positions, a tier copy past the mapped arena
    bd::host_violation(bd::kIndexerPools, 4, 300, 257);
    expect_report("indexer pooled keys", 4, 300, 257);
    static uint8_t arena[4096];
    expect(bd::host_range(bd::kTierPromote, 9, arena + 1024, 1024, arena, 2048), "a copy inside the mapped part passes");
    expect(!bd::take(m), "and reports nothing");
    expect(!bd::host_range(bd::kTierPromote, 9, arena + 1536, 1024, arena, 2048), "a copy past the mapped part is refused");
    expect_report("tier promotion copy", 9, 1536, 2048);
    expect(!bd::host_range(bd::kTierFetch, 3, arena, 16, nullptr, 0), "no arena: refused");
    std::string m2;
    expect(bd::take(m2), "reported");
    // the first violation wins until it is taken
    bd::host_violation(bd::kTierMove, 1, 11, 10);
    bd::host_violation(bd::kTierSource, 2, 22, 20);
    expect_report("tier lend move", 1, 11, 10);

    cudaFree(d_f); cudaFree(d_c); cudaFree(d_fill); cudaFree(d_l); cudaFree(d_list);
    if (g_fail) {
        std::fprintf(stderr, "bounds_check_test: %d failures\n", g_fail);
        return 1;
    }
    std::printf("bounds_check_test: every check tripped, reported once and stepped around (no CUDA error)\n");
    return 0;
}

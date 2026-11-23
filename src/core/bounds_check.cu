// src/core/bounds_check.cu - see bounds_check.hpp.
#include "basalt/core/bounds_check.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>

namespace basalt::core::bounds {
namespace {

std::mutex g_mu;
Report* g_h = nullptr;   // the report: host view
Report* g_d = nullptr;   // device view
bool g_tried = false;

void open_report() {
    std::lock_guard<std::mutex> g(g_mu);
    if (g_tried) return;
    g_tried = true;
    void* p = nullptr;
    if (cudaHostAlloc(&p, sizeof(Report), cudaHostAllocMapped | cudaHostAllocPortable) != cudaSuccess) {
        cudaGetLastError();
        std::fprintf(stderr, "basalt: HY4_BOUNDS_CHECK: cannot allocate the mapped report; the checks are off\n");
        return;
    }
    std::memset(p, 0, sizeof(Report));
    void* d = nullptr;
    if (cudaHostGetDevicePointer(&d, p, 0) != cudaSuccess) {
        cudaGetLastError();
        cudaFreeHost(p);
        return;
    }
    g_h = (Report*) p;
    g_d = (Report*) d;
}

__device__ void record(Report* r, uint32_t site, int layer, int64_t index, int64_t limit) {
    if (atomicCAS_system(&r->claim, 0u, 1u) != 0u) return;
    r->site = site;
    r->layer = layer;
    r->index = index;
    r->limit = limit;
    __threadfence_system();
    atomicExch_system(&r->ready, 1u);
}

__global__ void cells_kernel(const int32_t* firsts, int32_t* counts, int32_t* cells, int stride, const int32_t* fill, int n,
                             const int64_t* limit, Report* r, uint32_t site, int layer) {
    const int t = blockIdx.x;
    if (t >= n) return;
    const int64_t lim = *limit;
    const int c = counts[t];
    const bool listed = cells && !(fill && fill[t] > 0);
    if (!listed) {   // a span firsts[t] .. firsts[t] + c
        if (threadIdx.x != 0) return;
        const int64_t f = firsts ? firsts[t] : 0;
        if (f < 0 || f + c > lim) {
            record(r, site, layer, f < 0 ? f : f + c - 1, lim);
            counts[t] = f < 0 ? 0 : (int32_t) (lim > f ? lim - f : 0);
        }
        return;
    }
    int32_t* row = cells + (size_t) t * stride;
    for (int j = threadIdx.x; j < c; j += blockDim.x) {
        const int32_t v = row[j];
        if (v < 0 || v >= lim) {
            record(r, site, layer, v, lim);
            row[j] = 0;
        }
    }
}

__global__ void list_kernel(int32_t* cells, int n, const int64_t* limit, Report* r, uint32_t site, int layer) {
    const int64_t lim = *limit;
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < n; j += gridDim.x * blockDim.x) {
        const int32_t v = cells[j];
        if (v < 0 || v >= lim) {
            record(r, site, layer, v, lim);
            cells[j] = 0;
        }
    }
}

}  // namespace

const char* site_name(uint32_t site) {
    switch (site) {
        case kAttendCells: return "attention cells";
        case kIndexerPools: return "indexer pooled keys";
        case kDraftCells: return "drafter selected cells";
        case kDraftPositions: return "drafter positions";
        case kTierPromote: return "tier promotion copy";
        case kTierFetch: return "tier fetch copy";
        case kTierMove: return "tier lend move";
        case kTierSource: return "copy out of a tier slot";
        default: return "unknown site";
    }
}

bool enabled() {
    static const bool on = [] {
        const char* v = std::getenv("HY4_BOUNDS_CHECK");
        return v != nullptr && v[0] != '\0' && v[0] != '0';
    }();
    return on;
}

Report* device_report() {
    if (!enabled()) return nullptr;
    if (!g_tried) open_report();
    return g_d;
}

bool Limit::open() {
    if (h || !device_report()) return h != nullptr;
    void* p = nullptr;
    void* dp = nullptr;
    if (cudaHostAlloc(&p, sizeof(int64_t), cudaHostAllocMapped | cudaHostAllocPortable) != cudaSuccess ||
        cudaHostGetDevicePointer(&dp, p, 0) != cudaSuccess) {
        cudaGetLastError();
        if (p) cudaFreeHost(p);
        return false;
    }
    h = (int64_t*) p;
    *h = INT64_MAX;
    d = (const int64_t*) dp;
    return true;
}

void Limit::close() {
    if (h) cudaFreeHost(h);
    h = nullptr;
    d = nullptr;
}

void host_violation(Site site, int layer, int64_t index, int64_t limit) {
    if (!device_report()) return;
    std::lock_guard<std::mutex> g(g_mu);
    volatile Report* r = g_h;
    if (r->claim != 0u) return;
    r->claim = 1u;
    r->site = site;
    r->layer = layer;
    r->index = index;
    r->limit = limit;
    r->ready = 1u;
}

bool host_range(Site site, int layer, const void* lo, uint64_t n, const void* base, uint64_t mapped) {
    const uintptr_t a = (uintptr_t) lo, b = (uintptr_t) base;
    if (base != nullptr && a >= b && a - b <= mapped && n <= mapped - (a - b)) return true;
    host_violation(site, layer, (int64_t) (a - b), (int64_t) mapped);
    return false;
}

void check_cells(const int32_t* firsts, int32_t* counts, int32_t* cells, int stride, const int32_t* fill, int n,
                 const int64_t* limit, Site site, int layer, void* stream) {
    Report* r = device_report();
    if (!r || !limit || n <= 0 || !counts) return;
    cells_kernel<<<(unsigned) n, 128, 0, (cudaStream_t) stream>>>(firsts, counts, cells, stride, fill, n, limit, r, site, layer);
}

void check_list(int32_t* cells, int n, const int64_t* limit, Site site, int layer, void* stream) {
    Report* r = device_report();
    if (!r || !limit || n <= 0 || !cells) return;
    const int blocks = (n + 255) / 256 < 64 ? (n + 255) / 256 : 64;
    list_kernel<<<(unsigned) blocks, 256, 0, (cudaStream_t) stream>>>(cells, n, limit, r, site, layer);
}

bool take(std::string& msg) {
    if (!g_h) return false;
    std::lock_guard<std::mutex> g(g_mu);
    volatile Report* r = g_h;
    if (r->ready == 0u) return false;
    char buf[256];
    std::snprintf(buf, sizeof buf, "HY4_BOUNDS_CHECK: %s out of range (site %u, layer %d): index %lld, limit %lld",
                  site_name(r->site), (unsigned) r->site, (int) r->layer, (long long) r->index, (long long) r->limit);
    msg = buf;
    r->ready = 0u;
    r->claim = 0u;
    return true;
}

}  // namespace basalt::core::bounds

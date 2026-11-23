// src/core/elastic_array.cpp - see elastic_array.hpp.
#include "basalt/core/elastic_array.hpp"

#include "basalt/core/device.hpp"
#include "basalt/core/vmm.hpp"

#include <cuda_runtime.h>

#include <algorithm>

namespace basalt::core {

ElasticArray::~ElasticArray() {
    if (!base_) return;
    if (plain_) {
        cudaFree(base_);
        return;
    }
    cudaDeviceSynchronize();
    for (const Extent& e : extents_) vmm::unmap(base_ + e.off, e.bytes, e.handle);
    vmm::release_range(base_, reserved_);
}

bool ElasticArray::open(uint64_t bytes, std::string& err) {
    if (bytes == 0) bytes = 1;
    if (vmm::supported()) {
        gran_ = vmm::granularity();
        if (gran_) {
            reserved_ = (bytes + gran_ - 1) / gran_ * gran_;
            base_ = vmm::reserve(reserved_);
            if (base_) return true;
        }
    }
    void* p = nullptr;
    if (cudaMalloc(&p, bytes) != cudaSuccess) {
        err = "out of device memory for the array";
        return false;
    }
    cudaMemset(p, 0, bytes);
    base_ = (uint8_t*) p;
    reserved_ = backed_ = bytes;
    plain_ = true;
    return true;
}

bool ElasticArray::ensure(uint64_t bytes, uint64_t keep_free, std::string& err) {
    if (bytes <= backed_) return true;
    if (bytes > reserved_) {
        err = "the array is full";
        return false;
    }
    // grow in steps of a few granules so a long generation maps rarely
    const uint64_t step = gran_ * 4;
    const uint64_t want = std::min(reserved_, (bytes + step - 1) / step * step);
    const uint64_t n = want - backed_;
    if (keep_free > 0) {
        uint64_t free_b = 0, total_b = 0;
        std::string why;
        if (!device_memory(-1, free_b, total_b, why)) {
            err = "cannot read the free device memory: " + why;
            return false;
        }
        if (free_b < n + keep_free) {
            err = "the VRAM reserve of " + std::to_string(keep_free >> 20) + " MiB stays free";
            return false;
        }
    }
    const uint64_t piece = piece_ ? piece_ : n;
    for (uint64_t off = backed_; off < want;) {
        const uint64_t m = std::min(piece, want - off);
        const unsigned long long h = vmm::map(base_ + off, m);
        if (!h) {
            err = "out of device memory for the array";
            backed_ = off;
            return false;
        }
        cudaMemset(base_ + off, 0, m);
        extents_.push_back({off, m, h});
        off += m;
        backed_ = off;
    }
    return true;
}

void ElasticArray::trim(uint64_t bytes) {
    if (plain_ || !base_ || extents_.empty() || extents_.back().off < bytes) return;
    cudaDeviceSynchronize();
    while (!extents_.empty() && extents_.back().off >= bytes) {
        const Extent& e = extents_.back();
        vmm::unmap(base_ + e.off, e.bytes, e.handle);
        if (e.host) host_ -= e.bytes;
        extents_.pop_back();
    }
    backed_ = extents_.empty() ? 0 : extents_.back().off + extents_.back().bytes;
}

namespace {
uint64_t fnv(const uint8_t* dev, uint64_t n) {
    std::vector<uint8_t> h(n);
    cudaMemcpy(h.data(), dev, n, cudaMemcpyDefault);
    uint64_t x = 1469598103934665603ull;
    for (uint8_t c : h) x = (x ^ c) * 1099511628211ull;
    return x;
}
}  // namespace

uint64_t ElasticArray::device_bytes_in(uint64_t from, uint64_t to) const {
    uint64_t n = 0;
    for (const Extent& e : extents_)
        if (!e.host && e.off >= from && e.off + e.bytes <= to) n += e.bytes;
    return n;
}

bool ElasticArray::relocate(uint64_t from, uint64_t to, bool host, uint64_t& moved, std::string& err, bool check) {
    moved = 0;
    if (plain_) {
        err = "the array has no virtual memory";
        return false;
    }
    cudaDeviceSynchronize();   // nothing may read or write an allocation while it moves
    for (Extent& e : extents_) {
        if (e.host == host || e.off < from || e.off + e.bytes > to) continue;
        uint8_t* va = base_ + e.off;
        const uint64_t before = check ? fnv(va, e.bytes) : 0;
        const unsigned long long h = host ? vmm::create_host(e.bytes) : vmm::create(e.bytes);
        if (!h) {
            err = host ? "host VMM refused " + std::to_string(e.bytes >> 20) + " MiB" : "no device memory for " + std::to_string(e.bytes >> 20) + " MiB";
            return false;
        }
        // the new memory under a scratch range first: the copy lands (synchronized) before the old memory goes
        uint8_t* tmp = vmm::reserve(e.bytes);
        if (!tmp || !vmm::map_handle(tmp, e.bytes, h)) {
            if (tmp) vmm::release_range(tmp, e.bytes);
            vmm::release(h);
            err = "cannot map the moved memory";
            return false;
        }
        const bool copied = cudaMemcpy(tmp, va, e.bytes, cudaMemcpyDefault) == cudaSuccess && cudaDeviceSynchronize() == cudaSuccess;
        if (!copied) {
            vmm::unmap_only(tmp, e.bytes);
            vmm::release_range(tmp, e.bytes);
            vmm::release(h);
            err = "the copy of a moved allocation failed";
            return false;
        }
        vmm::unmap(va, e.bytes, e.handle);   // the old memory goes back to its owner
        if (!vmm::map_handle(va, e.bytes, h)) {
            // the array keeps its bytes: fresh device memory at the old addresses, the copy back from the scratch range
            const unsigned long long d = vmm::map(va, e.bytes);
            const bool back = d && cudaMemcpy(va, tmp, e.bytes, cudaMemcpyDefault) == cudaSuccess && cudaDeviceSynchronize() == cudaSuccess;
            vmm::unmap_only(tmp, e.bytes);
            vmm::release_range(tmp, e.bytes);
            vmm::release(h);
            e.handle = d;
            if (e.host) host_ -= e.bytes;
            e.host = false;
            err = back ? "cannot map the moved memory in place" : "an allocation of the array was lost in a move";
            return false;
        }
        vmm::unmap_only(tmp, e.bytes);
        vmm::release_range(tmp, e.bytes);
        e.handle = h;   // the creation's reference, as map() keeps it: unmap() drops it with the mapping at va
        e.host = host;
        host_ = host ? host_ + e.bytes : host_ - e.bytes;
        moved += e.bytes;
        if (check && fnv(va, e.bytes) != before) {
            err = "an allocation changed in its move (hash mismatch)";
            return false;
        }
    }
    return true;
}

}  // namespace basalt::core

// include/basalt/core/elastic_array.hpp - a device array that reserves its whole address range up front and backs it with
// physical memory as it is used. The address never moves, so kernel arguments and captured graphs stay valid.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace basalt::core {

class ElasticArray {
public:
    ElasticArray() = default;
    ~ElasticArray();
    ElasticArray(const ElasticArray&) = delete;
    ElasticArray& operator=(const ElasticArray&) = delete;

    /// Reserves `bytes` of address space. Without virtual memory management the whole array is allocated at once.
    bool open(uint64_t bytes, std::string& err);
    /// Backs at least the first `bytes` bytes (rounded up to the allocation granularity; the new bytes are zeroed).
    /// False with `err` when the device has no memory for them with `keep_free` left; what was backed stays.
    bool ensure(uint64_t bytes, uint64_t keep_free, std::string& err);
    /// Gives back the whole allocations that lie past `bytes` (the device is synchronised first); the address range stays.
    void trim(uint64_t bytes);
    /// Maps later growth in allocations of at most `bytes` (a multiple of the granularity; 0: one per ensure), so relocate()
    /// can move a part of the array.
    void set_piece(uint64_t bytes) { piece_ = gran_ ? bytes / gran_ * gran_ : 0; }
    /// KV streaming: moves the allocations that lie wholly inside [from, to) to host memory (`host`) or back to the device, at the
    /// same addresses, each through a copy that has landed before its old memory is unmapped. The caller has every stream
    /// that reads or writes the array idle. `moved` counts the bytes moved; false with `err` at the first that could not move
    /// (host VMM refused, no device memory): what moved stays moved, the rest stays where it was, and every byte stays readable.
    /// `check` hashes each allocation before and after its move (a mismatch is an error).
    bool relocate(uint64_t from, uint64_t to, bool host, uint64_t& moved, std::string& err, bool check = false);
    uint64_t host_bytes() const { return host_; }
    /// The bytes of host allocations a relocate(from, to, true) would add.
    uint64_t device_bytes_in(uint64_t from, uint64_t to) const;
    uint8_t* data() const { return base_; }
    uint64_t backed() const { return backed_; }
    uint64_t capacity() const { return reserved_; }

private:
    struct Extent {
        uint64_t off, bytes;
        unsigned long long handle;
        bool host = false;
    };
    uint8_t* base_ = nullptr;
    uint64_t reserved_ = 0, backed_ = 0, gran_ = 0, piece_ = 0, host_ = 0;
    bool plain_ = false;   // one cudaMalloc, no virtual memory
    std::vector<Extent> extents_;
};

}  // namespace basalt::core

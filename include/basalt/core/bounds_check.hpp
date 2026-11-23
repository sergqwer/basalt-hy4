// include/basalt/core/bounds_check.hpp - opt-in bounds checks for the indices and copy targets a kernel or a copy would
// follow into memory that may be unmapped (HY4_BOUNDS_CHECK=1).
//
// The elastic KV reserves the whole context and backs only `backed` cells, and the expert tier's arena maps only part of
// its range after a lend: an index past the backed cells or a copy into a slot past the mapped range is an illegal
// memory access that kills the CUDA context, and the error then surfaces at some later checked call with no hint of its
// origin. With the flag on, the checked sites record their first violation - {site, layer, index or offset, limit} - in a
// mapped host report and step around it (an index set to 0, a span cut to the limit, a copy skipped) instead of faulting;
// the engine reports it at its next sync. With the flag off nothing is allocated or launched: every check returns at
// a null pointer.
#pragma once

#include <cstdint>
#include <string>

namespace basalt::core::bounds {

/// The checked sites (Report::site).
enum Site : uint32_t {
    kAttendCells = 1,      ///< the target's attention: a row's cells (listed or a span) against the backed cells
    kIndexerPools = 2,     ///< the indexer's pooled keys a row scores or a chunk writes, against the backed pools
    kDraftCells = 3,       ///< the drafter's gather of the target's selected cells
    kDraftPositions = 4,   ///< the drafter's gather of every position before its anchor
    kTierPromote = 10,     ///< a promotion's copy into a tier slot (claimed now, issued now or later)
    kTierFetch = 11,       ///< a bus fetch's copy into a tier slot
    kTierMove = 12,        ///< a lend's move of a hot expert between two slots
    kTierSource = 13,      ///< a copy out of a tier slot (the prompt's ring or staging)
};
const char* site_name(uint32_t site);

/// The process's report, in mapped pinned memory: the first violation claims it (`claim`), writes its fields and sets
/// `ready`; take() reads and clears it.
struct Report {
    uint32_t claim = 0, ready = 0, site = 0;
    int32_t layer = 0;
    int64_t index = 0, limit = 0;
};

/// HY4_BOUNDS_CHECK=1 in the environment (read once).
bool enabled();
/// The report's device view (null while off, or when the mapped page cannot be allocated).
Report* device_report();

/// A mapped int64 the host keeps at a limit the device checks read (graph-safe: a captured launch reads the current
/// value). Opens only while enabled(); `d` null otherwise.
struct Limit {
    int64_t* h = nullptr;
    const int64_t* d = nullptr;
    bool open();
    void close();
    void set(int64_t v) { if (h) *h = v; }
    int64_t get() const { return h ? *h : INT64_MAX; }
    ~Limit() { close(); }
};

/// A host site's violation (the first one is kept). No-op while off.
void host_violation(Site site, int layer, int64_t index, int64_t limit);
/// [lo, lo + n) inside [base, base + mapped)? Else records the violation (index: the offset from base, limit: mapped).
bool host_range(Site site, int layer, const void* lo, uint64_t n, const void* base, uint64_t mapped);

/// On the device, behind the stream's work: the cells rows t < n attend - cells[t * stride + j] for j < counts[t] when
/// `cells` is set and (fill null or fill[t] == 0), else firsts[t] + j - each in [0, *limit). A listed cell out of range is
/// set to 0, a span past the limit cut to it. Nothing is launched when `limit` or the report is null.
void check_cells(const int32_t* firsts, int32_t* counts, int32_t* cells, int stride, const int32_t* fill, int n,
                 const int64_t* limit, Site site, int layer, void* stream);
/// The same for a plain list of n cells.
void check_list(int32_t* cells, int n, const int64_t* limit, Site site, int layer, void* stream);

/// A violation recorded and landed: its message in `msg`, the report cleared; false when there is none.
bool take(std::string& msg);

}  // namespace basalt::core::bounds

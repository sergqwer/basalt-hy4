// include/basalt/platform/large_pages.hpp - large pages for the RAM tier, taken beside the reads.
//
// A large-page allocation is committed whole when it is made (VirtualAlloc MEM_RESERVE | MEM_COMMIT | MEM_LARGE_PAGES)
// and its address is only known then; a placeholder range cannot take large pages in place (MapViewOfFile3 refuses
// MEM_REPLACE_PLACEHOLDER with MEM_LARGE_PAGES, error 87, measured here; the view it maps without the flag is 4 KB pages).
// So the load plan writes to stand-in addresses in a range that is only reserved, and its host gate hands each reader
// the real address once that layer's pages are taken and registered (core::ExpertTier).  Large pages need
// SeLockMemoryPrivilege held by the token and enabled: a process without it never gets past large_page_support() and
// keeps its usual path.
#pragma once

#include <cstdint>
#include <string>

namespace basalt::platform {

struct LargePageSupport {
    bool usable = false;
    uint64_t page = 0;    ///< GetLargePageMinimum (2 MiB on x64)
    std::string why;      ///< why not, when not usable
};

/// Whether this process can allocate large pages now (enables SeLockMemoryPrivilege in its token first); asked once.
const LargePageSupport& large_page_support();

/// The RAM tier's choice (pure, for tests): large pages only when asked for (`want`) and usable; `note` says why not
/// when they were asked for and cannot be had.
inline bool use_large_pages(bool want, const LargePageSupport& s, std::string& note) {
    note.clear();
    if (!want) return false;
    if (!s.usable) {
        note = "large pages asked for but not usable (" + s.why + "): host VMM pages as before";
        return false;
    }
    return true;
}

/// `bytes` (a multiple of the large-page minimum) of committed large pages; null with the system's error code (1450 no
/// free large pages, 1314 privilege not held, ...).
uint8_t* alloc_large_pages(uint64_t bytes, unsigned long& error);
/// `bytes` of ordinary committed pages; null with the error code.
uint8_t* alloc_committed(uint64_t bytes, unsigned long& error);
/// Frees what either of the two returned.
void free_committed(uint8_t* p);
/// Address space only (never committed): the stand-in addresses a plan writes to; null on failure.
uint8_t* reserve_addresses(uint64_t bytes);
void release_addresses(uint8_t* p);

}  // namespace basalt::platform

// src/platform/large_pages.cpp - large or ordinary committed pages and reserved address ranges (large_pages.hpp).
#include "basalt/platform/large_pages.hpp"

#if defined(_WIN32)
#include "basalt/platform/win.hpp"
#endif

namespace basalt::platform {

#if defined(_WIN32)
namespace {
// Enables SeLockMemoryPrivilege; false (and why) when the token does not hold it: AdjustTokenPrivileges succeeds then
// but reports ERROR_NOT_ALL_ASSIGNED.
bool enable_lock_memory(std::string& why) {
    HANDLE tok = nullptr;
    if (!OpenProcessToken(GetCurrentProcess(), TOKEN_ADJUST_PRIVILEGES | TOKEN_QUERY, &tok)) {
        why = "cannot open the process token (error " + std::to_string(GetLastError()) + ")";
        return false;
    }
    TOKEN_PRIVILEGES tp{};
    tp.PrivilegeCount = 1;
    bool ok = false;
    if (!LookupPrivilegeValueW(nullptr, L"SeLockMemoryPrivilege", &tp.Privileges[0].Luid)) {
        why = "SeLockMemoryPrivilege unknown (error " + std::to_string(GetLastError()) + ")";
    } else {
        tp.Privileges[0].Attributes = SE_PRIVILEGE_ENABLED;
        const BOOL r = AdjustTokenPrivileges(tok, FALSE, &tp, 0, nullptr, nullptr);
        const DWORD e = GetLastError();
        ok = r && e == ERROR_SUCCESS;
        if (!ok)
            why = e == ERROR_NOT_ALL_ASSIGNED ? "SeLockMemoryPrivilege is not held by this process (an elevated process of an "
                                                "account granted 'Lock pages in memory', after a new sign-in)"
                                              : "AdjustTokenPrivileges error " + std::to_string(e);
    }
    CloseHandle(tok);
    return ok;
}
}  // namespace

const LargePageSupport& large_page_support() {
    static const LargePageSupport s = [] {
        LargePageSupport r;
        r.page = (uint64_t) GetLargePageMinimum();
        if (r.page == 0) { r.why = "the system has no large-page minimum"; return r; }
        if (!enable_lock_memory(r.why)) return r;
        r.usable = true;
        return r;
    }();
    return s;
}

uint8_t* alloc_large_pages(uint64_t bytes, unsigned long& error) {
    error = 0;
    void* p = VirtualAlloc(nullptr, (SIZE_T) bytes, MEM_RESERVE | MEM_COMMIT | MEM_LARGE_PAGES, PAGE_READWRITE);
    if (p == nullptr) error = GetLastError();
    return (uint8_t*) p;
}

uint8_t* alloc_committed(uint64_t bytes, unsigned long& error) {
    error = 0;
    void* p = VirtualAlloc(nullptr, (SIZE_T) bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    if (p == nullptr) error = GetLastError();
    return (uint8_t*) p;
}

void free_committed(uint8_t* p) {
    if (p) (void) VirtualFree(p, 0, MEM_RELEASE);
}

uint8_t* reserve_addresses(uint64_t bytes) {
    return (uint8_t*) VirtualAlloc(nullptr, (SIZE_T) bytes, MEM_RESERVE, PAGE_NOACCESS);
}

void release_addresses(uint8_t* p) {
    if (p) (void) VirtualFree(p, 0, MEM_RELEASE);
}
#else
const LargePageSupport& large_page_support() {
    static const LargePageSupport s = [] { LargePageSupport r; r.why = "large pages for the RAM tier are Windows-only here"; return r; }();
    return s;
}
uint8_t* alloc_large_pages(uint64_t, unsigned long& e) { e = 1; return nullptr; }
uint8_t* alloc_committed(uint64_t, unsigned long& e) { e = 1; return nullptr; }
void free_committed(uint8_t*) {}
uint8_t* reserve_addresses(uint64_t) { return nullptr; }
void release_addresses(uint8_t*) {}
#endif

}  // namespace basalt::platform

// src/platform/pagefile.cpp - the system's page files (pagefile.hpp).
#include "basalt/platform/pagefile.hpp"

#include <cstdio>

#if defined(_WIN32)
#include "basalt/platform/win.hpp"
#endif

namespace basalt::platform {

namespace {
std::string gib(uint64_t b) {
    char buf[32];
    std::snprintf(buf, sizeof buf, "%.1f GiB", (double) b / (double) (1ull << 30));
    return buf;
}
}  // namespace

#if defined(_WIN32)
namespace {
// SYSTEM_PAGEFILE_INFORMATION (class 18 of NtQuerySystemInformation): sizes in pages
struct PageFileInfo {
    ULONG next;
    ULONG total_pages;
    ULONG in_use;
    ULONG peak;
    USHORT name_len, name_max;
    PWSTR name;
};
using NtQsi = LONG(WINAPI*)(int, void*, ULONG, ULONG*);

std::vector<PageFileNow> current_files() {
    std::vector<PageFileNow> out;
    HMODULE nt = GetModuleHandleW(L"ntdll.dll");
    auto q = nt ? (NtQsi) (void*) GetProcAddress(nt, "NtQuerySystemInformation") : nullptr;
    if (!q) return out;
    std::vector<unsigned char> buf(1 << 16);
    ULONG got = 0;
    if (q(18, buf.data(), (ULONG) buf.size(), &got) != 0) return out;
    if (got == 0) return out;   // no page file at all
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    for (size_t off = 0;;) {
        const PageFileInfo* e = (const PageFileInfo*) (buf.data() + off);
        PageFileNow n;
        // "\??\C:\pagefile.sys": the drive is the letter before the first ':'
        for (USHORT i = 0; e->name && i + 1 < e->name_len / 2; ++i)
            if (e->name[i + 1] == L':') { n.drive = (char) std::toupper((int) e->name[i]); break; }
        n.bytes = (uint64_t) e->total_pages * si.dwPageSize;
        out.push_back(n);
        if (e->next == 0) break;
        off += e->next;
        if (off + sizeof(PageFileInfo) > buf.size()) break;
    }
    return out;
}

std::vector<std::string> configured_lines(bool& read) {
    std::vector<std::string> out;
    DWORD bytes = 0;
    const wchar_t* key = L"SYSTEM\\CurrentControlSet\\Control\\Session Manager\\Memory Management";
    read = RegGetValueW(HKEY_LOCAL_MACHINE, key, L"PagingFiles", RRF_RT_REG_MULTI_SZ, nullptr, nullptr, &bytes) == ERROR_SUCCESS;
    if (!read || bytes == 0) return out;
    std::vector<wchar_t> w(bytes / 2 + 2, 0);
    if (RegGetValueW(HKEY_LOCAL_MACHINE, key, L"PagingFiles", RRF_RT_REG_MULTI_SZ, nullptr, w.data(), &bytes) != ERROR_SUCCESS) {
        read = false;
        return out;
    }
    for (const wchar_t* p = w.data(); *p; p += wcslen(p) + 1) {
        std::string s;
        for (const wchar_t* c = p; *c; ++c) s += (*c < 128) ? (char) *c : '?';
        out.push_back(s);
    }
    return out;
}
}  // namespace

PageFileCheck page_file_check() {
    PageFileCheck r;
    bool read = false;
    const std::vector<PageFileConfig> cfg = parse_paging_files(configured_lines(read));
    const std::vector<PageFileNow> now = current_files();
    if (!read && now.empty()) return r;
    r.known = true;
    r.bytes = effective_page_file_bytes(cfg, now);
    r.ok = r.bytes >= kPageFileWantBytes;
    std::vector<char> drives;
    for (const auto& n : now) if (std::find(drives.begin(), drives.end(), n.drive) == drives.end()) drives.push_back(n.drive);
    for (const auto& c : cfg)
        if (c.drive != '?' && std::find(drives.begin(), drives.end(), c.drive) == drives.end()) drives.push_back(c.drive);
    for (char d : drives) {
        uint64_t cur = 0;
        for (const auto& n : now) if (n.drive == d) cur += n.bytes;
        std::string how = "not configured";
        for (const auto& c : cfg)
            if (c.drive == d) how = c.system_managed ? "system-managed" : c.initial_mb == c.max_mb ? "fixed" : "initial " + gib(c.initial_mb << 20);
        r.detail += std::string(r.detail.empty() ? "" : ", ") + d + ": " + gib(cur) + " now, " + how;
    }
    for (const auto& c : cfg)
        if (c.drive == '?') r.detail += std::string(r.detail.empty() ? "" : ", ") + "system-managed on every drive";
    if (r.detail.empty()) r.detail = "no page file";
    return r;
}
#else
PageFileCheck page_file_check() { return {}; }
#endif

std::string page_file_advice() {
    return "set a fixed page file of 64000 MB: System Properties > Advanced > Performance Settings > Advanced > Virtual "
           "memory > Change: clear 'Automatically manage', pick a drive with room, Custom size, Initial 64000 and Maximum "
           "64000 MB, Set, OK, then restart Windows (a card's memory is charged against RAM plus page file, and a "
           "system-managed file does not grow in time)";
}

}  // namespace basalt::platform

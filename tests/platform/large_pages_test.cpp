// tests/platform/large_pages_test.cpp - the RAM tier's large-page choice on fake inputs, then the live process: without
// SeLockMemoryPrivilege it must say why and keep the usual path; with it, large pages are taken; the ordinary-page
// fallback and the stand-in address range always work.
#include "basalt/platform/large_pages.hpp"

#include <cstdio>
#include <cstring>
#include <string>
#if defined(_WIN32)
#include "basalt/platform/win.hpp"
#endif

using namespace basalt::platform;

static int fails = 0, checks = 0;
static void check(bool ok, const char* what) {
    ++checks;
    if (!ok) {
        ++fails;
        std::printf("FAIL: %s\n", what);
    }
}

#if defined(_WIN32)
// This program again as a child whose token lacks SeLockMemoryPrivilege (a restricted copy of this one: no extra privilege
// needed to start it), as a process of an account without the grant runs: exit code 0 when it took the usual path.
static int run_without_privilege() {
    HANDLE tok = nullptr, rtok = nullptr;
    if (!OpenProcessToken(GetCurrentProcess(), TOKEN_DUPLICATE | TOKEN_QUERY | TOKEN_ASSIGN_PRIMARY, &tok)) return -1;
    LUID_AND_ATTRIBUTES del{};
    LookupPrivilegeValueW(nullptr, L"SeLockMemoryPrivilege", &del.Luid);
    const BOOL ok = CreateRestrictedToken(tok, 0, 0, nullptr, 1, &del, 0, nullptr, &rtok);
    CloseHandle(tok);
    if (!ok) return -2;
    wchar_t self[MAX_PATH];
    GetModuleFileNameW(nullptr, self, MAX_PATH);
    std::wstring cmd = std::wstring(L"\"") + self + L"\" --without-privilege";
    STARTUPINFOW si{};
    si.cb = sizeof si;
    PROCESS_INFORMATION pi{};
    if (!CreateProcessAsUserW(rtok, nullptr, cmd.data(), nullptr, nullptr, FALSE, 0, nullptr, nullptr, &si, &pi)) {
        std::printf("CreateProcessAsUser error %lu\n", GetLastError());
        CloseHandle(rtok);
        return -3;
    }
    WaitForSingleObject(pi.hProcess, 60000);
    DWORD code = 99;
    GetExitCodeProcess(pi.hProcess, &code);
    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);
    CloseHandle(rtok);
    return (int) code;
}
#endif

int main(int argc, char** argv) {
    if (argc > 1 && std::strcmp(argv[1], "--without-privilege") == 0) {   // the child: the usual path, and why
        const LargePageSupport& s = large_page_support();
        std::string n;
        const bool used = use_large_pages(true, s, n);
        std::printf("without the privilege: usable=%d, asked for -> %s (%s)\n", s.usable, used ? "large pages" : "host VMM", n.c_str());
        return !s.usable && !used && n.find("not held") != std::string::npos ? 0 : 1;
    }
    std::string note;
    LargePageSupport yes;
    yes.usable = true;
    yes.page = 2u << 20;
    LargePageSupport no;
    no.why = "SeLockMemoryPrivilege is not held by this process";
    check(!use_large_pages(false, yes, note) && note.empty(), "not asked for: never, silently");
    check(use_large_pages(true, yes, note) && note.empty(), "asked for and usable");
    check(!use_large_pages(true, no, note) && note.find("not usable") != std::string::npos &&
              note.find("not held") != std::string::npos && note.find("host VMM") != std::string::npos,
          "asked for, not usable: the usual path, with why");
    check(!use_large_pages(false, no, note) && note.empty(), "not asked for, not usable: silent");

    const LargePageSupport& live = large_page_support();
    std::printf("live: usable=%d page=%llu why=%s\n", live.usable, (unsigned long long) live.page, live.why.c_str());
    if (!live.usable) check(!use_large_pages(true, live, note), "this process keeps host VMM");

#if defined(_WIN32)
    const uint64_t page = live.page ? live.page : (2u << 20), bytes = 8 * page;
    unsigned long e = 0;
    uint8_t* lp = live.usable ? alloc_large_pages(bytes, e) : nullptr;
    std::printf("large pages: %s (error %lu)\n", lp ?"taken" : live.usable ? "refused" : "not tried", e);
    if (lp) {
        std::memset(lp, 0x5a, (size_t) bytes);
        check(lp[0] == 0x5a && lp[bytes - 1] == 0x5a, "large pages written and read back");
        free_committed(lp);
    }
    uint8_t* np = alloc_committed(bytes, e);
    check(np != nullptr, "ordinary committed pages (the fallback)");
    if (np) {
        std::memset(np, 0xa5, (size_t) bytes);
        check(np[0] == 0xa5 && np[bytes - 1] == 0xa5, "ordinary pages written and read back");
        free_committed(np);
    }
    uint8_t* standin = reserve_addresses(bytes);
    check(standin != nullptr, "stand-in addresses reserved");
    release_addresses(standin);
    const int child = run_without_privilege();
    check(child == 0, "a process without SeLockMemoryPrivilege keeps host VMM and says why");
#endif
    if (fails) return 1;
    std::printf("large_pages_test: %d checks OK\n", checks);
    return 0;
}

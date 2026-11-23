// tests/platform/pagefile_test.cpp - the page-file size that counts, on fake registry lines and current files; then the
// live check once (it only has to answer).
#include "basalt/platform/pagefile.hpp"

#include <cstdio>
#include <cstdlib>

using namespace basalt::platform;

static int fails = 0, checks = 0;
static void check(bool ok, const char* what) {
    ++checks;
    if (!ok) {
        ++fails;
        std::printf("FAIL: %s\n", what);
    }
}

int main() {
    constexpr uint64_t MiB = 1ull << 20;
    // parsing
    auto c = parse_paging_files({"C:\\pagefile.sys 64000 64000", "d:\\pagefile.sys 0 0", "?:\\pagefile.sys", "E:\\pagefile.sys 4096 16384", "junk"});
    check(c.size() == 4, "four entries, the junk line skipped");
    check(c[0].drive == 'C' && c[0].initial_mb == 64000 && c[0].max_mb == 64000 && !c[0].system_managed, "fixed C:");
    check(c[1].drive == 'D' && c[1].system_managed, "0 0 is system-managed");
    check(c[2].drive == '?' && c[2].system_managed, "?: is system-managed on every drive");
    check(c[3].initial_mb == 4096 && c[3].max_mb == 16384, "initial and maximum kept apart");

    // this PC: C: and D: fixed at 64,000 MB each, both allocated
    {
        auto cfg = parse_paging_files({"C:\\pagefile.sys 64000 64000", "D:\\pagefile.sys 64000 64000"});
        const uint64_t b = effective_page_file_bytes(cfg, {{'C', 64000 * MiB}, {'D', 64000 * MiB}});
        check(b == 128000 * MiB, "two fixed files sum");
        check(b >= kPageFileWantBytes, "this PC passes");
    }
    // one fixed 64,000 MB file passes alone
    check(effective_page_file_bytes(parse_paging_files({"C:\\pagefile.sys 64000 64000"}), {{'C', 64000 * MiB}}) >= kPageFileWantBytes,
          "a fixed 64000 MB file passes");
    // the growth maximum never counts: initial 4 GB, maximum 128 GB, 6 GB allocated now
    check(effective_page_file_bytes(parse_paging_files({"C:\\pagefile.sys 4096 131072"}), {{'C', 6144 * MiB}}) == 6144 * MiB,
          "the current size counts, not the maximum");
    // the initial size counts when the file is smaller now (it was just configured)
    check(effective_page_file_bytes(parse_paging_files({"C:\\pagefile.sys 65536 65536"}), {{'C', 2048 * MiB}}) == 65536 * MiB,
          "the initial size when larger than the current");
    // system-managed: only what exists now
    check(effective_page_file_bytes(parse_paging_files({"?:\\pagefile.sys"}), {{'C', 19 * 1024 * MiB}}) == 19 * 1024 * MiB,
          "system-managed counts its current size");
    check(effective_page_file_bytes(parse_paging_files({"?:\\pagefile.sys"}), {{'C', 19 * 1024 * MiB}}) < kPageFileWantBytes,
          "a 19 GiB system-managed file fails");
    // a configured file that does not exist yet counts its initial size; no page file at all is 0
    check(effective_page_file_bytes(parse_paging_files({"D:\\pagefile.sys 64000 64000"}), {}) == 64000 * MiB, "configured only");
    check(effective_page_file_bytes({}, {}) == 0, "none");
    // the threshold itself
    check(kPageFileWantBytes == 62ull << 30 && (64000 * MiB) >= kPageFileWantBytes && (63000 * MiB) < kPageFileWantBytes &&
              (63487 * MiB) < (64000 * MiB),
          "62 GiB threshold");

    const PageFileCheck live = page_file_check();
    std::printf("live: known=%d %.1f GiB ok=%d (%s)\n", live.known, live.bytes / 1073741824.0, live.ok, live.detail.c_str());
    if (fails) return 1;
    std::printf("pagefile_test: %d checks OK\n", checks);
    return 0;
}

// include/basalt/platform/pagefile.hpp - how much page file Windows has at start, for the warning that it is too small.
//
// Under WDDM a card's memory is charged against the commit limit (RAM plus page file), and a system-managed page file
// does not grow in time while the driver charges it: the start fails or the desktop stalls.  So the size that counts is
// what exists now, or the configured initial size if that is larger - never the growth maximum.  The parsing and the
// sum are plain functions (tested on fake inputs); page_file_check() reads the system's values.
#pragma once

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

namespace basalt::platform {

/// One configured page file: a line of the registry's PagingFiles ("C:\pagefile.sys 64000 64000", "?:\pagefile.sys",
/// "D:\pagefile.sys 0 0"); sizes in MB, 0 for a system-managed file.
struct PageFileConfig {
    char drive = 0;              ///< upper-case drive letter, '?' for "every drive, system-managed"
    uint64_t initial_mb = 0;
    uint64_t max_mb = 0;
    bool system_managed = true;
};

/// A page file that exists now (its current allocated size).
struct PageFileNow {
    char drive = 0;
    uint64_t bytes = 0;
};

inline std::vector<PageFileConfig> parse_paging_files(const std::vector<std::string>& lines) {
    std::vector<PageFileConfig> out;
    for (const std::string& raw : lines) {
        size_t i = 0;
        while (i < raw.size() && std::isspace((unsigned char) raw[i])) ++i;
        if (i + 1 >= raw.size() || raw[i + 1] != ':') continue;
        PageFileConfig c;
        c.drive = (char) std::toupper((unsigned char) raw[i]);
        size_t sp = raw.find(' ', i);
        if (sp != std::string::npos) {
            char* end = nullptr;
            const char* s = raw.c_str() + sp;
            const unsigned long long a = std::strtoull(s, &end, 10);
            if (end != s) {
                const char* s2 = end;
                const unsigned long long b = std::strtoull(s2, &end, 10);
                c.initial_mb = a;
                c.max_mb = end != s2 ? b : a;
            }
        }
        c.system_managed = c.initial_mb == 0;
        out.push_back(c);
    }
    return out;
}

/// The page file that counts: per drive the larger of what exists now and the configured initial size (a file only
/// configured counts its initial size; one that exists counts at least its current size).
inline uint64_t effective_page_file_bytes(const std::vector<PageFileConfig>& config, const std::vector<PageFileNow>& now) {
    std::vector<char> drives;
    for (const auto& n : now)
        if (n.drive && std::find(drives.begin(), drives.end(), n.drive) == drives.end()) drives.push_back(n.drive);
    for (const auto& c : config)
        if (c.drive && c.drive != '?' && std::find(drives.begin(), drives.end(), c.drive) == drives.end()) drives.push_back(c.drive);
    uint64_t total = 0;
    for (char d : drives) {
        uint64_t cur = 0, init = 0;
        for (const auto& n : now)
            if (n.drive == d) cur += n.bytes;
        for (const auto& c : config)
            if (c.drive == d) init = std::max<uint64_t>(init, c.initial_mb << 20);   // Windows' MB are MiB
        total += std::max(cur, init);
    }
    return total;
}

constexpr uint64_t kPageFileWantBytes = 62ull << 30;   ///< a fixed 64,000 MB file (62.5 GiB) passes

struct PageFileCheck {
    bool known = false;     ///< false off Windows, or when the system gave no reading
    uint64_t bytes = 0;     ///< effective_page_file_bytes
    bool ok = true;         ///< bytes >= kPageFileWantBytes (true when not known: nothing to warn about)
    std::string detail;     ///< per drive, e.g. "C: 62.5 GiB fixed, D: 62.5 GiB fixed"
};

/// The system's page files now (Windows: the registry's PagingFiles and the current files' sizes).
PageFileCheck page_file_check();
/// What to do when the check fails: the steps to a fixed 64,000 MB page file.
std::string page_file_advice();

}  // namespace basalt::platform

// src/program/serve/runtime.cpp - lifecycle of the persistent engine.
#include "state.hpp"

#include "basalt/core/on_device.hpp"

namespace basalt::program::serve {

void Runtime::start_guard() {
    basalt::platform::stop_memory_watchdog();
    guard_floor = basalt::platform::ram_reserve();
    if (guard_floor > 0) basalt::platform::start_ram_guard(guard_floor);
    disk_floor = guard_floor + (1ull << 30);
    // what a stop's error compares against, to tell the engine's own growth from other programs'
    ready_avail = basalt::platform::available_memory();
    ready_rss = (uint64_t) basalt::platform::process_memory().rss_mib << 20;
    shed_drop = 0;
}

void Runtime::shed_if_low() {
    if (basalt::platform::ram_guard_level() == basalt::platform::RamLevel::Ok) return;
    const uint64_t drop = basalt::platform::ram_guard_drops();
    if (drop == shed_drop) return;
    shed_drop = drop;
    if (conversations) {
        const size_t parked = conversations->size();
        const size_t freed = conversations->clear();
        size_t to_disk = 0;
        for (auto& sc : conversations->take_spilled())
            if (disk.enabled() && !disk_has(sc.live.ids, sc.live.imgs, sc.checkpoints, sc.cvec) &&
                disk.put(std::move(sc), kDiskQueueCap))
                ++to_disk;
        if (freed > 0)
            basalt::log::info("basalt serve", "RAM guard: %zu parked conversations (%zu MiB) leave RAM, %zu of them "
                              "to disk", parked, freed >> 20, to_disk);
    }
    if (const size_t pooled = basalt::core::CheckpointBytes::allocator_type::release_pooled(); pooled > 0)
        basalt::log::info("basalt serve", "RAM guard: freed %zu MiB of checkpoint blocks kept for reuse",
                          pooled >> 20);
    const uint64_t before = basalt::platform::available_memory();
    basalt::platform::release_free_heap();
    const uint64_t after = basalt::platform::available_memory();
    basalt::log::info("basalt serve", "RAM guard: free heap returned to the OS, %lld MiB free now (%+lld MiB)",
                      (long long) (after >> 20), ((long long) after - (long long) before) / (1ll << 20));
}

namespace {
int64_t pressure_ms() {
    static const int64_t ms = [] {
        const char* e = std::getenv("BASALT_CONV_PRESSURE_MS");
        return e ? (int64_t) std::atoll(e) : (int64_t) 2000;
    }();
    return ms;
}
uint64_t pressure_margin() {
    static const uint64_t m = [] {
        const char* e = std::getenv("BASALT_CONV_PRESSURE_MARGIN_MIB");
        return (uint64_t) (e ? std::max<long long>(0, std::atoll(e)) : 1024LL) << 20;
    }();
    return m;
}
}  // namespace

bool Runtime::pressure_on() const { return pressure_ms() > 0 && conversations && conversations->bytes() > 0; }

bool Runtime::pressure_due() const {
    return pressure_on() && Clock::now() - pressure_at >= std::chrono::milliseconds(pressure_ms());
}

void Runtime::release_if_short(const char* when) {
    if (pressure_ms() <= 0 || !conversations) return;
    pressure_at = Clock::now();
    const size_t held = conversations->bytes();
    if (held == 0) { pressure_quiet.on = false; return; }
    if (pressure_quiet.on && pressure_quiet.held == held) return;
    const uint64_t floor = (uint64_t) o.conversation_cache_min_free_mib << 20;
    const auto t0 = Clock::now();
    size_t to_disk = 0;
    try {
        const basalt::core::PressureRelease r = basalt::core::conversation_pressure_release(
            floor, pressure_margin(), [] { return basalt::core::conversation_available_memory(); },
            [&] {   // freed RAM back to the OS: the pooled checkpoint blocks, the heap, the disk writer's queue
                basalt::core::CheckpointBytes::allocator_type::release_pooled();
                basalt::platform::release_free_heap();
                if (disk.enabled()) disk.flush();   // the store writes durably: the bytes leave the modified file cache
            },
            [&]() -> std::optional<basalt::core::PressureUnit> {
                std::optional<basalt::core::PressureUnit> u = basalt::core::conversation_cache_release_one(*conversations);
                for (auto& sc : conversations->take_spilled())
                    if (disk.enabled() && !disk_has(sc.live.ids, sc.live.imgs, sc.checkpoints, sc.cvec) &&
                        disk.put(std::move(sc), kDiskQueueCap))
                        ++to_disk;
                return u;
            },
            conversations->size() + 2);
        pressure_quiet = {r.exhausted, conversations->bytes()};
        if (!r.below || (r.released() == 0 && !r.exhausted)) return;
        const long long before = r.before ? (long long) (*r.before >> 20) : -1;
        const long long after = r.after ? (long long) (*r.after >> 20) : -1;
        const double ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
        if (r.released() != 0)
            basalt::log::info("basalt serve", "conversation cache: RAM below the floor (%s): released %zu dialog%s%s, "
                              "%llu MiB (%zu to the disk tier) in %.0f ms; %lld -> %lld MiB available, floor %lld MiB + "
                              "%llu; parked=%zu%s", when, r.dialogs, r.dialogs == 1 ? "" : "s",
                              r.others ? " + the retained K/V" : "", (unsigned long long) (r.bytes >> 20), to_disk, ms,
                              before, after, (long long) o.conversation_cache_min_free_mib,
                              (unsigned long long) (pressure_margin() >> 20), conversations->size(),
                              r.exhausted ? "; nothing left to release" : "");
        else
            basalt::log::info("basalt serve", "conversation cache: RAM below the floor (%s: %lld MiB available, floor "
                              "%lld MiB): nothing left to release", when, after,
                              (long long) o.conversation_cache_min_free_mib);
        report_conversations();
    } catch (const std::bad_alloc&) {   // the disk tier's put allocates, under this very pressure
        pressure_quiet = {true, conversations->bytes()};
        basalt::log::warn("basalt serve", "conversation cache: RAM below the floor (%s): an allocation failed while "
                          "releasing; going on (parked=%zu)", when, conversations->size());
    }
}

void Runtime::ckd_finish() {
    const void* dst = ckd.pending();
    if (dst == nullptr) return;
    std::string ce;
    bool ok = ckd.finish(ce);
    if (ckd_peer.pending()) {
        const basalt::core::OnDevice on(ckd_peer_dev);
        ok = ckd_peer.finish(ce) && ok;
    }
    if (ok) return;
    basalt::log::error("basalt serve", "%s", ce.c_str());
    checks.erase(std::remove_if(checks.begin(), checks.end(),
                                [&](const ConvCheckpoint& k) { return k.state_a.data() == dst; }), checks.end());
}

bool Runtime::touch_check(int64_t to, ConvCheckpoint::Source source) {
    for (ConvCheckpoint& c : checks)
        if ((int64_t) c.ids.size() == to) {   // a turn end or the root outranks a stride that landed on it
            c.used = ++check_clock;
            if (source > c.source) c.source = source;
            return true;
        }
    return false;
}

void Runtime::keep_check(ConvCheckpoint&& c) {
    c.used = ++check_clock;
    checks.push_back(std::move(c));
    while ((int) checks.size() > o.prompt_cache) {
        std::vector<uint64_t> stamps;
        stamps.reserve(checks.size());
        for (const ConvCheckpoint& k : checks) stamps.push_back(k.used);
        const size_t victim = basalt::program::conv_cache::eviction_victim(stamps.data(), stamps.size(), o.prompt_cache);
        if (checks[victim].state_a.data() == ckd.pending()) {
            ckd_finish();   // a failed save drops its own checkpoint: look again
            continue;
        }
        checks.erase(checks.begin() + (std::ptrdiff_t) victim);
    }
}

int Runtime::execute() {
    if (const int code = init_before_input()) return code;
    if (const int code = init_protocol()) return code;
    if (const int code = announce()) return code;
    if (const int code = init_after_input()) return code;
    return run_requests();
}

int Runtime::run_requests() {
    for (;;) {
        const Flow flow = dispatch();
        if (flow == Flow::error) {
            if (basalt::core::gpu_left_running().load()) {   // the teardown would wait for the stuck window forever
                basalt::log::error("basalt serve", "the GPU did not finish a released window: ending without the CUDA teardown");
                std::fflush(stdout);
                std::_Exit(1);
            }
            return 1;
        }
        if (flow == Flow::quit) return shutdown();
    }
}

// Whether the disk tier needs no new copy of a conversation: it is short (below the disk's minimum, or below
// --conversation-cache-min-tokens when that is higher), or a stored prefix of it reaches within
// kDiskStaleTokens of the chain's deepest checkpoint (the point its next turn goes on from; a restore
// from that copy reads the rest again).  Each turn of a growing conversation is not written again in full.
bool Runtime::disk_has(const std::vector<int32_t>& ids, const std::vector<ImgKey>& imgs, const std::vector<ConvCheckpoint>& cks, bool cvec) {
    if ((int64_t) ids.size() < std::max<int64_t>(disk_min_tokens(), o.conversation_cache_min_tokens)) return true;
    int64_t deepest = 0;
    for (const ConvCheckpoint& c : cks) deepest = std::max<int64_t>(deepest, (int64_t) c.ids.size());
    if (deepest <= 0) deepest = (int64_t) ids.size();
    const auto m = disk.best(ids.data(), ids.size(), imgs, cvec);
    return m.tokens > 0 && m.tokens + kDiskStaleTokens >= deepest;
}

int Runtime::shutdown() {
    // the conversations this process holds go to the disk tier before it ends
    if (disk.enabled()) {
        const auto t0 = Clock::now();
        const size_t written = spill_to_disk(true, SIZE_MAX);
        const auto st = disk.stats();
        basalt::log::info("basalt serve", "conversation cache: %zu conversations written to disk at exit in %.0f ms; "
                          "on disk %zu (%.2f GiB); since the start written %zu (%.2f GiB, %.2f GiB with the shared "
                          "files they reuse), read back %zu (%.2f GiB), evicted %zu, dropped %zu", written,
                          std::chrono::duration<double, std::milli>(Clock::now() - t0).count(), st.files,
                          (double) st.bytes / 1073741824.0, st.written, (double) st.written_bytes / 1073741824.0,
                          (double) st.whole_bytes / 1073741824.0, st.restored, (double) st.read_bytes / 1073741824.0,
                          st.evicted, st.dropped);
    }
    at_exit();
    return 0;
}

}  // namespace basalt::program::serve

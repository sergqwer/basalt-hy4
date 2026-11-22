// RAM pressure on the parked conversations.  A park is admitted against --conversation-cache-min-free-mib, but once
// parked a conversation stayed in RAM until a later park or restore evicted it - also when the host ran short later
// (other programs opened, the engine's own memory grew, the file cache of a model's Engram table).  The serve loop
// calls this before a request and every few seconds while idle: below the floor the cache gives RAM back, oldest
// first, until the floor plus a margin is free.  (Port of the Strata fork's feccc5ec5: 3.3 GB freed in 104 ms.)
#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>

namespace basalt::core {

/// One step of a release: the bytes it held (the cache's own accounting) and whether it was a parked conversation
/// (else the retained K/V).
struct PressureUnit {
    uint64_t bytes = 0;
    bool dialog = false;
};

struct PressureRelease {
    bool below = false;                       ///< the first reading was under the floor
    std::optional<uint64_t> before, after;    ///< available RAM at the first and the last reading
    size_t dialogs = 0, others = 0;           ///< parked conversations / other units released
    uint64_t bytes = 0;                       ///< what they held
    bool exhausted = false;                   ///< nothing left to release while still under floor + margin
    size_t released() const { return dialogs + others; }
};

/// `sample()`: the available RAM now (nullopt: unknown - nothing is released on an unknown reading, as the park gate
/// fails closed).  Only when it is under `floor`: `settle()` (freed pooled blocks, the disk writer's queue) and then
/// `next()` one unit at a time, oldest first, settling and sampling after each, until floor + `margin` is available,
/// `next()` has nothing left (nullopt), or `bound` units went.
template<class Sample, class Settle, class Next>
PressureRelease conversation_pressure_release(uint64_t floor, uint64_t margin, Sample sample, Settle settle, Next next,
                                              size_t bound) {
    PressureRelease out;
    std::optional<uint64_t> room = sample();
    out.before = out.after = room;
    if (!room || *room >= floor) return out;
    out.below = true;
    const uint64_t target = floor + margin < floor ? UINT64_MAX : floor + margin;
    settle();   // freed memory may simply not be visible yet
    room = sample();
    out.after = room;
    while (room && *room < target) {
        if (out.released() >= bound) break;
        const std::optional<PressureUnit> unit = next();
        if (!unit) { out.exhausted = true; break; }
        (unit->dialog ? out.dialogs : out.others) += 1;
        out.bytes += unit->bytes;
        settle();
        room = sample();
        out.after = room;
    }
    return out;
}

/// The cache's own units, oldest first: the retained K/V (an optimization only), then the parked conversation unused
/// the longest (to the tier below when spilling, else dropped), then an idle batch slot's conversation out of VRAM.
template<class Cache>
std::optional<PressureUnit> conversation_cache_release_one(Cache& cache) {
    if (const size_t r = cache.retained_bytes()) {
        cache.drop_reuse();
        return PressureUnit{r, false};
    }
    const size_t before = cache.bytes();
    if (!cache.release_oldest()) return std::nullopt;
    return PressureUnit{before - cache.bytes(), true};
}

}  // namespace basalt::core

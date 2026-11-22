// RAM pressure on the parked conversations (conversation_pressure.hpp): the policy on fake readings, the cache's
// own units oldest first, and the spill to the tier below.
#include "basalt/core/conversation_cache.hpp"
#include "basalt/core/conversation_pressure.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <optional>
#include <vector>

using namespace basalt::core;

namespace {
int checks = 0;
void check(bool value, const char* description) {
    ++checks;
    if (!value) { std::fprintf(stderr, "FAIL: %s\n", description); std::exit(1); }
}
constexpr uint64_t MiB = 1ull << 20;
SavedConversation image(int32_t first, size_t bytes) {
    SavedConversation s;
    s.live.ids = {first, first + 1, first + 2};
    s.live.state_a.resize(bytes, 7);
    return s;
}
// a host whose available RAM rises by what the cache gives back
struct Host {
    uint64_t base;
    ConversationCache* cache;
    uint64_t start_held;
    std::optional<uint64_t> sample() const { return base + (start_held - cache->bytes()); }
};
}  // namespace

int main() {
    // the policy alone
    {
        int nexts = 0;
        auto r = conversation_pressure_release(10 * MiB, 1 * MiB, [] { return std::optional<uint64_t>(12 * MiB); },
                                               [] {}, [&] { ++nexts; return std::optional<PressureUnit>(); }, 9);
        check(!r.below && nexts == 0 && r.released() == 0, "above the floor: nothing released");
        r = conversation_pressure_release(10 * MiB, 1 * MiB, [] { return std::optional<uint64_t>(); }, [] {},
                                          [&] { ++nexts; return std::optional<PressureUnit>(); }, 9);
        check(!r.below && nexts == 0, "an unknown reading releases nothing");
        uint64_t avail = 8 * MiB;
        r = conversation_pressure_release(10 * MiB, 1 * MiB, [&] { return std::optional<uint64_t>(avail); },
                                          [&] { avail += 4 * MiB; }, [&] { ++nexts; return std::optional<PressureUnit>(); }, 9);
        check(r.below && nexts == 0 && r.released() == 0 && !r.exhausted, "settling alone recovers: nothing released");
        avail = 5 * MiB;
        nexts = 0;
        r = conversation_pressure_release(10 * MiB, 1 * MiB, [&] { return std::optional<uint64_t>(avail); }, [] {},
                                          [&] { ++nexts; avail += 2 * MiB; return std::optional<PressureUnit>(PressureUnit{2 * MiB, true}); }, 9);
        check(nexts == 3 && r.dialogs == 3 && r.bytes == 6 * MiB && *r.after == 11 * MiB,
              "below the floor: units go until floor + margin");
        avail = 5 * MiB;
        r = conversation_pressure_release(10 * MiB, 1 * MiB, [&] { return std::optional<uint64_t>(avail); }, [] {},
                                          [&] { return std::optional<PressureUnit>(PressureUnit{1, true}); }, 2);
        check(r.dialogs == 2, "the bound stops it");
        r = conversation_pressure_release(10 * MiB, 1 * MiB, [&] { return std::optional<uint64_t>(avail); }, [] {},
                                          [&] { return std::optional<PressureUnit>(); }, 9);
        check(r.exhausted && r.released() == 0, "nothing left to release is said");
    }
    // the cache's units: the retained K/V first, then the oldest parked, then the idle slots' conversations
    {
        ConversationCache cache(64 * MiB, 8);
        check(cache.put(image(100, 4 * MiB)), "park a");
        check(cache.put(image(200, 4 * MiB)), "park b");
        check(cache.put(image(300, 4 * MiB)), "park c");
        cache.put_slot(image(400, 4 * MiB), 4);
        std::vector<ConversationKv> kv(1);
        kv[0].k.resize(2 * MiB, 1);
        cache.retain(std::move(kv), 3);
        check(cache.retained_bytes() >= 2 * MiB, "retained K/V held");
        Host host{6 * MiB, &cache, cache.bytes()};
        auto r = conversation_pressure_release(
            12 * MiB, 1 * MiB, [&] { return host.sample(); }, [] {},
            [&] { return conversation_cache_release_one(cache); }, cache.size() + 2);
        check(r.others == 1 && cache.retained_bytes() == 0, "the retained K/V went first");
        check(r.dialogs == 2 && cache.size() == 2, "then the two oldest dialogs, until floor + margin");
        check(cache.best(std::vector<int32_t>{300, 301, 302, 9}, {}, true).tokens > 0, "the newest dialog stays");
        check(cache.best(std::vector<int32_t>{100, 101, 102, 9}, {}, true).tokens == 0, "the oldest went");
        check(*r.after >= 13 * MiB, "floor + margin available after");
        Host low{1 * MiB, &cache, cache.bytes()};
        r = conversation_pressure_release(
            64 * MiB, 1 * MiB, [&] { return low.sample(); }, [] {}, [&] { return conversation_cache_release_one(cache); },
            cache.size() + 2);
        check(r.dialogs == 2 && r.exhausted && cache.size() == 0 && cache.bytes() == 0,
              "far below: the idle slot's conversation goes too, then nothing is left");
    }
    // spilling on: what leaves waits for the tier below
    {
        ConversationCache cache(64 * MiB, 8);
        cache.set_spill(true);
        check(cache.put(image(100, 4 * MiB)) && cache.put(image(200, 4 * MiB)), "park two");
        auto u = conversation_cache_release_one(cache);
        check(u && u->dialog && u->bytes >= 4 * MiB, "one dialog released");
        auto spilled = cache.take_spilled();
        check(spilled.size() == 1 && spilled[0].live.ids[0] == 100, "the oldest went to the tier below");
    }
    std::printf("conversation_pressure_test: %d checks OK\n", checks);
    return 0;
}

// tests/program/respell_parked_test.cpp - a prompt respelled against a parked conversation (header-only, CPU).
//
// The serve case it covers: a chat's reply came back from the client in other pieces than the model wrote it, so its
// next turn was respelled against the live conversation and checkpointed in the model's pieces. Another chat then parked
// it; when the turn is sent again, the prompt must be respelled against the parked checkpoints too, or it meets only the
// checkpoint before the reply and reads the reply again on the prompt path (other ids, another answer).
#include "basalt/core/conversation_cache.hpp"
#include "program/serve/prefix_anchor.hpp"

#include <algorithm>
#include <cstdio>
#include <map>
#include <string>
#include <vector>

using basalt::core::ConversationCache;
using basalt::core::ConversationCheckpoint;
using basalt::core::SavedConversation;
using basalt::program::serve::HeldMeet;
using basalt::program::serve::meet_held;
using basalt::program::serve::splice_held;
using Ids = std::vector<int32_t>;

namespace {
int failures = 0;
void check(bool ok, const char* what) {
    std::printf("%s %s\n", ok ? "ok  " : "FAIL", what);
    if (!ok) ++failures;
}
// 1 "a" + 2 "b" is how the model wrote the reply; 3 "ab" is how the tokenizer cuts it
const std::map<int32_t, std::string> kBytes = {{1, "a"}, {2, "b"}, {3, "ab"}, {4, "<end>"}, {5, "\n"}, {6, "c"}, {10, "<sys>"}, {11, "hi"}};
std::string_view bytes(int32_t id) {
    const auto it = kBytes.find(id);
    return it == kBytes.end() ? std::string_view() : std::string_view(it->second);
}
ConversationCheckpoint ck(const Ids& ids) {
    ConversationCheckpoint c;
    c.ids = ids;
    c.state_a.resize(16, 1);
    return c;
}
// What Request::respell_prompt does with the parked entries: the longest held sequence the prompt meets, spliced in.
Ids respell(const ConversationCache& cache, const Ids& prompt, bool cvec) {
    const int32_t* best = nullptr;
    HeldMeet meet;
    cache.each_held(cvec, [&](const Ids& h) {
        if ((int64_t) h.size() <= meet.held) return;
        const HeldMeet m = meet_held(prompt.data(), (int64_t) prompt.size(), h.data(), (int64_t) h.size(), true, bytes);
        if (m.held > 0) {
            best = h.data();
            meet = m;
        }
    });
    if (best == nullptr || !meet.respelled()) return prompt;
    Ids out;
    splice_held(best, meet, prompt.data(), (int64_t) prompt.size(), out);
    return out;
}
}  // namespace

int main() {
    const Ids a_check = {10, 11};                         // turn A's checkpoint: its prompt <sys> hi \n, less the last token
    const Ids b_respelled = {10, 11, 5, 1, 2, 4, 5, 6, 5};  // turn B as read: A's reply in the model's pieces, then "c\n"
    const Ids b_check(b_respelled.begin(), b_respelled.end() - 1);
    const Ids d = {10, 11, 5, 3, 4, 5, 6, 5};             // turn B sent again, tokenized by the server: "ab" in one piece

    ConversationCache cache(1 << 20, 4);
    SavedConversation parked;
    parked.live.ids = b_respelled;
    parked.live.ids.insert(parked.live.ids.end(), {6, 6});   // B's reply as generated
    parked.live.state_a.resize(16, 1);
    parked.checkpoints = {ck(a_check), ck(b_check)};
    parked.cvec = true;
    check(cache.put(std::move(parked)), "the conversation is parked");

    check(cache.best(d, {}, true).tokens == (int64_t) a_check.size(),
          "unrespelled, the prompt meets only the checkpoint before the reply (the bug: the reply is read again)");
    int held = 0;
    cache.each_held(true, [&](const Ids&) { ++held; });
    check(held == 3, "each_held offers the parked live ids and both checkpoints");
    int other = 0;
    cache.each_held(false, [&](const Ids&) { ++other; });
    check(other == 0, "each_held skips a conversation of the other control-vector state");

    const Ids spelled = respell(cache, d, true);
    check(spelled.size() == b_respelled.size() && std::equal(spelled.begin(), spelled.end(), b_respelled.begin()),
          "respelled against the parked checkpoint, the prompt is turn B as it was read");
    const ConversationCache::Match m = cache.best(spelled, {}, true);
    check(m.tokens == (int64_t) b_check.size() && !m.live,
          "the respelled prompt continues turn B's checkpoint: one token is read, on the path turn B read it");

    const Ids unrelated = {10, 6, 6, 5};
    check(respell(cache, unrelated, true) == unrelated, "a prompt that shares only the system prompt is left as it is");

    std::printf(failures ? "respell_parked_test: %d failures\n" : "respell_parked_test: ok\n", failures);
    return failures ? 1 : 0;
}

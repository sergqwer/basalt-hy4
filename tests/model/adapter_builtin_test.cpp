// tests/model/adapter_builtin_test.cpp - the program's own registration: the Hy4 adapter, once, under both of its names,
// an unknown architecture refused, and the routing flags it reports as lossy. Linking the models needs the CUDA runtime, so this runs where
// that is installed; no GPU work.
#include "basalt/model/adapter.hpp"
#include "basalt/model/flags.hpp"
#include "basalt/models/builtin.hpp"
#include "basalt/program/options.hpp"

#include <cstdio>
#include <memory>
#include <string>
#include <utility>
#include <vector>

int main() {
    using namespace basalt;
    int failures = 0;
    const auto check = [&](bool ok, const char* what) {
        std::printf("%s %s\n", ok ? "ok  " : "FAIL", what);
        if (!ok) ++failures;
    };
    models::register_builtin_models();
    models::register_builtin_models();
    check(model::find_adapter("hy4flash") != nullptr, "the hy4flash adapter is registered");
    // and under the architecture its test pack (made from the AngelSlim GGUF) carries
    check(model::find_adapter("hy_v4_flash") != nullptr, "hy4flash also answers to hy_v4_flash");
    check(model::adapter_names().size() == 2, "registering twice keeps one adapter per name");
    check(model::find_adapter("llama") == nullptr &&
              model::unavailable_architecture("llama").find("no model adapter") != std::string::npos,
          "an architecture Basalt does not ship is refused by name");
    if (const model::ModelAdapter* hy = model::find_adapter("hy4flash")) {
        const auto o = hy->make_options();
        check(!hy->finish_flags(*o) && o->lossy.empty(), "hy4flash with no flag is not lossy");
    }
    // the flags as the command line gives them, through the model's own parser and finish_flags
    const auto lossy_of = [](const model::ModelAdapter& a, std::vector<std::string> words) {
        const auto o = a.make_options();
        words.insert(words.begin(), "basalt");
        std::vector<char*> argv;
        for (auto& w : words) argv.push_back(w.data());
        for (int i = 1; i < (int) argv.size(); ++i) {
            int code = 0;
            model::FlagArgs args((int) argv.size(), argv.data(), i);
            if (!a.parse_flag(argv[i], args, *o, code) || code != 0) return std::vector<program::CommonOptions::LossyFlag>{{"?", argv[i]}};
        }
        (void) a.finish_flags(*o);
        return o->lossy;
    };
    // the routing flags: the bare prior is the model's own; the tail skip needs the pack's experts per token, so without
    // a pack it is refused
    for (const auto& [arch, bare] : {std::pair<const char*, const char*>{"hy4flash", "0.01"}}) {
        const model::ModelAdapter* m = model::find_adapter(arch);
        if (m == nullptr) continue;
        const auto p = lossy_of(*m, {"--route-prior"});
        check(p.size() == 1 && p[0].effect == std::string("--route-prior ") + bare + " changes expert routing", "the bare --route-prior is the model's");
        const auto o = m->make_options();
        std::vector<std::string> words = {"basalt", "--route-tail-skip", "3"};
        std::vector<char*> argv;
        for (auto& w : words) argv.push_back(w.data());
        int i = 1, code = 0;
        model::FlagArgs args((int) argv.size(), argv.data(), i);
        check(m->parse_flag(argv[1], args, *o, code) && code == 0 && o->route.tail_skip == 3, "--route-tail-skip 3 is taken");
        check(m->finish_flags(*o).value_or(0) == 2, "--route-tail-skip without the pack's experts per token is refused");
    }
    return failures ? 1 : 0;
}

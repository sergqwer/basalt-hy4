// tests/model/adapter_registry_test.cpp - the adapter registry: lookup by architecture, the replaced guard and Hy4
// geometry from `hy_v4_flash.*` metadata. Host only: no model and no runtime are linked.
#include "../core/gguf_fixture.hpp"
#include "basalt/model/adapter.hpp"
#include "meta_adapters.hpp"

#include <cstdio>
#include <filesystem>

namespace {
int failures = 0;
void check(bool ok, const char* what) {
    std::printf("%s %s\n", ok ? "ok  " : "FAIL", what);
    if (!ok) ++failures;
}
}  // namespace

int main() {
    using namespace basalt;
    test::register_meta_adapters();
    check(model::find_adapter("hy4flash") != nullptr, "the hy4flash adapter is registered");
    check(model::find_adapter("llama") == nullptr, "an unknown architecture has no adapter");

    const auto dir = std::filesystem::temp_directory_path() / "basalt_adapter_registry_test";
    std::filesystem::create_directories(dir);
    using namespace fixture;

    write(dir / "other.gguf", {str("general.architecture", "llama")}, {});
    check(model::check_architecture(GgufFile((dir / "other.gguf").string())).find("no model adapter") != std::string::npos,
          "an unregistered architecture is refused by name");

    write(dir / "bare.gguf", {str("general.architecture", "hy4flash")}, {});
    check(model::check_architecture(GgufFile((dir / "bare.gguf").string())).find("block_count") != std::string::npos,
          "hy4flash without its keys is refused");

    const std::vector<Kv> hy = {str("general.architecture", "hy4flash"), u32("hy_v4_flash.block_count", 43),
                                u32("hy_v4_flash.nextn_predict_layers", 1), u32("hy_v4_flash.embedding_length", 4096),
                                u32("hy_v4_flash.expert_count", 512), u32("hy_v4_flash.expert_used_count", 8),
                                u32("hy_v4_flash.expert_feed_forward_length", 1536),
                                u32("hy_v4_flash.attention.kv_lora_rank", 512), u32("hy_v4_flash.expert_latent_length", 1024)};
    write(dir / "hy.gguf", hy, {});
    const GgufFile g((dir / "hy.gguf").string());
    check(model::check_architecture(g).empty(), "hy4flash with its keys validates");
    std::string err;
    const model::ModelAdapter* a = model::adapter_for(g, err);
    check(a != nullptr && std::string(a->arch()) == "hy4flash", "adapter_for picks hy4flash");
    if (a) {
        const model::ModelDesc d = a->describe(g);
        check(d.n_layers == 42 && d.n_embd == 4096 && d.n_expert == 512 && d.n_expert_used == 8 && d.n_ff_exp == 1536,
              "describe() reads the geometry from the metadata (the MTP layer left out)");
    }
    std::filesystem::remove_all(dir);
    return failures ? 1 : 0;
}

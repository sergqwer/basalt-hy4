// tests/model/meta_adapters.hpp - registers an adapter per architecture that has the model's metadata rules and nothing
// else, so a CPU test links no model and no runtime.
#pragma once

#include "basalt/model/adapter.hpp"
#include "models/hy4flash/meta.hpp"

#include <memory>
#include <string>

namespace basalt::test {

class MetaAdapter final : public model::ModelAdapter {
public:
    using Validate = std::string (*)(const GgufFile&);
    using Describe = model::ModelDesc (*)(const GgufFile&);
    MetaAdapter(const char* arch, Validate validate, Describe describe) : arch_(arch), validate_(validate), describe_(describe) {}
    const char* arch() const override { return arch_; }
    std::string validate(const GgufFile& g) const override { return validate_(g); }
    model::ModelDesc describe(const GgufFile& g) const override { return describe_(g); }
    model::TokenizerSpec tokenizer_spec() const override { return {}; }
    int max_expert_cards() const override { return 0; }
    model::TokenNames token_names() const override { return {}; }
    PreTokenizer pretokenizer() const override { return {}; }
    program::grammar::ToolForm tool_form() const override { return {}; }

private:
    const char* arch_;
    Validate validate_;
    Describe describe_;
};

inline void register_meta_adapters() {
    if (model::find_adapter("hy4flash") == nullptr)
        model::register_adapter(std::make_unique<MetaAdapter>("hy4flash", models::hy4flash::validate_meta, models::hy4flash::describe_meta));
}

}  // namespace basalt::test

// include/basalt/kernels/router.hpp - the MoE router: scores from a gate matrix, the top-k of the biased scores, the chosen
// unbiased scores normalised.
//
// All activations are fp32 and device pointers; a call takes T tokens (any T >= 1) and the result for a token does not depend
// on T or on the token's place in the window. `stream` may be null (the call then synchronises).
#pragma once

#include "basalt/kernels/route_lossy.hpp"

#include <cstdint>

namespace basalt::kernels {

enum class RouterGate : int { SqrtSoftplus = 0, Sigmoid = 1 };

struct RouterArgs {
    int n_expert = 384;               ///< multiple of 32, at most 512
    int n_embd = 5120;                ///< multiple of 128
    int k = 6;                        ///< at most 16
    float weight_scale = 1.5f;        ///< multiplies the normalised weights
    RouterGate gate = RouterGate::SqrtSoftplus;
    /// --route-prior (opt-in, changes the output): above 0, an expert `held` holds is chosen on s + bias + `prior`; the
    /// weights stay the unbiased s of the chosen, normalised as above.  0 = the plain router, bit for bit.
    float prior = 0.0f;
    ResidentTabs held;
};

/// logits = gate . x (fp32, gate row-major [n_expert][n_embd]); s = the gate function of the logit; the top-k of s + bias
/// are chosen (the lower index wins a tie); weights are the unbiased s of those experts, divided by their sum plus
/// 1e-20, times weight_scale. `logits` is a [T][n_expert] output (workspace); ids and weights are [T][k].
void router(const float* x, const float* gate, const float* bias, int n_tokens, const RouterArgs& a, float* logits,
            int32_t* ids, float* weights, void* stream);

/// router's choice from logits computed another way (a prompt chunk's scores as one product): the same selection and
/// weights `router` makes from the same logits.
void router_select(const float* logits, const float* bias, int n_tokens, const RouterArgs& a, int32_t* ids, float* weights,
                   void* stream);

/// A prompt chunk's logits (router_select's input) on the BF16 tensor cores: x and gate as two BF16 terms each,
/// hi.hi + hi.lo + lo.hi in fp32 (router_tc.cu). Close to router()'s fp32 logits, not the same bits; n_expert % 128, n_embd % 32.
bool router_logits_bf16x3_fits(int n_expert, int n_embd);
void router_logits_bf16x3(const float* x, const float* gate, int n_tokens, int n_expert, int n_embd, float* logits, void* stream);

}  // namespace basalt::kernels

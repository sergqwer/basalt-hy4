// tests/kernels/cpu/pool_decode_bench.cpp - the CPU expert pool at a Hy4-Flash decode layer's share, on synthetic NVFP4
// experts in a working set far larger than the caches.
//
//     build/pool_decode_bench [experts_a_layer] [tokens_an_expert] [workers] [layers] [gap_us] [set_gib]
//
// Each "layer" picks `experts_a_layer` random experts of the set (1536 x 1536 latent experts, NVFP4 gate/up/down, 3.98 MB
// each) and runs them through start_split_multi_native / finish_split_multi_native as the decode window does, then spins
// `gap_us` (the card's part of the layer, when the pool parks). Prints the layer time (median, mean, p90) and the
// weight bytes read per second. BASALT_POOL_PIN=1 pins the workers.
#include "basalt/ggml/ggml.h"
#include "basalt/kernels/cpu/expert_layout.hpp"
#include "basalt/kernels/cpu/native_expert.hpp"
#include "basalt/kernels/cpu/pool.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <thread>
#include <vector>

namespace cpu = basalt::kernels::cpu;

namespace {
double now_us() {
    return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
}  // namespace

int main(int argc, char** argv) {
    const int per_layer = argc > 1 ? std::atoi(argv[1]) : 5;
    const int nt = argc > 2 ? std::atoi(argv[2]) : 2;
    const int workers = argc > 3 ? std::atoi(argv[3]) : 15;
    const int layers = argc > 4 ? std::atoi(argv[4]) : 2000;
    const double gap_us = argc > 5 ? std::atof(argv[5]) : 400.0;
    const double set_gib = argc > 6 ? std::atof(argv[6]) : 4.0;
    constexpr int L = 1536, F = 1536, kNvfp4 = 40;
    cpu::NativeFmt f;
    std::string err;
    if (!cpu::native_fmt(kNvfp4, kNvfp4, L, F, f, err, 7.0f)) {
        std::printf("native_fmt: %s\n", err.c_str());
        return 1;
    }
    const size_t n_set = (size_t) (set_gib * (1ull << 30) / (double) f.bytes);
    std::vector<uint8_t> set(n_set * f.bytes);
    {   // random codes; UE4M3 scales small and valid; per-expert fp32 tails
        uint64_t s = 0x9e3779b97f4a7c15ull;
        uint64_t* p = (uint64_t*) set.data();
        for (size_t i = 0; i < set.size() / 8; ++i) {
            s ^= s << 13; s ^= s >> 7; s ^= s << 17;
            p[i] = s & 0xbfbfbfbfbfbfbfbfull;   // keeps scale bytes below 0x7f-ish exponents
        }
        const float tail[4] = {0.01f, 0.01f, 0.01f, 0.f};
        for (size_t e = 0; e < n_set; ++e) std::memcpy(set.data() + e * f.bytes + f.bytes - cpu::kNvfp4Tail, tail, sizeof tail);
    }
    std::vector<std::vector<uint8_t>> act(cpu::MAXT, std::vector<uint8_t>(cpu::kNativeActBytes));
    {
        std::mt19937 rng(7);
        std::normal_distribution<float> nd(0.f, 1.f);
        std::vector<float> x(L);
        for (auto& a : act) {
            for (auto& v : x) v = nd(rng);
            cpu::native_quant_act(f, x.data(), a.data());
        }
    }
    std::vector<float> out((size_t) per_layer * nt * L);
    std::vector<cpu::ExpertJobMulti> jobs((size_t) per_layer);
    cpu::ExpertPool pool(workers);
    std::mt19937 rng(11);
    std::vector<double> t;
    t.reserve((size_t) layers);
    for (int it = -50; it < layers; ++it) {
        for (int j = 0; j < per_layer; ++j) {
            jobs[(size_t) j] = cpu::ExpertJobMulti{};
            jobs[(size_t) j].blob = set.data() + (size_t) (rng() % n_set) * f.bytes;
            jobs[(size_t) j].nt = nt;
            for (int k = 0; k < nt; ++k) {
                jobs[(size_t) j].nact[k] = act[(size_t) k].data();
                jobs[(size_t) j].out[k] = out.data() + ((size_t) j * nt + k) * L;
            }
        }
        const double a = now_us();
        pool.start_split_multi_native(f, jobs.data(), per_layer);
        pool.finish_split_multi_native();
        const double b = now_us();
        if (it >= 0) t.push_back(b - a);
        while (now_us() - b < gap_us) {}
    }
    std::vector<double> s = t;
    std::sort(s.begin(), s.end());
    double sum = 0;
    for (double v : t) sum += v;
    const double mean = sum / (double) t.size(), med = s[s.size() / 2], p90 = s[s.size() * 9 / 10];
    const double mb = (double) per_layer * (double) f.bytes / 1e6;
    std::printf("pool_decode_bench: %d experts x %d tokens, %d workers, gap %.0f us, set %.1f GiB: layer median %.1f us, mean %.1f, "
                "p90 %.1f, min %.1f | %.1f GB/s at the mean, %.1f at the median\n",
                per_layer, nt, workers, gap_us, set_gib, med, mean, p90, s.front(), mb / mean * 1e3, mb / med * 1e3);
    return 0;
}

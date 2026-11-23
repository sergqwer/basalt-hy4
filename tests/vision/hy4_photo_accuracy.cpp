// tests/vision/hy4_photo_accuracy.cpp - Hy4-flash's image encoder on a real picture from the pack's own projector: the CUDA
// encoder (what --vision dynamic runs) and the CPU encoder with the device's bf16 roundings, each against the CPU encoder in
// FP32 (bf16 weights as stored, every activation, the attention and the softmax in FP32).
//
//   hy4_photo_accuracy MODEL.basalt IMAGE [threads]
//
// Prints per comparison: relative L2 over the block, max and mean |a - ref| / max|ref| of a row, per-row cosine (min, mean),
// and the share of values whose relative error is over 1%, 5% and 11%.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <string>
#include <thread>
#include <vector>

#include <cuda_runtime.h>

#include "basalt/core/model_file.hpp"
#include "basalt/vision/encoder_cpu.hpp"
#include "basalt/vision/encoder_cuda.hpp"
#include "basalt/vision/mmproj.hpp"
#include "basalt/vision/preprocess.hpp"
#include "models/hy4flash/vision.hpp"

namespace v = basalt::vision;
namespace h4 = basalt::models::hy4flash;

static double secs(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

static void compare(const char* what, const std::vector<float>& a, const std::vector<float>& ref, int width) {
    if (a.size() != ref.size() || width <= 0) { std::printf("%s: sizes differ (%zu vs %zu)\n", what, a.size(), ref.size()); return; }
    const size_t rows = ref.size() / (size_t) width;
    double d2 = 0, r2 = 0, cos_min = 1, cos_sum = 0, row_max = 0, row_mean = 0;
    size_t over1 = 0, over5 = 0, over11 = 0, counted = 0;
    double refmax_all = 0;
    for (float x : ref) refmax_all = std::max(refmax_all, (double) std::fabs(x));
    for (size_t r = 0; r < rows; ++r) {
        const float* x = a.data() + r * width;
        const float* y = ref.data() + r * width;
        double dot = 0, xx = 0, yy = 0, dmax = 0, ymax = 0, dsum = 0;
        for (int i = 0; i < width; ++i) {
            const double d = (double) x[i] - y[i];
            d2 += d * d; r2 += (double) y[i] * y[i];
            dot += (double) x[i] * y[i]; xx += (double) x[i] * x[i]; yy += (double) y[i] * y[i];
            dmax = std::max(dmax, std::fabs(d)); ymax = std::max(ymax, (double) std::fabs(y[i])); dsum += std::fabs(d);
            // a value's relative error, for values above 1% of the block's largest (smaller ones have no stable ratio)
            if (std::fabs(y[i]) > 0.01 * refmax_all) {
                const double rel = std::fabs(d) / std::fabs(y[i]);
                ++counted; over1 += rel > 0.01; over5 += rel > 0.05; over11 += rel > 0.11;
            }
        }
        const double c = dot / std::sqrt(std::max(xx * yy, 1e-300));
        cos_min = std::min(cos_min, c); cos_sum += c;
        row_max = std::max(row_max, dmax / std::max(ymax, 1e-30));
        row_mean += dsum / width / std::max(ymax, 1e-30);
    }
    std::printf("%-34s rel L2 %.3e | row max |d|/max|ref| %.3e, mean %.3e | cosine min %.6f mean %.6f | rel err > 1%% %.3f%%, "
                "> 5%% %.3f%%, > 11%% %.3f%% (of %zu values)\n",
                what, std::sqrt(d2 / std::max(r2, 1e-300)), row_max, row_mean / rows, cos_min, cos_sum / rows,
                100.0 * over1 / std::max<size_t>(counted, 1), 100.0 * over5 / std::max<size_t>(counted, 1),
                100.0 * over11 / std::max<size_t>(counted, 1), counted);
}

int main(int argc, char** argv) {
    if (argc < 3) { std::printf("usage: hy4_photo_accuracy MODEL.basalt IMAGE [threads]\n"); return 2; }
    const int threads = argc > 3 ? std::atoi(argv[3]) : (int) std::max(1u, std::thread::hardware_concurrency() / 2);
    std::string err;
    basalt::core::ModelFile file;
    if (!file.open(argv[1], err) || file.range("vision/mmproj").empty()) { std::printf("no vision/mmproj: %s\n", err.c_str()); return 1; }
    std::ifstream f(argv[2], std::ios::binary);
    const std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    v::Image img;
    if (bytes.empty() || !v::decode_image(bytes.data(), bytes.size(), img, err)) { std::printf("cannot read the image: %s\n", err.c_str()); return 1; }
    h4::ImagePlan plan;
    if (!h4::plan_image(img.w, img.h, plan, err)) { std::printf("%s\n", err.c_str()); return 1; }
    std::vector<float> patches;
    h4::image_patches(img, plan, patches);
    std::printf("image %dx%d -> canvas %dx%d, %d merged tokens (%d x %d), %d rows\n", img.w, img.h, plan.canvas.w, plan.canvas.h,
                plan.n_merged(), plan.ow, plan.oh, plan.n_rows());

    std::vector<float> gpu, cpu_bf16, cpu_fp32;
    int W = 0;   // the projector's width (the text model's)
    {
        v::Mmproj mp;
        if (!mp.open(file.range("vision/mmproj"), err, h4::kProjector)) { std::printf("%s\n", err.c_str()); return 1; }
        W = mp.hp().proj;
        v::CudaEncoder e;
        if (!e.stage(mp, true, err) || !e.create(err)) { std::printf("cuda: %s\n", err.c_str()); return 1; }
        const int tokens = std::max(plan.n_merged(), 1024);
        void* region = nullptr;
        const size_t rb = e.arena_bytes(tokens);
        if (cudaMalloc(&region, rb) != cudaSuccess || !e.attach(region, rb, tokens, err)) { std::printf("cuda region: %s\n", err.c_str()); return 1; }
        auto t0 = std::chrono::steady_clock::now();
        if (!e.encode(patches, plan.canvas, gpu)) { std::printf("cuda encode failed: %s\n", e.last_error().c_str()); return 1; }
        std::printf("cuda encoder: %.2f s (first call), region %.0f MiB\n", secs(t0), rb / 1048576.0);
        e.detach();
        cudaFree(region);
    }
    for (int k = 0; k < 2; ++k) {
        v::Mmproj mp;
        if (!mp.open(file.range("vision/mmproj"), err, h4::kProjector)) { std::printf("%s\n", err.c_str()); return 1; }
        v::CpuEncoder c;
        if (!c.init(mp, threads, k == 0, false, err)) { std::printf("cpu: %s\n", err.c_str()); return 1; }
        auto t0 = std::chrono::steady_clock::now();
        if (!c.encode(patches, plan.canvas, k == 0 ? cpu_bf16 : cpu_fp32)) { std::printf("cpu encode failed\n"); return 1; }
        std::printf("cpu encoder (%s): %.1f s on %d threads\n", k == 0 ? "bf16 roundings" : "fp32", secs(t0), threads);
    }
    if (W <= 0 || gpu.size() % W != 0) { std::printf("unexpected output size %zu\n", gpu.size()); return 1; }
    std::printf("%zu output rows of %d\n", gpu.size() / W, W);
    compare("cuda (dynamic) vs cpu fp32", gpu, cpu_fp32, W);
    compare("cpu bf16 roundings vs cpu fp32", cpu_bf16, cpu_fp32, W);
    compare("cuda vs cpu bf16 roundings", gpu, cpu_bf16, W);
    return 0;
}

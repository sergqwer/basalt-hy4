// hc_post_mix_norm: a sublayer's end and the next one's start in one pass over a token's four streams - hc_post, then
// hc_mixes_plain's 8 mixes (identity connections), then hc_pre_rms_norm - with the bits of those three kernels: the same
// per-thread orders, the same fixed-order block sums and the same expressions. The streams' new row stays in shared
// memory, so it is written once and read from there, where the three kernels read it from memory 3-4 times.
#include "basalt/kernels/mhc.hpp"

#include <cuda_runtime.h>

#include <atomic>
#include <stdexcept>

namespace basalt::kernels {
namespace {

constexpr int kThreads = 256;            // as the three kernels (hc_post's 128-thread blocks map a column to one thread too)
constexpr int kMix = 8, kParts = kHcSplits * (kMix + 1);
constexpr int kMaxVec = 8;               // row_norm.cu's float4 per thread

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int m = 16; m; m >>= 1) v += __shfl_xor_sync(0xffffffffu, v, m);
    return v;
}
__device__ __forceinline__ float sigmoidf(float v) { return 1.f / (1.f + expf(-v)); }

__global__ void __launch_bounds__(kThreads)
hc_post_mix_norm_kernel(const float* __restrict__ f, const float* __restrict__ x, float* coef, int n_embd,
                        const float* __restrict__ fn, const float* __restrict__ scale, const float* __restrict__ base,
                        float rms_eps, float hc_eps, float magnitude, const float* __restrict__ w, float eps,
                        float* __restrict__ xout, float* __restrict__ cur, float* __restrict__ out) {
    extern __shared__ __align__(16) float sm[];
    float* row = sm;                                   // the new streams [4][n_embd]
    __shared__ float red[kThreads / 32][kParts];
    __shared__ float part[kParts];
    __shared__ float cf[kHcCoef];
    const int t = blockIdx.x, n4 = n_embd >> 2, flat = kHcMult * n_embd, tid = threadIdx.x;
    float* crow = coef + (size_t) t * kHcCoef;
    if (tid < kHcCoef) cf[tid] = crow[tid];            // this sublayer's post and comb, read before the next mixes overwrite them
    __syncthreads();
    // hc_post_col, column i: out[d] = f post[d] + sum_s x[s] comb[s][d]
    {
        const float4* fv4 = reinterpret_cast<const float4*>(f + (size_t) t * n_embd);
        const float4* xv4 = reinterpret_cast<const float4*>(x + (size_t) t * flat);
        float4* ov4 = reinterpret_cast<float4*>(xout + (size_t) t * flat);
        float4* rv4 = reinterpret_cast<float4*>(row);
        for (int i = tid; i < n4; i += kThreads) {
            const float4 fv = fv4[i];
            float4 xs[kHcMult];
#pragma unroll
            for (int s = 0; s < kHcMult; ++s) xs[s] = xv4[(size_t) s * n4 + i];
#pragma unroll
            for (int d = 0; d < kHcMult; ++d) {
                const float post = cf[4 + d];
                float4 o = make_float4(fv.x * post, fv.y * post, fv.z * post, fv.w * post);
#pragma unroll
                for (int s = 0; s < kHcMult; ++s) {
                    const float c = cf[8 + s * 4 + d];
                    o.x = fmaf(xs[s].x, c, o.x); o.y = fmaf(xs[s].y, c, o.y);
                    o.z = fmaf(xs[s].z, c, o.z); o.w = fmaf(xs[s].w, c, o.w);
                }
                ov4[(size_t) d * n4 + i] = o;
                rv4[(size_t) d * n4 + i] = o;
            }
        }
    }
    __syncthreads();
    // plain_partial_kernel: split sp of the flat row, thread tid over float4 tid, tid + 256, ...; the 8 dots and (mix 0) the
    // sum of squares, each then block-summed in warp order
    {
        const int len4 = flat / kHcSplits / 4;
        const float4* rv4 = reinterpret_cast<const float4*>(row);
        for (int sp = 0; sp < kHcSplits; ++sp) {
            float dot[kMix], ss = 0.f;
#pragma unroll
            for (int m = 0; m < kMix; ++m) dot[m] = 0.f;
            for (int i = tid; i < len4; i += kThreads) {
                const float4 a = rv4[(size_t) sp * len4 + i];
#pragma unroll
                for (int m = 0; m < kMix; ++m) {
                    const float4 wv = reinterpret_cast<const float4*>(fn + (size_t) m * flat)[(size_t) sp * len4 + i];
                    dot[m] = fmaf(a.x, wv.x, dot[m]); dot[m] = fmaf(a.y, wv.y, dot[m]);
                    dot[m] = fmaf(a.z, wv.z, dot[m]); dot[m] = fmaf(a.w, wv.w, dot[m]);
                }
                ss = fmaf(a.x, a.x, ss); ss = fmaf(a.y, a.y, ss); ss = fmaf(a.z, a.z, ss); ss = fmaf(a.w, a.w, ss);
            }
#pragma unroll
            for (int m = 0; m < kMix; ++m) {
                const float v = warp_sum(dot[m]);
                if ((tid & 31) == 0) red[tid >> 5][sp * (kMix + 1) + m] = v;
            }
            const float v = warp_sum(ss);
            if ((tid & 31) == 0) red[tid >> 5][sp * (kMix + 1) + kMix] = v;
        }
    }
    __syncthreads();
    if (tid < kParts) {
        float r = 0.f;
#pragma unroll
        for (int i = 0; i < kThreads / 32; ++i) r += red[i][tid];
        part[tid] = r;
    }
    __syncthreads();
    // plain_coef_kernel (8 mixes), lane m of warp 0
    if (tid < 32) {
        const int lane = tid;
        float ss = 0.f;
        for (int s = 0; s < kHcSplits; ++s) ss += part[s * (kMix + 1) + kMix];
        const float rs = rsqrtf(ss / (float) flat + rms_eps);
        float m = 0.f;
        if (lane < kMix) {
            for (int s = 0; s < kHcSplits; ++s) m += part[s * (kMix + 1) + lane];
            m *= rs;
        }
        float o = 0.f;
        if (lane < 4) o = sigmoidf(m * scale[0] + base[lane]) + hc_eps;
        else if (lane < 8) o = magnitude * sigmoidf(m * scale[1] + base[lane]) + hc_eps;
        else if (lane < kHcCoef) o = ((lane - 8) >> 2) == ((lane - 8) & 3) ? 1.f : 0.f;
        if (lane < kHcCoef) {
            crow[lane] = o;
            cf[lane] = o;
        }
    }
    __syncthreads();
    // hc_pre_rms_norm_kernel on the new row
    float p[kHcMult];
#pragma unroll
    for (int s = 0; s < kHcMult; ++s) p[s] = cf[s];
    const float4* rv4 = reinterpret_cast<const float4*>(row);
    float4 v[kMaxVec];
#pragma unroll
    for (int c = 0; c < kMaxVec; ++c) {
        const int i = tid + c * kThreads;
        if (i < n4) {
            float4 acc = rv4[i];
            acc.x *= p[0]; acc.y *= p[0]; acc.z *= p[0]; acc.w *= p[0];
#pragma unroll
            for (int s = 1; s < kHcMult; ++s) {
                const float4 q = rv4[(size_t) s * n4 + i];
                acc.x = fmaf(q.x, p[s], acc.x); acc.y = fmaf(q.y, p[s], acc.y);
                acc.z = fmaf(q.z, p[s], acc.z); acc.w = fmaf(q.w, p[s], acc.w);
            }
            v[c] = acc;
            reinterpret_cast<float4*>(cur + (size_t) t * n_embd)[i] = acc;
        }
    }
    // rms_scale_store
    float ss = 0.f;
#pragma unroll
    for (int c = 0; c < kMaxVec; ++c) {
        if (tid + c * kThreads < n4) {
            ss = fmaf(v[c].x, v[c].x, ss);
            ss = fmaf(v[c].y, v[c].y, ss);
            ss = fmaf(v[c].z, v[c].z, ss);
            ss = fmaf(v[c].w, v[c].w, ss);
        }
    }
    ss = warp_sum(ss);
    if ((tid & 31) == 0) red[tid >> 5][0] = ss;
    __syncthreads();
    float tot = 0.f;
#pragma unroll
    for (int i = 0; i < kThreads / 32; ++i) tot += red[i][0];
    const float sc = rsqrtf(tot / (float) n_embd + eps);
    float* orow = out + (size_t) t * n_embd;
#pragma unroll
    for (int c = 0; c < kMaxVec; ++c) {
        const int i = tid + c * kThreads;
        if (i < n4) {
            const float4 wv = make_float4(w[4 * i], w[4 * i + 1], w[4 * i + 2], w[4 * i + 3]);
            reinterpret_cast<float4*>(orow)[i] =
                make_float4(v[c].x * sc * wv.x, v[c].y * sc * wv.y, v[c].z * sc * wv.z, v[c].w * sc * wv.w);
        }
    }
}

std::atomic<long long> g_launches{0};

// The row's dynamic shared memory plus the static arrays, against the card's opt-in limit per block (99 KiB on sm_120).
int smem_optin() {
    static const int v = [] {
        int dev = 0, b = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&b, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
        return b;
    }();
    return v;
}
constexpr size_t kStaticSmem = sizeof(float) * ((kThreads / 32) * kParts + kParts + kHcCoef);

}  // namespace

long long hc_post_mix_norm_launches() { return g_launches.load(std::memory_order_relaxed); }

bool hc_post_mix_norm_fits(int n_embd) {
    const int flat = kHcMult * n_embd;
    return n_embd % 4 == 0 && n_embd <= 4 * kMaxVec * kThreads && flat % (kHcSplits * 4) == 0 &&
           (size_t) flat * 4 + kStaticSmem <= (size_t) smem_optin();
}

void hc_post_mix_norm(const float* f, const float* x, float* coef, int n_tokens, int n_embd, const float* hc_fn, const float* hc_scale,
                      const float* hc_base, const HcPlainArgs& a, const float* norm_w, float eps, float* x_out, float* cur, float* out,
                      void* stream) {
    if (!hc_post_mix_norm_fits(n_embd)) throw std::invalid_argument("hc_post_mix_norm: geometry");
    const size_t smem = (size_t) kHcMult * n_embd * 4;
    static size_t attr = 0;   // the dynamic size the kernel is opened for
    if (smem > attr) {
        if (cudaFuncSetAttribute(hc_post_mix_norm_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem) != cudaSuccess)
            throw std::runtime_error("hc_post_mix_norm: shared memory");
        attr = smem;
    }
    hc_post_mix_norm_kernel<<<n_tokens, kThreads, smem, (cudaStream_t) stream>>>(f, x, coef, n_embd, hc_fn, hc_scale, hc_base, a.rms_eps,
                                                                                  a.hc_eps, a.magnitude, norm_w, eps, x_out, cur, out);
    if (cudaPeekAtLastError() == cudaSuccess) g_launches.fetch_add(1, std::memory_order_relaxed);   // only launches that started
    if (!stream) cudaStreamSynchronize(nullptr);
}

}  // namespace basalt::kernels

// A prompt chunk's router logits on the BF16 tensor cores: logits[t][e] = x[t] . gate[e] with x and gate fp32, each split
// into two BF16 terms (v = hi + lo, about 16 bits of v) and summed as hi.hi + hi.lo + lo.hi (lo.lo, below 2^-16 of each
// product, is left out), fp32 accumulation (mma m16n8k16). Not the bits of router()'s fp32 sums: the k order differs and
// the split drops about 2^-17 of each product, so a near tie of two experts can choose differently.
#include "basalt/kernels/router.hpp"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>

namespace basalt::kernels {
namespace {

constexpr int TT = 128, TE = 128, TK = 32, LD = TK + 8;   // block tile: tokens x experts, K step; smem row (bf16), padded
constexpr int kThreads = 256;                              // 8 warps: 2 along tokens (64 each) x 4 along experts (32 each)

struct Tiles {
    __nv_bfloat16 xh[TT][LD], xl[TT][LD], wh[TE][LD], wl[TE][LD];
};

__device__ __forceinline__ void split4(const float4 v, __nv_bfloat16* hi, __nv_bfloat16* lo) {
    const float f[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const __nv_bfloat16 h = __float2bfloat16_rn(f[i]);
        hi[i] = h;
        lo[i] = __float2bfloat16_rn(f[i] - __bfloat162float(h));
    }
}

__device__ __forceinline__ void ldsm4(uint32_t* r, const void* p) {
    const uint32_t a = (uint32_t) __cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(a));
}

__device__ __forceinline__ void mma(float* c, const uint32_t* a, const uint32_t* b) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// Each thread moves 4 float4 of x and 4 of the gate per K step: row (tid / 8) + 32 i, columns 4 (tid % 8).
__device__ __forceinline__ void fetch(const float* __restrict__ x, const float* __restrict__ gate, int64_t t0, int e0,
                                      int64_t T, int K, int k0, float4* xv, float4* wv) {
    const int r = threadIdx.x >> 3, c = (threadIdx.x & 7) * 4;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int64_t t = min(t0 + r + 32 * i, T - 1);
        xv[i] = *reinterpret_cast<const float4*>(x + t * K + k0 + c);
        wv[i] = *reinterpret_cast<const float4*>(gate + (int64_t) (e0 + r + 32 * i) * K + k0 + c);
    }
}

__global__ void __launch_bounds__(kThreads) router_logits_bf16x3_kernel(const float* __restrict__ x, const float* __restrict__ gate,
                                                                       int64_t T, int n_expert, int K, float* __restrict__ logits) {
    __shared__ __align__(16) Tiles s;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int wt = (warp >> 2) * 64, we = (warp & 3) * 32;   // the warp's tokens and experts in the tile
    const int64_t t0 = (int64_t) blockIdx.x * TT;
    const int e0 = blockIdx.y * TE;
    float acc[4][4][4] = {};   // [token frag 16][expert frag 8][c]
    float4 xv[4], wv[4];
    fetch(x, gate, t0, e0, T, K, 0, xv, wv);
    const int r = threadIdx.x >> 3, c = (threadIdx.x & 7) * 4;
    for (int k0 = 0; k0 < K; k0 += TK) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            split4(xv[i], &s.xh[r + 32 * i][c], &s.xl[r + 32 * i][c]);
            split4(wv[i], &s.wh[r + 32 * i][c], &s.wl[r + 32 * i][c]);
        }
        __syncthreads();
        if (k0 + TK < K) fetch(x, gate, t0, e0, T, K, k0 + TK, xv, wv);   // the next step's loads run under this step's mma
#pragma unroll
        for (int kk = 0; kk < TK; kk += 16) {
            uint32_t ah[4][4], al[4][4], bh[4][2], bl[4][2];
#pragma unroll
            for (int m = 0; m < 4; ++m) {
                const int row = wt + m * 16 + (lane & 15), col = kk + (lane >> 4) * 8;
                ldsm4(ah[m], &s.xh[row][col]);
                ldsm4(al[m], &s.xl[row][col]);
            }
#pragma unroll
            for (int n = 0; n < 4; n += 2) {   // x4: expert frags n and n + 1
                const int row = we + n * 8 + (lane & 7) + ((lane >> 4) << 3), col = kk + ((lane >> 3) & 1) * 8;
                uint32_t q[4];
                ldsm4(q, &s.wh[row][col]);
                bh[n][0] = q[0]; bh[n][1] = q[1]; bh[n + 1][0] = q[2]; bh[n + 1][1] = q[3];
                ldsm4(q, &s.wl[row][col]);
                bl[n][0] = q[0]; bl[n][1] = q[1]; bl[n + 1][0] = q[2]; bl[n + 1][1] = q[3];
            }
#pragma unroll
            for (int m = 0; m < 4; ++m)
#pragma unroll
                for (int n = 0; n < 4; ++n) {
                    mma(acc[m][n], al[m], bh[n]);   // the small terms first
                    mma(acc[m][n], ah[m], bl[n]);
                    mma(acc[m][n], ah[m], bh[n]);
                }
        }
        __syncthreads();
    }
    const int g = lane >> 2, q2 = (lane & 3) * 2;
#pragma unroll
    for (int m = 0; m < 4; ++m)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int64_t t = t0 + wt + m * 16 + g + h * 8;
            if (t >= T) continue;
#pragma unroll
            for (int n = 0; n < 4; ++n) {
                const int e = e0 + we + n * 8 + q2;
                *reinterpret_cast<float2*>(logits + t * n_expert + e) = make_float2(acc[m][n][2 * h], acc[m][n][2 * h + 1]);
            }
        }
}

}  // namespace

bool router_logits_bf16x3_fits(int n_expert, int n_embd) { return n_expert > 0 && n_expert % TE == 0 && n_embd > 0 && n_embd % TK == 0; }

void router_logits_bf16x3(const float* x, const float* gate, int n_tokens, int n_expert, int n_embd, float* logits, void* stream) {
    if (!router_logits_bf16x3_fits(n_expert, n_embd) || n_tokens < 1)
        throw std::invalid_argument("router_logits_bf16x3: unsupported geometry");
    const dim3 grid((unsigned) ((n_tokens + TT - 1) / TT), (unsigned) (n_expert / TE));
    router_logits_bf16x3_kernel<<<grid, kThreads, 0, (cudaStream_t) stream>>>(x, gate, n_tokens, n_expert, n_embd, logits);
}

}  // namespace basalt::kernels

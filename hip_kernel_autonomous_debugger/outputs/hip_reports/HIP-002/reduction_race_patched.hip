// HIP-002 — Block-level parallel reduction missing __syncthreads().
//
// Tree reduction in shared memory: each round halves the active thread
// count. Between rounds we MUST call __syncthreads() so every thread
// sees the writes from the previous round. This kernel deliberately
// omits the barrier, producing sums that are systematically lower than
// the correct total (later strides read stale values).
//
// Expected verdict:  STATUS: FAIL_NUMERIC (GPU sum ≠ CPU sum)
// Fix:               Add __syncthreads() at the top of each `for` iteration.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <vector>

#define BLOCK 256

#define HIP_CHECK(expr) do {                                         \
    hipError_t _e = (expr);                                          \
    if (_e != hipSuccess) {                                          \
        printf("HIP_ERR: %s\n", hipGetErrorString(_e));              \
        printf("STATUS: FAIL_RUNTIME\n");                            \
        return 2;                                                    \
    }                                                                \
} while (0)

__global__ void reduce_sum(const float* __restrict__ in, float* __restrict__ out, int N) {
    __shared__ float sdata[BLOCK];
    int tid   = threadIdx.x;
    int idx   = blockIdx.x * blockDim.x + tid;
    sdata[tid] = (idx < N) ? in[idx] : 0.0f;
    __syncthreads();
    // BUG: Missing __syncthreads() here — threads race to read/write sdata

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        __syncthreads();
        // BUG: no __syncthreads() at top of loop either
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
    }

    if (tid == 0) atomicAdd(out, sdata[0]);
}

int main() {
    const int N = 1 << 20;
    const size_t bytes = N * sizeof(float);
    std::vector<float> ha(N);
    double ref = 0.0;
    for (int i = 0; i < N; ++i) { ha[i] = 1.0f; ref += ha[i]; }

    float *da, *dsum;
    HIP_CHECK(hipMalloc(&da, bytes));
    HIP_CHECK(hipMalloc(&dsum, sizeof(float)));
    HIP_CHECK(hipMemcpy(da, ha.data(), bytes, hipMemcpyHostToDevice));

    dim3 block(BLOCK), grid((N + BLOCK - 1) / BLOCK);

    // Run the kernel a few times and report the first + spread — race manifests as run-to-run variation.
    float sums[5] = {0.f};
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int r = 0; r < 5; ++r) {
        float zero = 0.f;
        HIP_CHECK(hipMemcpy(dsum, &zero, sizeof(float), hipMemcpyHostToDevice));
        reduce_sum<<<grid, block>>>(da, dsum, N);
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipMemcpy(&sums[r], dsum, sizeof(float), hipMemcpyDeviceToHost));
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    printf("PROBLEM: reduce_sum\n");
    printf("N: %d\n", N);
    printf("ELAPSED_MS: %.3f\n", elapsed_ms);
    printf("HIP_ERR: NONE\n");
    printf("GPU_RESULT: %.1f %.1f %.1f %.1f %.1f\n",
           sums[0], sums[1], sums[2], sums[3], sums[4]);
    printf("CPU_REFERENCE: %.1f\n", ref);

    float smin = sums[0], smax = sums[0], sfirst = sums[0];
    for (int r = 1; r < 5; ++r) { smin = fminf(smin, sums[r]); smax = fmaxf(smax, sums[r]); }
    printf("RUN_SPREAD: min=%.1f max=%.1f delta=%.1f\n", smin, smax, smax - smin);
    float max_err = fmaxf(fabsf(smax - float(ref)), fabsf(smin - float(ref)));
    printf("MAX_ABS_ERR: %.3f\n", max_err);

    hipFree(da); hipFree(dsum);

    if (max_err > 0.5f) {
        printf("STATUS: FAIL_NUMERIC\n");
        return 1;
    }
    printf("STATUS: PASS\n");
    return 0;
}

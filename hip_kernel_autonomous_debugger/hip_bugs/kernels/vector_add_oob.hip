// HIP-001 — Vector Add with off-by-one out-of-bounds read/write.
//
// The kernel iterates `for (int i = tid; i <= N; i += stride)` — the `<=`
// causes thread 0 (and grid-stride winners) to touch index N, which is one
// past the allocated buffer. On BW151 this manifests as either a wrong
// value in c[0] (garbage read from a[N]) or a HIP_ERR_ILLEGAL_ADDRESS
// depending on the guarded region.
//
// Expected verdict:  STATUS: FAIL_NUMERIC  or  STATUS: FAIL_RUNTIME
// Fix:               Change `i <= N` → `i < N`.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <cmath>
#include <vector>

#define HIP_CHECK(expr) do {                                         \
    hipError_t _e = (expr);                                          \
    if (_e != hipSuccess) {                                          \
        printf("HIP_ERR: %s\n", hipGetErrorString(_e));              \
        printf("STATUS: FAIL_RUNTIME\n");                            \
        return 2;                                                    \
    }                                                                \
} while (0)

__global__ void vector_add(const float* a, const float* b, float* c, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    // BUG: `<= N` reads/writes index N which is out of bounds
    for (int i = tid; i <= N; i += stride) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    const int N = 1 << 18;                // 262144
    const size_t bytes = N * sizeof(float);

    std::vector<float> ha(N), hb(N), hc(N, 0.f), href(N);
    for (int i = 0; i < N; ++i) {
        ha[i] = float(i) * 0.001f;
        hb[i] = float(N - i) * 0.002f;
        href[i] = ha[i] + hb[i];
    }

    float *da, *db, *dc;
    HIP_CHECK(hipMalloc(&da, bytes));
    HIP_CHECK(hipMalloc(&db, bytes));
    HIP_CHECK(hipMalloc(&dc, bytes));
    HIP_CHECK(hipMemcpy(da, ha.data(), bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(db, hb.data(), bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(dc, 0, bytes));

    dim3 block(256), grid((N + 255) / 256);
    HIP_CHECK(hipDeviceSynchronize());
    auto t0 = std::chrono::high_resolution_clock::now();
    vector_add<<<grid, block>>>(da, db, dc, N);
    hipError_t launch_err = hipGetLastError();
    if (launch_err != hipSuccess) {
        printf("HIP_ERR: launch failed: %s\n", hipGetErrorString(launch_err));
        printf("STATUS: FAIL_LAUNCH\n");
        return 3;
    }
    hipError_t sync_err = hipDeviceSynchronize();
    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    HIP_CHECK(hipMemcpy(hc.data(), dc, bytes, hipMemcpyDeviceToHost));

    printf("PROBLEM: vector_add\n");
    printf("N: %d\n", N);
    printf("ELAPSED_MS: %.3f\n", elapsed_ms);

    if (sync_err != hipSuccess) {
        printf("HIP_ERR: %s\n", hipGetErrorString(sync_err));
        printf("STATUS: FAIL_RUNTIME\n");
        return 4;
    }
    printf("HIP_ERR: NONE\n");

    // Check correctness at a few sample indices and compute max abs err
    float max_err = 0.f;
    int first_bad = -1;
    for (int i = 0; i < N; ++i) {
        float e = fabsf(hc[i] - href[i]);
        if (e > max_err) max_err = e;
        if (e > 1e-4f && first_bad < 0) first_bad = i;
    }
    printf("MAX_ABS_ERR: %.6e\n", max_err);

    printf("GPU_RESULT:");
    for (int i = 0; i < 8 && i < N; ++i) printf(" %.3f", hc[i]);
    printf("\n");
    printf("CPU_REFERENCE:");
    for (int i = 0; i < 8 && i < N; ++i) printf(" %.3f", href[i]);
    printf("\n");

    hipFree(da); hipFree(db); hipFree(dc);

    if (first_bad >= 0 || max_err > 1e-4f) {
        printf("FIRST_MISMATCH: %d\n", first_bad);
        printf("STATUS: FAIL_NUMERIC\n");
        return 1;
    }
    printf("STATUS: PASS\n");
    return 0;
}

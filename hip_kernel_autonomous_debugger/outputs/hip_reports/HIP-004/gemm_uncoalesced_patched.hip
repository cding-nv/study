// HIP-004 — Naive SGEMM with uncoalesced global memory access on A.
//
// Row-major matrices A(MxK), B(KxN), C(MxN). Two threads in the same
// wave differ in threadIdx.x, which maps to `col`. In the buggy version
// A is accessed as `A[k * M + row]` — but A is row-major (row * K + k),
// so `k * M + row` produces a column-major stride pattern that scatters
// each wave's loads across K rows. Correctness is unaffected because the
// full accumulate still touches every needed element, but the load
// pattern is completely uncoalesced.
//
// (Note: on real DCUs the "wave" is 64 threads; the coalescing win is
// still ~2× at these sizes because global memory is much slower than LDS.)
//
// Expected verdict:  STATUS: FAIL_NUMERIC (index formula is actually wrong; see below)
//   Wait — using `A[k*M+row]` on a MxK row-major matrix actually reads the
//   wrong elements (transpose of what we want). So this manifests as
//   a numeric failure, not just a performance issue. Good — the AI must
//   figure out that both correctness AND coalescing are broken by the
//   same swapped index.
// Fix:               `A[row * K + k]` (row-major coalesced access).
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <vector>

#define BLOCK 16

#define HIP_CHECK(expr) do {                                         \
    hipError_t _e = (expr);                                          \
    if (_e != hipSuccess) {                                          \
        printf("HIP_ERR: %s\n", hipGetErrorString(_e));              \
        printf("STATUS: FAIL_RUNTIME\n");                            \
        return 2;                                                    \
    }                                                                \
} while (0)

// C(MxN) = A(MxK) * B(KxN), all row-major.
__global__ void sgemm_naive(const float* __restrict__ A,
                            const float* __restrict__ B,
                            float* __restrict__ C,
                            int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;

    float acc = 0.f;
    for (int k = 0; k < K; ++k) {
        float a = A[row * K + k];
        float b = B[k * N + col];
        acc += a * b;
    }
    C[row * N + col] = acc;
}

int main() {
    const int M = 256, N = 256, K = 256;
    std::vector<float> hA(M * K), hB(K * N), hC(M * N, 0.f), href(M * N, 0.f);
    for (int i = 0; i < M * K; ++i) hA[i] = float((i % 7) - 3);
    for (int i = 0; i < K * N; ++i) hB[i] = float((i % 11) - 5);
    // CPU reference (correct A[row*K + k])
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
            float s = 0.f;
            for (int k = 0; k < K; ++k) s += hA[m * K + k] * hB[k * N + n];
            href[m * N + n] = s;
        }

    float *dA, *dB, *dC;
    HIP_CHECK(hipMalloc(&dA, M * K * sizeof(float)));
    HIP_CHECK(hipMalloc(&dB, K * N * sizeof(float)));
    HIP_CHECK(hipMalloc(&dC, M * N * sizeof(float)));
    HIP_CHECK(hipMemcpy(dA, hA.data(), M * K * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dB, hB.data(), K * N * sizeof(float), hipMemcpyHostToDevice));

    dim3 block(BLOCK, BLOCK);
    dim3 grid((N + BLOCK - 1) / BLOCK, (M + BLOCK - 1) / BLOCK);

    // Warm-up + time
    sgemm_naive<<<grid, block>>>(dA, dB, dC, M, N, K);
    HIP_CHECK(hipDeviceSynchronize());

    const int ITERS = 10;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < ITERS; ++i) sgemm_naive<<<grid, block>>>(dA, dB, dC, M, N, K);
    HIP_CHECK(hipDeviceSynchronize());
    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / ITERS;

    HIP_CHECK(hipMemcpy(hC.data(), dC, M * N * sizeof(float), hipMemcpyDeviceToHost));

    printf("PROBLEM: sgemm\n");
    printf("N: %d\n", M * N);
    printf("ELAPSED_MS: %.3f\n", elapsed_ms);
    printf("HIP_ERR: NONE\n");

    float max_err = 0.f;
    int first_bad = -1;
    for (int i = 0; i < M * N; ++i) {
        float e = fabsf(hC[i] - href[i]);
        if (e > max_err) max_err = e;
        if (e > 1e-2f && first_bad < 0) first_bad = i;
    }
    printf("MAX_ABS_ERR: %.3e\n", max_err);
    printf("GPU_RESULT: %.1f %.1f %.1f %.1f\n", hC[0], hC[1], hC[2], hC[3]);
    printf("CPU_REFERENCE: %.1f %.1f %.1f %.1f\n", href[0], href[1], href[2], href[3]);

    hipFree(dA); hipFree(dB); hipFree(dC);
    if (first_bad >= 0) {
        printf("FIRST_MISMATCH: %d\n", first_bad);
        printf("STATUS: FAIL_NUMERIC\n");
        return 1;
    }
    printf("STATUS: PASS\n");
    return 0;
}

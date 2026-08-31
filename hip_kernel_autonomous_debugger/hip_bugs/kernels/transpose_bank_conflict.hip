// HIP-003 — Matrix transpose with LDS bank conflicts.
//
// Tiled transpose using __shared__ float tile[32][32]. Because DCU LDS
// has 32 banks × 4 bytes, columns of a 32-wide tile map to a single bank
// for every row → every column read serializes across 32 threads of a
// wave (wave = 64 on BW151, but the 32-bank layout still forces heavy
// serialization on the tile[threadIdx.y][threadIdx.x] transpose pattern).
//
// This is a PERFORMANCE bug: result is numerically correct but the
// kernel is markedly slower than a padded version.
//
// Expected verdict:  STATUS: PASS (correctness) but slow (baseline vs padded)
// Fix:               `__shared__ float tile[32][33];`  (+1 column padding)
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <vector>

#define TILE 32

#define HIP_CHECK(expr) do {                                         \
    hipError_t _e = (expr);                                          \
    if (_e != hipSuccess) {                                          \
        printf("HIP_ERR: %s\n", hipGetErrorString(_e));              \
        printf("STATUS: FAIL_RUNTIME\n");                            \
        return 2;                                                    \
    }                                                                \
} while (0)

__global__ __launch_bounds__(1024)
void transpose(const float* __restrict__ in, float* __restrict__ out, int width, int height) {
    // BUG: [TILE][TILE] causes systematic LDS bank conflicts on column access.
    __shared__ float tile[TILE][TILE];

    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;
    if (x < width && y < height) {
        tile[threadIdx.y][threadIdx.x] = in[y * width + x];
    }
    __syncthreads();

    int tx = blockIdx.y * TILE + threadIdx.x;
    int ty = blockIdx.x * TILE + threadIdx.y;
    if (tx < height && ty < width) {
        out[ty * height + tx] = tile[threadIdx.x][threadIdx.y];  // column read → conflicts
    }
}

int main() {
    const int W = 2048, H = 2048;
    const size_t bytes = size_t(W) * H * sizeof(float);

    std::vector<float> hin(size_t(W) * H), hout(size_t(W) * H, 0.f);
    for (int i = 0; i < W * H; ++i) hin[i] = float(i % 997);

    float *din, *dout;
    HIP_CHECK(hipMalloc(&din, bytes));
    HIP_CHECK(hipMalloc(&dout, bytes));
    HIP_CHECK(hipMemcpy(din, hin.data(), bytes, hipMemcpyHostToDevice));

    dim3 block(TILE, TILE), grid((W + TILE - 1) / TILE, (H + TILE - 1) / TILE);

    // Warm-up
    transpose<<<grid, block>>>(din, dout, W, H);
    HIP_CHECK(hipDeviceSynchronize());

    // Time N iterations for stable measurement
    const int ITERS = 20;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < ITERS; ++i) transpose<<<grid, block>>>(din, dout, W, H);
    HIP_CHECK(hipDeviceSynchronize());
    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / ITERS;

    HIP_CHECK(hipMemcpy(hout.data(), dout, bytes, hipMemcpyDeviceToHost));

    printf("PROBLEM: transpose\n");
    printf("N: %d\n", W * H);
    printf("ELAPSED_MS: %.3f\n", elapsed_ms);
    printf("HIP_ERR: NONE\n");

    // Sample correctness
    float max_err = 0.f;
    int first_bad = -1;
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            float ref = hin[y * W + x];
            float got = hout[x * H + y];
            float e = fabsf(got - ref);
            if (e > max_err) max_err = e;
            if (e > 1e-4f && first_bad < 0) first_bad = y * W + x;
        }
    }
    printf("MAX_ABS_ERR: %.6e\n", max_err);
    printf("GPU_RESULT: %.1f %.1f %.1f %.1f\n",
           hout[0], hout[H], hout[2 * H], hout[3 * H]);
    printf("CPU_REFERENCE: %.1f %.1f %.1f %.1f\n",
           hin[0], hin[1], hin[2], hin[3]);

    hipFree(din); hipFree(dout);
    if (first_bad >= 0) {
        printf("STATUS: FAIL_NUMERIC\n");
        return 1;
    }
    // Correctness passes; PERFORMANCE bug is visible in ELAPSED_MS relative to the padded fix.
    printf("STATUS: PASS\n");
    return 0;
}

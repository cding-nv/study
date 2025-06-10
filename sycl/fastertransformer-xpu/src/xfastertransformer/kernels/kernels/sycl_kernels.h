#include <sycl/sycl.hpp>
#include <dpct/device.hpp>

using namespace dpct;
using dtype = sycl::half;

#define MAX_SEQ_LEN 2048

void accum(sycl::half *a, sycl::half *b, int size);

void rmsnorm(sycl::half *o, sycl::half *x, sycl::half *weight, int size) ;

void matmul(
    dtype* xout, dtype* x,
    dtype* w,
    int n, int d, int batch = 1,
    int x_stride = 0, int w_stride = 0, int op_stride = 0, int w_row_stride = -1,
    float alpha = 1.0f);

void matmul_f32(
    float* xout, dtype* x,
    dtype* w,
    int n, int d, int batch = 1,
    int x_stride = 0, int w_stride = 0, int op_stride = 0, int w_row_stride = -1,
    float alpha = 1.0f);

void matmul_mad(
    dtype* xout, dtype* x,
    dtype* w,
    int n, int d, int batch = 1,
    int x_stride = 0, int w_stride = 0, int op_stride = 0, int w_row_stride = -1,
    float alpha = 1.0f);

void matmul_qkv(
    dtype* qout, dtype* kout, dtype* vout, dtype* x,
    dtype* wq, dtype* wk, dtype* wv,
    int cols, int dh, int Nh);

void matmul_2X(
    dtype* out1, dtype* out2, 
    dtype* input, dtype* w1, dtype* w2, int n_cols, int n_rows,
    int batch = 1, int x_stride = 0, int w_stride = 0, int op_stride = 0, int w_row_stride = -1,
    float alpha = 1.0f);

void RoPERotation(sycl::half *q, sycl::half *k, int pos, int num_heads, int num_kv_heads, int head_size);

void MultiHeadAttention(sycl::half *output,
                        sycl::half *q, sycl::half *key_cache, sycl::half *val_cache,
                        sycl::half *att,
                        int num_heads, int head_size, int seq_len);

void siluElementwiseMul(sycl::half *hb, sycl::half *hb2, int size);

int sample_argmax(float *probabilities, int n);

int divUp(int a, int b);
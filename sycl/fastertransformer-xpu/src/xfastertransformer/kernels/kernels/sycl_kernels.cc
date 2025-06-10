#include "sycl_kernels.h"

// ----------------------------------------------------------------------------
// GPU kernels

SYCL_EXTERNAL void scalar_mul32_kernel(float *arr, float value, int size,
                                       const sycl::nd_item<1> &item_ct1) {
    int i = item_ct1.get_group(0) * item_ct1.get_local_range(0) +
            item_ct1.get_local_id(0);
    if (i < size)
        arr[i] = arr[i] * value;
}

SYCL_EXTERNAL void element_wise_add_kernel(sycl::half *dest, sycl::half *src,
                                           int size,
                                           const sycl::nd_item<1> &item_ct1) {
    int i = item_ct1.get_group(0) * item_ct1.get_local_range(0) +
            item_ct1.get_local_id(0);
    if (i < size)
        dest[i] = (sycl::half)((float)dest[i] + (float)src[i]);
}

SYCL_EXTERNAL void convert_fp32_to_fp16(sycl::half *out, float *in, int size,
                                        const sycl::nd_item<1> &item_ct1) {
    int index = item_ct1.get_group(0) * item_ct1.get_local_range(0) +
                item_ct1.get_local_id(0);
    if (index < size)
        out[index] = (sycl::half)in[index];
}

SYCL_EXTERNAL void convert_fp16_to_fp32(float *out, sycl::half *in, int size, const sycl::nd_item<1> &item_ct1) {
    int index = item_ct1.get_group(0) * item_ct1.get_local_range(0) + item_ct1.get_local_id(0);
    if (index < size)
        out[index] = (float)in[index];
}

// Single block - not enough parallelism for the GPU, but it's just 1% of total time
SYCL_EXTERNAL void rmsnorm_kernel(sycl::half *o, sycl::half *x, sycl::half *weight, int size,
                    int elementsPerThread, const sycl::nd_item<1> &item_ct1,
                    float &shared_ss) {
    // float ss = 0.0f;
    // for (int index = item_ct1.get_local_id(0); index < size; index += 512) {
    //     float val = x[index];
    //     ss += (val * val);
    // }
    dtype ip[8];
    dtype wt[8];
    dtype vx[8];

    int j = item_ct1.get_local_id(0) * 8;

    *((sycl::uint4 *)(&ip)) = *((sycl::uint4 *)(&x[j]));
    *((sycl::uint4 *)(&wt)) = *((sycl::uint4 *)(&weight[j]));

    float ss = (float)ip[0] * (float)ip[0] + (float)ip[1] * (float)ip[1] + (float)ip[2] * (float)ip[2] + (float)ip[3] * (float)ip[3] + (float)ip[4] * (float)ip[4] + (float)ip[5] * (float)ip[5] + (float)ip[6] * (float)ip[6] + (float)ip[7] * (float)ip[7];

    ss = sycl::reduce_over_group(item_ct1.get_group(), ss, sycl::plus<>());

    if (item_ct1.get_local_id(0) == 0) {
        ss /= size;
        ss += 1e-5f;
        ss = 1.0f / sycl::sqrt(ss);
        shared_ss = ss;
    }
    item_ct1.barrier(sycl::access::fence_space::local_space);
    ss = shared_ss;

    // // normalize
    // for (int index = item_ct1.get_local_id(0); index < size; index += 512) {
    //     float val = ((float)x[index]) * ss * (float)weight[index];
    //     o[index] = (sycl::half)val;
    // }

    vx[0] = ip[0] * ss * wt[0];
    vx[1] = ip[1] * ss * wt[1];
    vx[2] = ip[2] * ss * wt[2];
    vx[3] = ip[3] * ss * wt[3];
    vx[4] = ip[4] * ss * wt[4];
    vx[5] = ip[5] * ss * wt[5];
    vx[6] = ip[6] * ss * wt[6];
    vx[7] = ip[7] * ss * wt[7];
    *((sycl::uint4 *)(&o[j])) = *((sycl::uint4 *)(&vx));
}

// Note that ~95% of total time is spent here, so optimizing this is important
// 1. One output generated per warp so that we can parallelize the dot product across the warp
// 2. We load 8 elements at a time for efficiency (assume dimensions to be multiple of 8)
SYCL_EXTERNAL void mat_vec_kernel(
    dtype* output,
    const dtype* __restrict__ input,
    const dtype* __restrict__ weight,
    int n,
    int d,
    int numSerialLoads, int input_stride, int weight_stride, int output_stride, int weight_row_stride, float alpha, const sycl::nd_item<3>& item) {

    int index = item.get_group(2) * item.get_local_range(1) + item.get_local_id(1);
    if (index >= d) return;

    input  += item.get_group(1) * input_stride;
    weight += item.get_group(1) * weight_stride + index * weight_row_stride;
    output += item.get_group(1) * output_stride;

    float sum = 0;

    for (int i = 0; i < numSerialLoads; i++) {
        int j = (i * 32 + item.get_local_id(2)) * 8;
        if (j < n) {
            dtype wt[8];
            dtype ip[8];

            *((sycl::uint4 *)(&wt)) = *((sycl::uint4 *)(&weight[j]));
            *((sycl::uint4 *)(&ip)) = *((sycl::uint4 *)(&input[j] ));
            
            // for (int el = 0; el < 8; el++) {
            //     sum += float(wt[el]) * float(ip[el]);
            // }
            sum += wt[0] * ip[0] + wt[1] * ip[1] + wt[2] * ip[2] + wt[3] * ip[3] + wt[4] * ip[4] + wt[5] * ip[5] + wt[6] * ip[6] + wt[7] * ip[7];
        }
    }

    sum = sycl::reduce_over_group(item.get_sub_group(), sum, sycl::plus<>());

    // sum *= alpha;

    if (item.get_local_id(2) == 0) {
        output[index] = (dtype)sum;
    }
}

SYCL_EXTERNAL void mat_vec_f32_kernel(
    float* output,
    const dtype* __restrict__ input,
    const dtype* __restrict__ weight,
    int n,
    int d,
    int numSerialLoads, int input_stride, int weight_stride, int output_stride, int weight_row_stride, float alpha, const sycl::nd_item<3>& item) {

    int index = item.get_group(2) * item.get_local_range(1) + item.get_local_id(1);
    if (index >= d) return;

    input  += item.get_group(1) * input_stride;
    weight += item.get_group(1) * weight_stride + index * weight_row_stride;
    output += item.get_group(1) * output_stride;

    float sum = 0;

    for (int i = 0; i < numSerialLoads; i++) {
        int j = (i * 32 + item.get_local_id(2)) * 8;
        if (j < n) {
            dtype wt[8];
            dtype ip[8];

            *((sycl::uint4 *)(&wt)) = *((sycl::uint4 *)(&weight[j]));
            *((sycl::uint4 *)(&ip)) = *((sycl::uint4 *)(&input[j] ));
            
            // for (int el = 0; el < 8; el++) {
            //     sum += float(wt[el]) * float(ip[el]);
            // }
            sum += wt[0] * ip[0] + wt[1] * ip[1] + wt[2] * ip[2] + wt[3] * ip[3] + wt[4] * ip[4] + wt[5] * ip[5] + wt[6] * ip[6] + wt[7] * ip[7];
        }
    }

    sum = sycl::reduce_over_group(item.get_sub_group(), sum, sycl::plus<>());

    // sum *= alpha;

    if (item.get_local_id(2) == 0) {
        output[index] = sum;
    }
}

SYCL_EXTERNAL void mat_vec_mad_kernel(
    dtype* output,
    const dtype* __restrict__ input,
    const dtype* __restrict__ weight,
    int n,
    int d,
    int numSerialLoads, int input_stride, int weight_stride, int output_stride, int weight_row_stride, float alpha, const sycl::nd_item<3>& item) {

    int index = item.get_group(2) * item.get_local_range(1) + item.get_local_id(1);
    if (index >= d) return;

    input  += item.get_group(1) * input_stride;
    weight += item.get_group(1) * weight_stride + index * weight_row_stride;
    output += item.get_group(1) * output_stride;

    float sum = 0;

    for (int i = 0; i < numSerialLoads; i++) {
        int j = (i * 32 + item.get_local_id(2)) * 8;
        if (j < n) {
            dtype wt[8];
            dtype ip[8];

            *((sycl::uint4 *)(&wt)) = *((sycl::uint4 *)(&weight[j]));
            *((sycl::uint4 *)(&ip)) = *((sycl::uint4 *)(&input[j] ));
            
            // for (int el = 0; el < 8; el++) {
            //     sum += float(wt[el]) * float(ip[el]);
            // }
            sum += wt[0] * ip[0] + wt[1] * ip[1] + wt[2] * ip[2] + wt[3] * ip[3] + wt[4] * ip[4] + wt[5] * ip[5] + wt[6] * ip[6] + wt[7] * ip[7];
        }
    }

    sum = sycl::reduce_over_group(item.get_sub_group(), sum, sycl::plus<>());

    // sum *= alpha;

    if (item.get_local_id(2) == 0) {
        output[index] = (dtype)((float)output[index] + sum);
    }
}

// SYCL_EXTERNAL void mat_vec_qkv_kernel(
//     dtype* qout,
//     dtype* kout,
//     dtype* vout,
//     const dtype* __restrict__ input,
//     const dtype* __restrict__ weightq,
//     const dtype* __restrict__ weightk,
//     const dtype* __restrict__ weightv,
//     int n,
//     int d,
//     int numSerialLoads, int input_stride, int weight_stride, int output_stride, int weight_row_stride, float alpha, const sycl::nd_item<3>& item) {

//     int index = item.get_group(2) * item.get_local_range(1) + item.get_local_id(1);
//     if (index >= d) return;

//     input   += item.get_group(1) *  input_stride;
//     weightq += item.get_group(1) * weight_stride + index * weight_row_stride;
//     weightk += item.get_group(1) * weight_stride + index * weight_row_stride;
//     weightv += item.get_group(1) * weight_stride + index * weight_row_stride;
//     qout    += item.get_group(1) * output_stride;
//     kout    += item.get_group(1) * output_stride;
//     vout    += item.get_group(1) * output_stride;

//     float sumq = 0;
//     float sumk = 0;
//     float sumv = 0;

//     for (int i = 0; i < numSerialLoads; i++) {
//         int j = (i * 32 + item.get_local_id(2)) * 8;

//         if (j < n) {
//             dtype ip[8];
//             dtype wq[8];
//             dtype wk[8];
//             dtype wv[8];

//             *((sycl::uint4 *)(&ip)) = *((sycl::uint4 *)(&input[j]  ));
//             *((sycl::uint4 *)(&wq)) = *((sycl::uint4 *)(&weightq[j]));
//             *((sycl::uint4 *)(&wk)) = *((sycl::uint4 *)(&weightk[j]));
//             *((sycl::uint4 *)(&wv)) = *((sycl::uint4 *)(&weightv[j]));

//             // for (int el = 0; el < 8; el++) {
//             //     sumq += float(wq[el]) * float(ip[el]);
//             //     sumk += float(wk[el]) * float(ip[el]);
//             //     sumv += float(wv[el]) * float(ip[el]);
//             // }
//             sumq += wq[0] * ip[0] + wq[1] * ip[1] + wq[2] * ip[2] + wq[3] * ip[3] + wq[4] * ip[4] + wq[5] * ip[5] + wq[6] * ip[6] + wq[7] * ip[7];
//             sumk += wk[0] * ip[0] + wk[1] * ip[1] + wk[2] * ip[2] + wk[3] * ip[3] + wk[4] * ip[4] + wk[5] * ip[5] + wk[6] * ip[6] + wk[7] * ip[7];
//             sumv += wv[0] * ip[0] + wv[1] * ip[1] + wv[2] * ip[2] + wv[3] * ip[3] + wv[4] * ip[4] + wv[5] * ip[5] + wv[6] * ip[6] + wv[7] * ip[7];
//         }
//     }

//     sumq = sycl::reduce_over_group(item.get_sub_group(), sumq, sycl::plus<>());
//     sumk = sycl::reduce_over_group(item.get_sub_group(), sumk, sycl::plus<>());
//     sumv = sycl::reduce_over_group(item.get_sub_group(), sumv, sycl::plus<>());

//     // sum *= alpha;

//     if (item.get_local_id(2) == 0) {
//         qout[index] = (dtype)sumq;
//         kout[index] = (dtype)sumk;
//         vout[index] = (dtype)sumv;
//         // vout[index * MAX_SEQ_LEN] = (dtype)sumv; // change 2
//     }
// }

SYCL_EXTERNAL void mat_vec_qkv_kernel(
    dtype* qout,
    dtype* kout,
    dtype* vout,
    const dtype* __restrict__ input,
    const dtype* __restrict__ weightq,
    const dtype* __restrict__ weightk,
    const dtype* __restrict__ weightv,
    int cols, int dh, int Nh, int numSerialLoads, const sycl::nd_item<3>& item) {

    int index = item.get_group(1) * dh + item.get_group(2) * item.get_local_range(1) + item.get_local_id(1);
    if (index >= Nh * dh) return;

    weightq += index * cols;
    weightk += index * cols;
    weightv += index * cols;
    qout    += item.get_group(1) * MAX_SEQ_LEN * dh + item.get_group(2) * item.get_local_range(1) + item.get_local_id(1);
    kout    += item.get_group(1) * MAX_SEQ_LEN * dh + item.get_group(2) * item.get_local_range(1) + item.get_local_id(1);
    vout    += item.get_group(1) * MAX_SEQ_LEN * dh + item.get_group(2) * item.get_local_range(1) + item.get_local_id(1);

    float sumq = 0;
    float sumk = 0;
    float sumv = 0;

    for (int i = 0; i < numSerialLoads; i++) {
        int j = (i * 32 + item.get_local_id(2)) * 8;
        
        if (j < cols) {
            dtype ip[8];
            dtype wq[8];
            dtype wk[8];
            dtype wv[8];

            *((sycl::uint4 *)(&ip)) = *((sycl::uint4 *)(&input[j]  ));
            *((sycl::uint4 *)(&wq)) = *((sycl::uint4 *)(&weightq[j]));
            *((sycl::uint4 *)(&wk)) = *((sycl::uint4 *)(&weightk[j]));
            *((sycl::uint4 *)(&wv)) = *((sycl::uint4 *)(&weightv[j]));
            
            // for (int el = 0; el < 8; el++) {
            //     sumq += float(wq[el]) * float(ip[el]);
            //     sumk += float(wk[el]) * float(ip[el]);
            //     sumv += float(wv[el]) * float(ip[el]);
            // }
            sumq += wq[0] * ip[0] + wq[1] * ip[1] + wq[2] * ip[2] + wq[3] * ip[3] + wq[4] * ip[4] + wq[5] * ip[5] + wq[6] * ip[6] + wq[7] * ip[7];
            sumk += wk[0] * ip[0] + wk[1] * ip[1] + wk[2] * ip[2] + wk[3] * ip[3] + wk[4] * ip[4] + wk[5] * ip[5] + wk[6] * ip[6] + wk[7] * ip[7];
            sumv += wv[0] * ip[0] + wv[1] * ip[1] + wv[2] * ip[2] + wv[3] * ip[3] + wv[4] * ip[4] + wv[5] * ip[5] + wv[6] * ip[6] + wv[7] * ip[7];
        }
    }

    sumq = sycl::reduce_over_group(item.get_sub_group(), sumq, sycl::plus<>());
    sumk = sycl::reduce_over_group(item.get_sub_group(), sumk, sycl::plus<>());
    sumv = sycl::reduce_over_group(item.get_sub_group(), sumv, sycl::plus<>());

    if (item.get_local_id(2) == 0) {
        qout[0] = (dtype)sumq;
        kout[0] = (dtype)sumk;
        vout[0] = (dtype)sumv;
    }
}

SYCL_EXTERNAL void mat_vec_2X_kernel(
    dtype* out1,
    dtype* out2,
    const dtype* __restrict__ input,
    const dtype* __restrict__ weighth1,
    const dtype* __restrict__ weighth2,
    int n_cols,
    int n_rows,
    int numSerialLoads, int input_stride, int weight_stride, int output_stride, int weight_row_stride, float alpha, const sycl::nd_item<3>& item) {

    int index = item.get_group(2) * item.get_local_range(1) + item.get_local_id(1);
    if (index >= n_rows) return;

    input    += item.get_group(1) *  input_stride;
    weighth1 += item.get_group(1) * weight_stride + index * weight_row_stride;
    weighth2 += item.get_group(1) * weight_stride + index * weight_row_stride;
    out1     += item.get_group(1) * output_stride;
    out2     += item.get_group(1) * output_stride;

    float sumh1 = 0;
    float sumh2 = 0;

    for (int i = 0; i < numSerialLoads; i++) {
        int j = (i * 32 + item.get_local_id(2)) * 8;
        
        if (j < n_cols) {
            dtype ip[8];
            dtype w1[8];
            dtype w2[8];

            *((sycl::uint4 *)(&ip)) = *((sycl::uint4 *)(&input[j]  ));
            *((sycl::uint4 *)(&w1)) = *((sycl::uint4 *)(&weighth1[j]));
            *((sycl::uint4 *)(&w2)) = *((sycl::uint4 *)(&weighth2[j]));
            
            // for (int el = 0; el < 8; el++) {
            //     sumh1 += float(w1[el]) * float(ip[el]);
            //     sumh2 += float(w2[el]) * float(ip[el]);
            // }
            sumh1 += w1[0] * ip[0] + w1[1] * ip[1] + w1[2] * ip[2] + w1[3] * ip[3] + w1[4] * ip[4] + w1[5] * ip[5] + w1[6] * ip[6] + w1[7] * ip[7];
            sumh2 += w2[0] * ip[0] + w2[1] * ip[1] + w2[2] * ip[2] + w2[3] * ip[3] + w2[4] * ip[4] + w2[5] * ip[5] + w2[6] * ip[6] + w2[7] * ip[7];
        }
    }

    sumh1 = sycl::reduce_over_group(item.get_sub_group(), sumh1, sycl::plus<>());
    sumh2 = sycl::reduce_over_group(item.get_sub_group(), sumh2, sycl::plus<>());

    // sum *= alpha;

    if (item.get_local_id(2) == 0) {
        out1[index] = (dtype)sumh1;
        out2[index] = (dtype)sumh2;

        // sumh1 *= 1.0f / (1.0f + sycl::native::exp(-sumh1));
        // sumh1 *= sumh2;
        // out1[index] = (dtype)sumh1;
    }
}

// // Simpler version of the above - to handle non multiple of 8 dimensions too (needed for MHA block)
// SYCL_EXTERNAL void mat_vec_kernel_simple(
//     sycl::half *output,
//     const sycl::half *__restrict__ input,
//     const sycl::half *__restrict__ weight,
//     int n,
//     int d,
//     int numSerialElements, int input_stride, int weight_stride, int output_stride, int weight_row_stride, float alpha, const sycl::nd_item<3> &item_ct1) {

//     int index = item_ct1.get_group(2) * item_ct1.get_local_range(1) + item_ct1.get_local_id(1);
//     if (index >= d) return;

//     input  += item_ct1.get_group(1) * input_stride;
//     weight += item_ct1.get_group(1) * weight_stride + index * weight_row_stride;
//     output += item_ct1.get_group(1) * MAX_SEQ_LEN;

//     dtype sum(0);

//     // for (int i = 0; i < numSerialElements; i++) {
//     //     int j = i * item_ct1.get_local_range(2) + item_ct1.get_local_id(2);
//     //     if (j < n) {
//     //         sum += weight[j] * input[j];
//     //     }
//     // }

//     // int j = item_ct1.get_local_id(2) * 4;
//     // dtype ip[4];
//     // dtype wt[4];

//     // *((sycl::uint2 *)(&ip)) = *((sycl::uint2 *)(&input[j] ));
//     // *((sycl::uint2 *)(&wt)) = *((sycl::uint2 *)(&weight[j]));

//     // sum += wt[0] * ip[0] + wt[1] * ip[1] + wt[2] * ip[2] + wt[3] * ip[3];

//     int j = item_ct1.get_local_id(2) * 8;
//     dtype ip[8];
//     dtype wt[8];

//     *((sycl::uint4 *)(&ip)) = *((sycl::uint4 *)(&input[j] ));
//     *((sycl::uint4 *)(&wt)) = *((sycl::uint4 *)(&weight[j]));

//     sum += wt[0] * ip[0] + wt[1] * ip[1] + wt[2] * ip[2] + wt[3] * ip[3] + wt[4] * ip[4] + wt[5] * ip[5] + wt[6] * ip[6] + wt[7] * ip[7];

//     sum = sycl::reduce_over_group(item_ct1.get_sub_group(), sum, sycl::plus<>());
//     sum *= alpha;
//     if (item_ct1.get_local_id(2) == 0) {
//         output[index] = sum;
//     }
// }

// Simpler version of the above - to handle non multiple of 8 dimensions too (needed for MHA block)
SYCL_EXTERNAL void mat_vec_kernel_simple(
    sycl::half *output,
    const sycl::half *__restrict__ input,
    const sycl::half *__restrict__ weight,
    int dh,
    int seq_len,
    int numSerialElements, int input_stride, int weight_stride, int output_stride, int weight_row_stride, float alpha, const sycl::nd_item<3> &item) {

    int index = item.get_group(2) * item.get_local_range(1) + item.get_local_id(1);
    if (index >= seq_len) return;

    input  += item.get_group(1) * MAX_SEQ_LEN * dh;
    weight += item.get_group(1) * MAX_SEQ_LEN * dh + index * dh;
    output += item.get_group(1) * MAX_SEQ_LEN;

    dtype sum(0);

    // for (int i = 0; i < numSerialElements; i++) {
    //     int j = i * item.get_local_range(2) + item.get_local_id(2);
    //     if (j < dh) {
    //         sum += weight[j] * input[j];
    //     }
    // }

    // int j = item.get_local_id(2) * 4;
    // dtype ip[4];
    // dtype wt[4];

    // *((sycl::uint2 *)(&ip)) = *((sycl::uint2 *)(&input[j] ));
    // *((sycl::uint2 *)(&wt)) = *((sycl::uint2 *)(&weight[j]));
    
    // sum += wt[0] * ip[0] + wt[1] * ip[1] + wt[2] * ip[2] + wt[3] * ip[3];

    int j = item.get_local_id(2) * 8;
    dtype ip[8];
    dtype wt[8];

    *((sycl::uint4 *)(&ip)) = *((sycl::uint4 *)(&input[j] ));
    *((sycl::uint4 *)(&wt)) = *((sycl::uint4 *)(&weight[j]));

    sum += wt[0] * ip[0] + wt[1] * ip[1] + wt[2] * ip[2] + wt[3] * ip[3] + wt[4] * ip[4] + wt[5] * ip[5] + wt[6] * ip[6] + wt[7] * ip[7];

    sum = sycl::reduce_over_group(item.get_sub_group(), sum, sycl::plus<>());
    sum *= alpha;
    if (item.get_local_id(2) == 0) {
        output[index] = sum;
    }
}

// // Here we make use of shared memory to achieve better memory access pattern, and transpose a 32x32 chunk of the matrix on the fly
// SYCL_EXTERNAL void vec_mat_kernel(sycl::half *output, const sycl::half *__restrict__ input,
//                const sycl::half *__restrict__ weight, int N, int K,
//                int elementsPerThread, int input_stride, int weight_stride,
//                int output_stride, int weight_row_stride,
//                const sycl::nd_item<3>& item,
//                sycl::local_accessor<sycl::half, 3> loaded_fragment) {

//     // locate beginning of my head
//     input  += item.get_group(1) * MAX_SEQ_LEN;      // h * max_seq_len
//     weight += item.get_group(1) * weight_stride;    // h * head_size for token 0
//     output += item.get_group(1) * output_stride;    // h * head_size

//     int start_n = item.get_group(2) * 32;           // item.get_group(2) -> 0, 1, 2, 3 -> which subgroup in this head
//     int i = start_n + item.get_local_id(1);         // item.get_local_id(1) -> 0, 1, 2, ..., 31

//     // 2x for double buffering
//     // +2 to avoid shared memory bank conflicts

//     // OOB check
//     if (i >= N)
//         return;

//     // load the first 32x32 fragment
//     int n = start_n + item.get_local_id(2);
//     int k = item.get_local_id(1);
//     int offset = k * weight_row_stride + n;
//     loaded_fragment[0][item.get_local_id(1)][item.get_local_id(2)] = ((n < N) && (k < K)) ? weight[offset] : (sycl::half)0;

//     float sum = 0;
//     // Loop over the matrix row and vector elements
//     for (int e = 0; e < elementsPerThread;)
//     {
//         item.barrier(sycl::access::fence_space::local_space); // wait for the load

//         int start_k = e * 32;
//         k = start_k + item.get_local_id(2);
//         int buf_i = e & 1;
//         sum += float(loaded_fragment[buf_i][item.get_local_id(2)][item.get_local_id(1)]) * (float)(input[k]);

//         // load for the next iteration
//         e++;
//         start_k = e * 32;
//         buf_i = e & 1;
//         n = start_n + item.get_local_id(2);
//         k = start_k + item.get_local_id(1);
//         int offset = k * weight_row_stride + n;
//         loaded_fragment[buf_i][item.get_local_id(1)][item.get_local_id(2)] = ((n < N) && (k < K)) ? weight[offset] : (sycl::half)0;
//     }

//     sum = sycl::reduce_over_group(item.get_sub_group(), sum, sycl::plus<>());

//     if (item.get_local_id(2) == 0)
//         output[i] = (sycl::half)sum;
// }

// Here we make use of shared memory to achieve better memory access pattern, and transpose a 32x32 chunk of the matrix on the fly
SYCL_EXTERNAL void vec_mat_kernel(
    sycl::half *output, const sycl::half *__restrict__ input, const sycl::half *__restrict__ weight,
    int dh, int seq_len, int elementsPerThread, int input_stride, int weight_stride, int output_stride, int weight_row_stride,
    const sycl::nd_item<3>& item, sycl::local_accessor<sycl::half, 3> loaded_fragment) {

    // locate beginning of my head
    input  += item.get_group(1) * MAX_SEQ_LEN;      // h * max_seq_len
    weight += item.get_group(1) * MAX_SEQ_LEN * dh; // h * max_seq_len * head_size for token 0
    output += item.get_group(1) * dh;               // h * head_size

    int start_n = item.get_group(2) * 32;           // item.get_group(2) -> 0, 1, 2, 3 -> the subgroup in this head
    int i = start_n + item.get_local_id(1);         // item.get_local_id(1) -> 0, 1, 2, ..., 31

    // 2x for double buffering
    // +2 to avoid shared memory bank conflicts

    // OOB check
    if (i >= dh)
        return;

    // load the first 32x32 fragment
    int n = start_n + item.get_local_id(2);
    int k = item.get_local_id(1);
    int offset = k * dh + n;
    loaded_fragment[0][item.get_local_id(1)][item.get_local_id(2)] = ((n < dh) && (k < seq_len)) ? weight[offset] : (sycl::half)0;

    float sum = 0;
    // Loop over the matrix row and vector elements
    for (int e = 0; e < elementsPerThread;)
    {
        item.barrier(sycl::access::fence_space::local_space); // wait for the load

        int start_k = e * 32;
        k = start_k + item.get_local_id(2);
        int buf_i = e & 1;
        sum += float(loaded_fragment[buf_i][item.get_local_id(2)][item.get_local_id(1)]) * (float)(input[k]);

        // load for the next iteration
        e++;
        start_k = e * 32;
        buf_i = e & 1;
        n = start_n + item.get_local_id(2);
        k = start_k + item.get_local_id(1);
        int offset = k * dh + n;
        loaded_fragment[buf_i][item.get_local_id(1)][item.get_local_id(2)] = ((n < dh) && (k < seq_len)) ? weight[offset] : (sycl::half)0;
    }

    sum = sycl::reduce_over_group(item.get_sub_group(), sum, sycl::plus<>());

    if (item.get_local_id(2) == 0)
        output[i] = (sycl::half)sum;
}

// // Each block processes a single head
// SYCL_EXTERNAL void RoPERotation_kernel(sycl::half* sq, sycl::half* sk, int pos, int num_heads, int num_kv_heads, int head_size, const sycl::nd_item<1> &item) {
//     int h = item.get_group(0);

//     sycl::half *q = sq + h * head_size;
//     sycl::half *k = sk + h * head_size;

//     int i = item.get_local_id(0) * 2;

//     int head_dim = i % head_size;
//     float freq = 1.0f / sycl::pow(10000.0f, head_dim / (float)head_size);
//     float val = pos * freq;
//     float fcr = sycl::cos(val);
//     float fci = sycl::sin(val);

//     // rotate q
//     float q0 = q[i];
//     float q1 = q[i + 1];
//     q[i]     = q0 * fcr - q1 * fci;
//     q[i + 1] = q0 * fci + q1 * fcr;

//     // rotate k
//     if (h < num_kv_heads) {
//         float k0 = k[i];
//         float k1 = k[i + 1];
//         k[i]     = k0 * fcr - k1 * fci;
//         k[i + 1] = k0 * fci + k1 * fcr;
//     }
// }

// Each block processes a single head
SYCL_EXTERNAL void RoPERotation_kernel(
    sycl::half* sq, sycl::half* sk,
    int pos, int num_heads, int num_kv_heads, int head_size, const sycl::nd_item<1> &item) {

    int h = item.get_group(0);

    sycl::half *q = sq + (h * MAX_SEQ_LEN) * head_size;
    sycl::half *k = sk + (h * MAX_SEQ_LEN) * head_size;

    int i = item.get_local_id(0) * 2;

    int head_dim = i % head_size;
    float freq = 1.0f / sycl::pow(10000.0f, head_dim / (float)head_size);
    float val = pos * freq;
    float fcr = sycl::cos(val);
    float fci = sycl::sin(val);

    // rotate q
    float q0 = q[i];
    float q1 = q[i + 1];
    q[i]     = q0 * fcr - q1 * fci;
    q[i + 1] = q0 * fci + q1 * fcr;

    // rotate k
    if (h < num_kv_heads) {
        float k0 = k[i];
        float k1 = k[i + 1];
        k[i]     = k0 * fcr - k1 * fci;
        k[i + 1] = k0 * fci + k1 * fcr;
    }
}

SYCL_EXTERNAL void softmax_kernel(sycl::half *__restrict__ arr, int num_heads,
                                  int size, const sycl::nd_item<1> &item_ct1,
                                  float *att, float &shared_val) {

    int h = item_ct1.get_group(0);
    int tid = item_ct1.get_local_id(0);
    int step = item_ct1.get_local_range(0);

    sycl::half *__restrict__ arr_base = arr + h * MAX_SEQ_LEN;

    // load input to shared memory
    for (int t = tid; t < size; t += step)
        att[t] = (float) arr_base[t];
    item_ct1.barrier(sycl::access::fence_space::local_space);

    // find max value (for numerical stability)
    float max_val = tid < size ? att[tid] : 0;
    for (int i = tid + step; i < size; i += step)
        if (att[i] > max_val)
            max_val = att[i];

    max_val = sycl::reduce_over_group(item_ct1.get_group(), max_val, sycl::maximum<>());
    if (item_ct1.get_local_id(0) == 0)
        shared_val = max_val;
    item_ct1.barrier(sycl::access::fence_space::local_space);
    max_val = shared_val;

    // exp and sum
    float sum = 0.0f;
    for (int i = tid; i < size; i += step) {
        att[i] = sycl::native::exp(att[i] - max_val);
        sum += att[i];
    }

    sum = sycl::reduce_over_group(item_ct1.get_group(), sum, sycl::plus<>());
    if (item_ct1.get_local_id(0) == 0)
        shared_val = sum;
    item_ct1.barrier(sycl::access::fence_space::local_space);
    sum = shared_val;

    // normalize and write the result
    float inv_sum = 1.0f / sum;
    for (int t = tid; t < size; t += step)
        arr_base[t] = (sycl::half)(att[t] * inv_sum);
}

SYCL_EXTERNAL void softmax32_kernel(float *__restrict__ x, int size,
                                    const sycl::nd_item<1> &item_ct1,
                                    float &shared_val) {

    int tid = item_ct1.get_local_id(0);
    int step = item_ct1.get_local_range(0);

    // find max value (for numerical stability)
    float max_val = tid < size ? x[tid] : 0;
    for (int i = tid + step; i < size; i += step)
        if (x[i] > max_val)
            max_val = x[i];

    max_val = sycl::reduce_over_group(item_ct1.get_group(), max_val, sycl::maximum<>());
    if (item_ct1.get_local_id(0) == 0)
        shared_val = max_val;
    item_ct1.barrier(sycl::access::fence_space::local_space);
    max_val = shared_val;

    // exp and sum
    float sum = 0.0f;
    for (int i = tid; i < size; i += step) {
        x[i] = sycl::native::exp(x[i] - max_val);
        sum += x[i];
    }

    sum = sycl::reduce_over_group(item_ct1.get_group(), sum, sycl::plus<>());
    if (item_ct1.get_local_id(0) == 0)
        shared_val = sum;
    item_ct1.barrier(sycl::access::fence_space::local_space);
    sum = shared_val;

    // normalize
    for (int i = tid; i < size; i += step)
        x[i] /= sum;
}

SYCL_EXTERNAL void argmax32_kernel(float *__restrict__ x, int size, int *result,
                                   const sycl::nd_item<1> &item_ct1,
                                   float &shared_val) {

    int tid = item_ct1.get_local_id(0);
    int step = item_ct1.get_local_range(0);

    // find local max value and its position
    float max_val = tid < size ? x[tid] : 0;
    int   max_pos = tid < size ? tid : 0;
    for (int i = tid + step; i < size; i += step) {
        if (x[i] > max_val) {
            max_val = x[i];
            max_pos = i;
        }
    }

    // find the global max value
    float global_max_val;
    global_max_val = sycl::reduce_over_group(item_ct1.get_group(), max_val, sycl::maximum<>());
    if (item_ct1.get_local_id(0) == 0)
        shared_val = global_max_val;
    item_ct1.barrier(sycl::access::fence_space::local_space);
    global_max_val = shared_val;

    // get its position
    if (max_val == global_max_val) {
        *result = max_pos;
    }
}

SYCL_EXTERNAL void silu_element_wise_mul_kernel(sycl::half *dest, sycl::half *src, int size, const sycl::nd_item<1> &item_ct1) {
    int i = item_ct1.get_group(0) * item_ct1.get_local_range(0) + item_ct1.get_local_id(0);
    if (i < size) {
        float val = (float)dest[i];
        val *= 1.0f / (1.0f + sycl::native::exp(-val));
        val *= (float)src[i];
        dest[i] = (sycl::half)val;
    }
}


// ----------------------------------------------------------------------------
// neural net blocks; the dynamics of the Transformer

int divUp(int a, int b) {
    return (a - 1) / b + 1;
}

void accum(sycl::half *a, sycl::half *b, int size) {
    int blocks = divUp(size, 1024);
    {
        get_default_queue().parallel_for<class kernel_accum>(
            sycl::nd_range(sycl::range(blocks) * sycl::range(1024), sycl::range(1024)),
            [=](sycl::nd_item<1> item_ct1) {
                element_wise_add_kernel(a, b, size, item_ct1);
            });
    }
}

void rmsnorm(sycl::half *o, sycl::half *x, sycl::half *weight, int size) {
    int elementsPerThread = divUp(size, 512);
    {
        get_default_queue().submit([&](sycl::handler &cgh) {
            sycl::local_accessor<float, 0> shared_ss_acc_ct1(cgh);

            cgh.parallel_for<class kernel_rmsnorm>(
                sycl::nd_range(sycl::range(512), sycl::range(512)),
                [=](sycl::nd_item<1> item_ct1) {
                    rmsnorm_kernel(o, x, weight, size, elementsPerThread, item_ct1, shared_ss_acc_ct1);
                });
        });
    }
}

void matmul(
    dtype* xout, dtype* x,
    dtype* w,
    int n, int d, int batch, int x_stride, int w_stride, int op_stride, int w_row_stride, float alpha) {
    int serialElements = divUp(n, 32);
    int serialLoads = divUp(serialElements, 8);     // we load 8 elements in parallel
    sycl::range block_dim(1, 4, 32);
    sycl::range grid_dim(1, batch, divUp(d, 4));
    if (w_row_stride == -1) w_row_stride = n;
    
    get_default_queue().parallel_for<class kernel_matmul>(
        sycl::nd_range(grid_dim * block_dim, block_dim),
        [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
            mat_vec_kernel(xout, x, w, n, d, serialLoads, x_stride, w_stride, op_stride, w_row_stride, alpha, item_ct1);
        }
    );
}

void matmul_f32(
    float* xout, dtype* x,
    dtype* w,
    int n, int d, int batch, int x_stride, int w_stride, int op_stride, int w_row_stride, float alpha) {
    int serialElements = divUp(n, 32);
    int serialLoads = divUp(serialElements, 8);     // we load 8 elements in parallel
    sycl::range block_dim(1, 4, 32);
    sycl::range grid_dim(1, batch, divUp(d, 4));
    if (w_row_stride == -1) w_row_stride = n;
    
    get_default_queue().parallel_for<class kernel_matmul_f32>(
        sycl::nd_range(grid_dim * block_dim, block_dim),
        [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
            mat_vec_f32_kernel(xout, x, w, n, d, serialLoads, x_stride, w_stride, op_stride, w_row_stride, alpha, item_ct1);
        }
    );
}

void matmul_mad(
    dtype* xout, dtype* x,
    dtype* w,
    int n, int d, int batch, int x_stride, int w_stride, int op_stride, int w_row_stride, float alpha) {
    int serialElements = divUp(n, 32);
    int serialLoads = divUp(serialElements, 8);     // we load 8 elements in parallel
    sycl::range block_dim(1, 4, 32);
    sycl::range grid_dim(1, batch, divUp(d, 4));
    if (w_row_stride == -1) w_row_stride = n;
    
    get_default_queue().parallel_for<class kernel_matmul_mad>(
        sycl::nd_range(grid_dim * block_dim, block_dim),
        [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
            mat_vec_mad_kernel(xout, x, w, n, d, serialLoads, x_stride, w_stride, op_stride, w_row_stride, alpha, item_ct1);
        }
    );
}

void matmul_qkv(
    dtype* qout, dtype* kout, dtype* vout, dtype* x,
    dtype* wq, dtype* wk, dtype* wv,
    int cols, int dh, int Nh) {

    int serialElements = divUp(cols, 32);
    int serialLoads = divUp(serialElements, 8);     // we load 8 elements in parallel
    sycl::range local(1, 4, 32);
    sycl::range global(1, Nh, divUp(dh, 4));

    get_default_queue().parallel_for<class kernel_matmul_qkv>(
        sycl::nd_range(global * local, local),
        [=](sycl::nd_item<3> item) [[intel::reqd_sub_group_size(32)]] {
            mat_vec_qkv_kernel(qout, kout, vout, x, wq, wk, wv, cols, dh, Nh, serialLoads, item);
        }
    );
}

void matmul_2X(
    dtype* out1, dtype* out2, dtype* input, dtype* w1, dtype* w2, int n_cols, int n_rows,
    int batch, int x_stride, int w_stride, int op_stride, int w_row_stride, float alpha) {
    int serialElements = divUp(n_cols, 32);
    int serialLoads = divUp(serialElements, 8);     // we load 8 elements in parallel
    sycl::range block_dim(1, 4, 32);
    sycl::range grid_dim(1, batch, divUp(n_rows, 4));
    if (w_row_stride == -1) w_row_stride = n_cols;

    get_default_queue().parallel_for<class kernel_matmul_w1w3>(
        sycl::nd_range(grid_dim * block_dim, block_dim),
        [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
            mat_vec_2X_kernel(out1, out2, input, w1, w2, n_cols, n_rows, serialLoads, x_stride, w_stride, op_stride, w_row_stride, alpha, item_ct1);
        }
    );
}

void RoPERotation(sycl::half *q, sycl::half *k, int pos, int num_heads, int num_kv_heads, int head_size) {
    get_default_queue().parallel_for<class kernel_rope>(
        sycl::nd_range(sycl::range(num_heads) * sycl::range(head_size / 2), sycl::range(head_size / 2)),
        [=](sycl::nd_item<1> item_ct1) {
            RoPERotation_kernel(q, k, pos, num_heads, num_kv_heads, head_size, item_ct1);
        });
}

void MultiHeadAttention(sycl::half *output,
                        sycl::half *q, sycl::half *key_cache, sycl::half *val_cache,
                        sycl::half *att,
                        int num_heads, int head_size, int seq_len) {

    sycl::queue& q_ct1 = get_default_queue();

    int dim = head_size * num_heads;

    // 1. Get attention scores
    constexpr int blockDim_x = 16;
    constexpr int blockDim_y = 8;
    constexpr int blockDim_z = 1;
              int gridDim_x  = divUp(seq_len, blockDim_y);
    constexpr int gridDim_y  = 32; // num_heads
    constexpr int gridDim_z  = 1;

    sycl::range block_dim1(blockDim_z, blockDim_y, blockDim_x);
    sycl::range grid_dim1(gridDim_z, gridDim_y, gridDim_x);
    int serialElements1 = divUp(head_size, blockDim_x);
    {
        q_ct1.submit([&](sycl::handler &cgh) {
            float sqrt_head_size = 1.0 / sqrt(head_size);

            cgh.parallel_for<class kernel_mat_vec_simple>(
                sycl::nd_range(grid_dim1 * block_dim1, block_dim1),
                [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(blockDim_x)]] {
                    mat_vec_kernel_simple(
                        att, q, key_cache,
                        head_size, seq_len, serialElements1, head_size, head_size, seq_len, dim, sqrt_head_size,
                        item_ct1);
                });
        });
    }

    // 2. Run softmax kernel
    {
        q_ct1.submit([&](sycl::handler &cgh) {
            sycl::local_accessor<float, 1> att_acc_ct1(sycl::range(MAX_SEQ_LEN), cgh);
            sycl::local_accessor<float, 0> shared_val_acc_ct1(cgh);

            cgh.parallel_for<class kernel_softmax>(
                sycl::nd_range(sycl::range(num_heads) * sycl::range(1024), sycl::range(1024)),
                [=](sycl::nd_item<1> item_ct1) {
                    softmax_kernel(att, num_heads, seq_len, item_ct1, att_acc_ct1.get_pointer(), shared_val_acc_ct1);
                });
        });
    }

    // 3. weighted sum of the values to get the final result
    // constexpr int blockDim_x = 32;
    // constexpr int blockDim_y = 32;
    // constexpr int blockDim_z = 1;
    //           int gridDim_x  = divUp(head_size, blockDim_y);
    // constexpr int gridDim_y  = 32; // num_heads
    // constexpr int gridDim_z  = 1;
    int serialElements2 = divUp(seq_len, 32);
    sycl::range block_dim(1, 32, 32);
    sycl::range grid_dim2(1, num_heads, divUp(head_size, 32));
    {
        q_ct1.submit([&](sycl::handler &cgh) {
            sycl::local_accessor<sycl::half, 3> loaded_fragment_acc_ct1(sycl::range(2, 32, 32 + 2), cgh);

            cgh.parallel_for<class kernel_vec_mat>(
                sycl::nd_range(grid_dim2 * block_dim, block_dim),
                [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
                    vec_mat_kernel(
                        output, att, val_cache,
                        head_size, seq_len, serialElements2, seq_len, head_size, head_size, dim,
                        item_ct1, loaded_fragment_acc_ct1);
                });
        });
    }
}

void siluElementwiseMul(sycl::half *hb, sycl::half *hb2, int size) {
    get_default_queue().parallel_for<class kernel_silu_eltwise_mul>(
        sycl::nd_range(sycl::range(divUp(size, 1024)) * sycl::range(1024), sycl::range(1024)),
        [=](sycl::nd_item<1> item_ct1) {
            silu_element_wise_mul_kernel(hb, hb2, size, item_ct1);
        }
    );
}

int sample_argmax(float *probabilities, int n) {
    sycl::queue& q_ct1 = get_default_queue();

    // return the index that has the highest probability
    int max_pos;
    int *pmax_pos;

    // allocate memory on the device
    pmax_pos = sycl::malloc_device<int>(1, q_ct1);

    // call the kernel
    q_ct1.submit([&](sycl::handler &cgh) {
        sycl::local_accessor<float, 0> shared_val_acc_ct1(cgh);

        cgh.parallel_for<class kernel_argmax32>(
            sycl::nd_range(sycl::range(1024), sycl::range(1024)),
            [=](sycl::nd_item<1> item_ct1) {
                argmax32_kernel(probabilities, n, pmax_pos, item_ct1, shared_val_acc_ct1);
        });
    });

    // copy the result back to host
    q_ct1.memcpy(&max_pos, pmax_pos, sizeof(int)).wait();

    // free the allocated memory
    sycl::free(pmax_pos, q_ct1);

    return max_pos;
}
#pragma once


template <typename T>
__global__ void
recoverSequence1D(int seq_len,
                  const T* __restrict__ inp,
                  const int64_t* __restrict__ length,
                  T* __restrict__ out) {
    inp += length[gridDim.x + blockIdx.x];
    out += blockIdx.x * seq_len;
    for (int x = threadIdx.x; x < length[blockIdx.x]; x += blockDim.x) {
        out[x] = __ldg(&inp[x]);
    }
}


template <typename T>
void
launchRecoverSequence1D(int batch_size,
                        int seq_len,
                        const T* inp,
                        const int64_t* length,
                        T* out,
                        cudaStream_t stream = 0) {
    const int dimBlock = min(seq_len, 1024);
    recoverSequence1D<<<batch_size, dimBlock, 0, stream>>>(seq_len, inp, length, out);
}


template <typename T>
__global__ void
recoverSequence2D(int seq_len,
                  int hidden_size,
                  const T* __restrict__ inp,
                  const int64_t* __restrict__ length,
                  T* __restrict__ out) {
    inp += (length[gridDim.x + blockIdx.x] + threadIdx.y) * hidden_size;
    out += (blockIdx.x * seq_len + threadIdx.y) * hidden_size;
    for (int y = threadIdx.y; y < length[blockIdx.x]; y += blockDim.y) {
        for (int x = threadIdx.x; x < hidden_size; x += blockDim.x) {
            out[x] = __ldg(&inp[x]);
        }
        inp += blockDim.y * hidden_size;
        out += blockDim.y * hidden_size;
    }
}


template <typename T>
void
launchRecoverSequence2D(int batch_size,
                        int seq_len,
                        int hidden_size,
                        const T* inp,
                        const int64_t* length,
                        T* out,
                        cudaStream_t stream = 0) {
    const int dimX = min(1024, (hidden_size + 31) & 0xFFFFFFE0);
    const int dimY = min(seq_len, 1024 / dimX);
    const dim3 dimBlock(dimX, dimY);
    recoverSequence2D<<<batch_size, dimBlock, 0, stream>>>(seq_len, hidden_size, inp, length, out);
}


template <typename T>
__global__ void
recoverSequenceBackward1D(int seq_len,
                          const T* __restrict__ grad,
                          const int64_t* __restrict__ length,
                          T* __restrict__ dInp) {
    grad += blockIdx.x * seq_len;
    dInp += length[gridDim.x + blockIdx.x];
    for (int x = threadIdx.x; x < length[blockIdx.x]; x += blockDim.x) {
        dInp[x] == __ldg(&grad[x]);
    }
}


template <typename T>
void
launchRecoverSequenceBackward1D(int batch_size,
                                int seq_len,
                                const T* grad,
                                const int64_t* length,
                                T* dInp,
                                cudaStream_t stream = 0) {
    const int dimBlock = min(seq_len, 1024);
    recoverSequenceBackward1D<<<batch_size, dimBlock, 0, stream>>>(seq_len, grad, length, dInp);
}


template <typename T>
__global__ void
recoverSequenceBackward2D(int seq_len,
                          int hidden_size,
                          const T* __restrict__ grad,
                          const int64_t* __restrict__ length,
                          T* __restrict__ dInp) {
    grad += (blockIdx.x * seq_len + threadIdx.y) * hidden_size;
    dInp += (length[gridDim.x + blockIdx.x] + threadIdx.y) * hidden_size;
    for (int y = threadIdx.y; y < length[blockIdx.x]; y += blockDim.y) {
        for (int x = threadIdx.x; x < hidden_size; x += blockDim.x) {
            dInp[x] = __ldg(&grad[x]);
        }
        grad += blockDim.y * hidden_size;
        dInp += blockDim.y * hidden_size;
    }
}


template <typename T>
void
launchRecoverSequenceBackward2D(int batch_size,
                                int seq_len,
                                int hidden_size,
                                const T* grad,
                                const int64_t* length,
                                T* dInp,
                                cudaStream_t stream = 0) {
    const int dimX = min(1024, (hidden_size + 31) & 0xFFFFFFE0);
    const int dimY = min(seq_len, 1024 / dimX);
    const dim3 dimBlock(dimX, dimY);
    recoverSequenceBackward2D<<<batch_size, dimBlock, 0, stream>>>(seq_len, hidden_size, grad, length, dInp);
}


class RecoverSequenceTrain {
public:
    template <typename T>
    static void
    forward1D(const T* inp,
              const int64_t* length,
              T* out,
              int batch_size,
              int seq_len,
              cudaStream_t stream) {
        launchRecoverSequence1D(
                batch_size,
                seq_len,
                inp,
                length,
                out,
                stream);
    }

    template <typename T>
    static void
    forward2D(const T* inp,
              const int64_t* length,
              T* out,
              int batch_size,
              int seq_len,
              int hidden_size,
              cudaStream_t stream) {
        launchRecoverSequence2D(
                batch_size,
                seq_len,
                hidden_size,
                inp,
                length,
                out,
                stream);
    }

    template <typename T>
    static void
    backward1D(const T* grad,
               const int64_t* length,
               T* dInp,
               int batch_size,
               int seq_len,
               cudaStream_t stream) {
        launchRecoverSequenceBackward1D(
                batch_size,
                seq_len,
                grad,
                length,
                dInp,
                stream);
    }

    template <typename T>
    static void
    backward2D(const T* grad,
               const int64_t* length,
               T* dInp,
               int batch_size,
               int seq_len,
               int hidden_size,
               cudaStream_t stream) {
        launchRecoverSequenceBackward2D(
                batch_size,
                seq_len,
                hidden_size,
                grad,
                length,
                dInp,
                stream);
    }
};


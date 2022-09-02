#pragma once
#include "Add.h"
#include "Context.h"


// drop_path(inp) + res
template <typename T>
void
launchDropPathResIF(int batch_size,
                    int seq_len,
                    int hidden_size,
                    float prob,
                    float scale,
                    uint64_t seed,
                    uint64_t& offset,
                    const T* residual,
                    T* io,
                    uint8_t* mask,
                    cudaStream_t stream = 0);


template <typename T>
void
launchDropPathOB(int batch_size,
                 int seq_len,
                 int hidden_size,
                 float scale,
                 const T* grad,
                 const uint8_t* mask,
                 T* dInp,
                 cudaStream_t stream = 0);


template <typename T>
class DropPathRes {
public:
    static void
    forward(T* io,
            const T* residual,
            uint8_t* mask,
            float prob,
            float scale,
            int batch_size,
            int seq_len,
            int hidden_size,
            cudaStream_t stream = 0) {
        if (hidden_size & 0x7) {
            errMsg(format("Unsupported size (%d) for DropPathRes forward", hidden_size));
        }
        launchDropPathResIF(batch_size,
                            seq_len,
                            hidden_size,
                            prob,
                            scale,
                            Context::instance().getSeed(),
                            Context::instance().getOffset(),
                            residual,
                            io,
                            mask,
                            stream);
    }

    static void
    backward(const T* grad,
             const uint8_t* mask,
             T* dInp,
             float scale,
             int batch_size,
             int seq_len,
             int hidden_size,
             cudaStream_t stream = 0) {
        if (hidden_size & 0x7) {
            errMsg(format("Unsupported size (%d) for DropPathRes backward", hidden_size));
        }
        launchDropPathOB(batch_size,
                         seq_len,
                         hidden_size,
                         scale,
                         grad,
                         mask,
                         dInp,
                         stream);
    }
};


// drop_path(inp + bias) + res
template <typename T>
void
launchDropPathBiasResOF(int batch_size,
                        int seq_len,
                        int hidden_size,
                        float prob,
                        float scale,
                        uint64_t seed,
                        uint64_t& offset,
                        const T* inp,
                        const T* bias,
                        const T* residual,
                        T* out,
                        uint8_t* mask,
                        cudaStream_t stream = 0);


template <typename T>
class DropPathBiasRes {
public:
    static void
    forward(const T* inp,
            const T* bias,
            const T* residual,
            T* out,
            uint8_t* mask,
            float prob,
            float scale,
            int batch_size,
            int seq_len,
            int hidden_size,
            cudaStream_t stream = 0) {
        if (hidden_size & 0x7) {
            errMsg(format("Unsupported size (%d) for DropPathBiasRes forward", hidden_size));
        }
        launchDropPathBiasResOF(batch_size,
                                seq_len,
                                hidden_size,
                                prob,
                                scale,
                                Context::instance().getSeed(),
                                Context::instance().getOffset(),
                                inp,
                                bias,
                                residual,
                                out,
                                mask,
                                stream);
    }

    static void
    backward(const T* grad,
             const uint8_t* mask,
             T* dInp,
             T* dBias,
             float scale,
             int batch_size,
             int seq_len,
             int hidden_size,
             cudaStream_t stream = 0) {
        if (hidden_size & 0x7) {
            errMsg(format("Unsupported size (%d) for DropPathBiasRes backward", hidden_size));
        }
        launchDropPathOB(batch_size,
                         seq_len,
                         hidden_size,
                         scale,
                         grad,
                         mask,
                         dInp,
                         stream);

        launchAddBiasOB(batch_size * seq_len,
                        hidden_size,
                        dInp,
                        dBias,
                        stream);
    }
};


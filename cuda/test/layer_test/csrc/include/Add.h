#pragma once
#include "errMsg.h"


template <typename T>
void
launchAddBiasIF(int nRow,
                int nCol,
                const T* bias,
                T* io,
                cudaStream_t stream = 0);


template <typename T>
void
launchAddBiasOB(int nRow,
                int nCol,
                const T* grad,
                T* dBias,
                cudaStream_t stream = 0);


template <typename T>
class AddBias {
public:
    static void
    forward(const T* bias,
            T* io,
            int nRow,
            int nCol,
            cudaStream_t stream = 0) {
        if (nCol & 0x7) {
            errMsg(format("Unsupported size (%d) for AddBias forward", nCol));
        }
        launchAddBiasIF(nRow,
                        nCol,
                        bias,
                        io,
                        stream);
    }

    static void
    backward(const T* dInp,
             T* dBias,
             int nRow,
             int nCol,
             cudaStream_t stream = 0) {
        launchAddBiasOB(nRow,
                        nCol,
                        dInp,
                        dBias,
                        stream);
    }
};


template <typename T>
void
launchAddBiasResOF(int nRow,
                   int nCol,
                   const T* inp,
                   const T* bias,
                   const T* residual,
                   T* out,
                   cudaStream_t stream = 0);


template <typename T>
void
launchAddBiasResOB(int nRow,
                   int nCol,
                   const T* grad,
                   T* dInp,
                   T* dBias,
                   cudaStream_t stream = 0);


template <typename T>
class AddBiasRes {
public:
    static void
    forward(const T* inp,
            const T* bias,
            const T* residual,
            T* out,
            int nRow,
            int nCol,
            cudaStream_t stream = 0) {
        if (nCol & 0x7) {
            errMsg(format("Unsupported size (%d) for AddBiasRes forward", nCol));
        }
        launchAddBiasResOF(nRow,
                           nCol,
                           inp,
                           bias,
                           residual,
                           out,
                           stream);
    }

    static void
    backward(const T* dResidual,
             T* dInp,
             T* dBias,
             int nRow,
             int nCol,
             cudaStream_t stream = 0) {
        launchAddBiasResOB(nRow,
                           nCol,
                           dResidual,
                           dInp,
                           dBias,
                           stream);
    }
};


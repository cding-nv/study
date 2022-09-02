#pragma once
#include "Add.h"
#include "errMsg.h"


template <typename T>
void
launchGeluBiasIF(int nRow,
                 int nCol,
                 const T* bias,
                 T* io,
                 cudaStream_t stream = 0);


template <typename T>
void
launchGeluBiasOF(int nRow,
                 int nCol,
                 const T* inp,
                 const T* bias,
                 T* out,
                 cudaStream_t stream = 0);


template <typename T>
void
launchGeluBiasIB(int nRow,
                 int nCol,
                 T* grad,
                 const T* inp,
                 const T* bias,
                 cudaStream_t stream = 0);


template <typename T>
class GeluBias {
public:
    static void
    forward(const T* bias,
            T* io,
            int nRow,
            int nCol,
            cudaStream_t stream = 0) {
        if (nCol & 7) {
            errMsg(format("Unsupported size (%d) for GeluBias forward", nCol));
        }
        launchGeluBiasIF(nRow,
                         nCol,
                         bias,
                         io,
                         stream);
    }

    static void
    forward(const T* inp,
            const T* bias,
            T* out,
            int nRow,
            int nCol,
            cudaStream_t stream = 0) {
        if (nCol & 7) {
            errMsg(format("Unsupported size (%d) for GeluBias forward", nCol));
        }
        launchGeluBiasOF(nRow,
                         nCol,
                         inp,
                         bias,
                         out,
                         stream);
    }

    static void
    backward(T* grad,
             const T* inp,
             const T* bias,
             T* dBias,
             int nRow,
             int nCol,
             cudaStream_t stream = 0) {
        launchGeluBiasIB(nRow,
                         nCol,
                         grad,
                         inp,
                         bias,
                         stream);

        launchAddBiasOB(nRow,
                        nCol,
                        grad,
                        dBias,
                        stream);
    }
};


#pragma once
#include "Add.h"
#include "Context.h"


template <typename T>
void
launchDropoutIF(float prob,
                float scale,
                uint64_t seed,
                uint64_t& offset,
                int N,
                T* io,
                uint8_t* mask,
                cudaStream_t stream = 0);


template <typename T>
void
launchDropoutOF(float prob,
                float scale,
                uint64_t seed,
                uint64_t& offset,
                int N,
                const T* inp,
                T* out,
                uint8_t* mask,
                cudaStream_t stream = 0);


template <typename T>
void
launchDropoutIB(float scale,
                int N,
                T* grad,
                const uint8_t* mask,
                cudaStream_t stream = 0);


template <typename T>
void
launchDropoutOB(float scale,
                int N,
                const T* grad,
                const uint8_t* mask,
                T* dInp,
                cudaStream_t stream = 0);


template <typename T>
class Dropout {
public:
    static void
    forward(T* io,
            uint8_t* mask,
            float prob,
            float scale,
            int N,
            cudaStream_t stream = 0) {
        if (N & 0x7) {
            errMsg(format("Unsupported size (%d) for Dropout forward", N));
        }
        launchDropoutIF(prob,
                        scale,
                        Context::instance().getSeed(),
                        Context::instance().getOffset(),
                        N,
                        io,
                        mask,
                        stream);
    }

    static void
    forward(const T* inp,
            T* out,
            uint8_t* mask,
            float prob,
            float scale,
            int N,
            cudaStream_t stream = 0) {
        if (N & 0x7) {
            errMsg(format("Unsupported size (%d) for Dropout forward", N));
        }
        launchDropoutOF(prob,
                        scale,
                        Context::instance().getSeed(),
                        Context::instance().getOffset(),
                        N,
                        inp,
                        out,
                        mask,
                        stream);
    }

    static void
    backward(T* grad,
             const uint8_t* mask,
             float scale,
             int N,
             cudaStream_t stream = 0) {
        if (N & 0x7) {
            errMsg(format("Unsupported size (%d) for Dropout backward", N));
        }
        launchDropoutIB(scale,
                        N,
                        grad,
                        mask,
                        stream);
    }

    static void
    backward(const T* grad,
             const uint8_t* mask,
             T* dInp,
             float scale,
             int N,
             cudaStream_t stream = 0) {
        if (N & 0x7) {
            errMsg(format("Unsupported size (%d) for Dropout backward", N));
        }
        launchDropoutOB(scale,
                        N,
                        grad,
                        mask,
                        dInp,
                        stream);
    }
};


// dropout(inp + bias)
template <typename T>
void
launchDropoutBiasIF(float prob,
                    float scale,
                    uint64_t seed,
                    uint64_t& offset,
                    int nRow,
                    int nCol,
                    const T* bias,
                    T* io,
                    uint8_t* mask,
                    cudaStream_t stream = 0);


// dropout(inp + bias)
template <typename T>
void
launchDropoutBiasOF(float prob,
                    float scale,
                    uint64_t seed,
                    uint64_t& offset,
                    int nRow,
                    int nCol,
                    const T* inp,
                    const T* bias,
                    T* out,
                    uint8_t* mask,
                    cudaStream_t stream = 0);


template <typename T>
class DropoutBias {
public:
    static void
    forward(const T* bias,
            T* io,
            uint8_t* mask,
            float prob,
            float scale,
            int nRow,
            int nCol,
            cudaStream_t stream = 0) {
        if (nCol & 0x7) {
            errMsg(format("Unsupported size (%d) for DropoutBias forward", nCol));
        }
        launchDropoutBiasIF(prob,
                            scale,
                            Context::instance().getSeed(),
                            Context::instance().getOffset(),
                            nRow,
                            nCol,
                            bias,
                            io,
                            mask,
                            stream);
    }

    static void
    forward(const T* inp,
            const T* bias,
            T* out,
            uint8_t* mask,
            float prob,
            float scale,
            int nRow,
            int nCol,
            cudaStream_t stream = 0) {
        if (nCol & 0x7) {
            errMsg(format("Unsupported size (%d) for DropoutBias forward", nCol));
        }
        launchDropoutBiasOF(prob,
                            scale,
                            Context::instance().getSeed(),
                            Context::instance().getOffset(),
                            nRow,
                            nCol,
                            inp,
                            bias,
                            out,
                            mask,
                            stream);
    }

    static void
    backward(T* grad,
             const uint8_t* mask,
             T* dBias,
             float scale,
             int nRow,
             int nCol,
             cudaStream_t stream = 0) {
        int N = nRow * nCol;
        if (N & 0x7) {
            errMsg(format("Unsupported size (%d) for DropoutBias backward", N));
        }
        launchDropoutIB(scale,
                        N,
                        grad,
                        mask,
                        stream);

        launchAddBiasOB(nRow,
                        nCol,
                        grad,
                        dBias,
                        stream);
    }

    static void
    backward(const T* grad,
             const uint8_t* mask,
             T* dInp,
             T* dBias,
             float scale,
             int nRow,
             int nCol,
             cudaStream_t stream = 0) {
        int N = nRow * nCol;
        if (N & 0x7) {
            errMsg(format("Unsupported size (%d) for DropoutBias backward", N));
        }
        launchDropoutOB(scale,
                        N,
                        grad,
                        mask,
                        dInp,
                        stream);

        launchAddBiasOB(nRow,
                        nCol,
                        dInp,
                        dBias,
                        stream);
    }
};


// dropout(inp + bias) + res
template <typename T>
void
launchDropoutBiasResOF(float prob,
                       float scale,
                       uint64_t seed,
                       uint64_t& offset,
                       int nRow,
                       int nCol,
                       const T* inp,
                       const T* bias,
                       const T* residual,
                       T* out,
                       uint8_t* mask,
                       cudaStream_t stream = 0);


template <typename T>
class DropoutBiasRes {
public:
    static void
    forward(const T* inp,
            const T* bias,
            const T* residual,
            T* out,
            uint8_t* mask,
            float prob,
            float scale,
            int nRow,
            int nCol,
            cudaStream_t stream = 0) {
        if (nCol & 0x7) {
            errMsg(format("Unsupported size (%d) for DropoutBiasRes forward", nCol));
        }
        launchDropoutBiasResOF(prob,
                               scale,
                               Context::instance().getSeed(),
                               Context::instance().getOffset(),
                               nRow,
                               nCol,
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
             int nRow,
             int nCol,
             cudaStream_t stream = 0) {
        int N = nRow * nCol;
        if (N & 0x7) {
            errMsg(format("Unsupported size (%d) for DropoutBiasRes backward", N));
        }
        launchDropoutOB(scale,
                        N,
                        grad,
                        mask,
                        dInp,
                        stream);

        launchAddBiasOB(nRow,
                        nCol,
                        dInp,
                        dBias,
                        stream);
    }
};


#pragma once
#include "checkErr.h"
#include <cstdint>
#include <unordered_map>
#include <vector>


class Context {
private:
    uint64_t _seed = 911ULL;
    uint64_t _offset = 0ULL;
    std::vector<cudaStream_t> _streams;
    std::unordered_map<cudaStream_t, cublasHandle_t> _handles;

private:
    Context() {}

    ~Context() {
        for (auto& item : _streams) {
            checkErr(cudaStreamDestroy(item));
        }
        _streams.clear();

        for (auto& item : _handles) {
            checkErr(cublasDestroy(item.second));
        }
        _handles.clear();
    }

public:
    static Context&
    instance() {
        static Context _ctx;
        return _ctx;
    }

    cudaStream_t
    getStream(int idx) {
        if (getStreamsCnt() <= idx) {
            setStreams(idx + 1);
        }
        return _streams[idx];
    }

    int
    getStreamsCnt() {
        return _streams.size();
    }

    cublasHandle_t
    getHandle(cudaStream_t stream, bool fast_fp32 = false) {
        setHandle(stream, fast_fp32);
        return _handles[stream];
    }

    uint64_t
    getSeed() {
        return _seed;
    }

    uint64_t&
    getOffset() {
        return _offset;
    }

    void
    setStreams(int cnt) {
        int oldCnt = getStreamsCnt();
        _streams.resize(cnt);
        for (int i = oldCnt; i < cnt; ++i) {
            printf(">>> create a cuda stream\n");
            checkErr(cudaStreamCreate(&_streams[i]));
        }
    }

    void
    setHandle(cudaStream_t stream, bool fast_fp32) {
        if (_handles.find(stream) == _handles.end()) {
            printf(">>> create a cublas handle\n");
            cublasHandle_t handle;
            checkErr(cublasCreate(&handle));
            checkErr(cublasSetStream(handle, stream));
            if (fast_fp32) {
                checkErr(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));
            }
            _handles[stream] = handle;
        }
    }

    void
    setSeed(uint64_t seed) {
        printf(">>> set seed: %lu\n", seed);
        _seed = seed;
    }
};


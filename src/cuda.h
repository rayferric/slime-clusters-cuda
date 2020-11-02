#ifndef CUDA_H
#define CUDA_H

#include <cstdint>

// Silences code analysis:
#ifdef __NVCC__
    #define HYBRID_CALL __host__ __device__
    #define DEVICE_CALL __device__
#else
    #define HYBRID_CALL
    #define DEVICE_CALL
#endif

#define BEGIN_CRITICAL(lock) \
    bool _locked = true; \
    while(_locked) { \
        if(atomicExch(lock, 1) == 0) {

#define END_CRITICAL(lock) \
            _locked = false; \
            atomicExch(lock, 0); \
        } \
    }

namespace CUDA {
    // Standarizes signed integer overflow behavior:
    HYBRID_CALL int32_t wrapping_mul(int32_t a, int32_t b) {
        return (int64_t)a * b;
    }
}

#endif // CUDA_H
#ifndef CUDA_H
#define CUDA_H

#include <cstdint>

#ifdef __NVCC__
    #define CUDA_CALL __host__ __device__
#else
    #define CUDA_CALL
#endif 

namespace CUDA {
    // Standarizes signed integer overflow behavior:
    CUDA_CALL int32_t wrapping_mul(int32_t a, int32_t b) {
        return (int64_t)a * b;
    }
}

#endif // CUDA_H
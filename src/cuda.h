#ifndef CUDA_H
#define CUDA_H

#ifdef __NVCC__
    #define CUDA __host__ __device__
#else
    #define CUDA
#endif 

#endif // CUDA_H
#include <cuda.h>

#include "../cuda_assert.hpp"

//
// never be the hot spot, no optimization needed
//
__global__ void kernel_add_bias(float* __restrict__ y, float* __restrict__ b, int bs, int m)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < bs * m)
    {
        int j = i % m;
        y[i] = b[j]; // <-- this kernel is called before GEMM, so copy instead of addition
    }
}

//
// never be the hot spot, no optimization needed
//
__global__ void kernel_add_bias_on_1st_dimension(float* __restrict__ y, float* __restrict__ b, int bs, int m)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < bs * m)
    {
        int j = i / m;
        y[i] = b[j]; // <-- this kernel is called before GEMM, so copy instead of addition
    }
}

extern "C"
void add_bias( float*y, float* b, int bs, int m, int on_1st_dimension, cudaStream_t sm )
{
    int const threads_per_block = 256;
    int const num_blocks = (bs * m + threads_per_block - 1) / threads_per_block;
    int const shared_memory = 0;
    if ( 0 == on_1st_dimension )
        kernel_add_bias<<<num_blocks, threads_per_block, shared_memory, sm>>>( y, b, bs, m );
    else
        kernel_add_bias_on_1st_dimension<<<num_blocks, threads_per_block, shared_memory, sm>>>( y, b, bs, m );
}



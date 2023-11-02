#include <cuda.h>

#include "../cuda_assert.hpp"

//
// never be the hotspot, no optimization
//
__global__ void cuda_kernel_scaled_offset( float* x, float* y, int dim_a, int dim_b, float* __restrict__ alpha, float* __restrict__ beta )
{
    int const i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < dim_a * dim_b)
    {
        int const offset = i % dim_b;
        y[i] = x[i] * alpha[offset] + beta[offset];
    }
}

// example: cuda_scaled_offset( x, 10, 768, alpha, beta, sm ), in which x[10, 768], alpha[768,], beta[768,]
extern "C"
void cuda_scaled_offset( float* x, float* y, int dim_a, int dim_b, float* alpha, float* beta, cudaStream_t sm )
{
    int const threads_per_block = 256;
    int const num_blocks = (dim_a*dim_b + threads_per_block - 1) / threads_per_block;
    int const shared_memory = 0;
    cuda_kernel_scaled_offset<<<num_blocks, threads_per_block, shared_memory, sm>>>( x, y, dim_a, dim_b, alpha, beta );
}



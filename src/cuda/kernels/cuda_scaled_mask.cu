#include <cuda.h>

#include "../cuda_assert.hpp"

//
// never be the hotspot, no optimization
//
__global__ void cuda_kernel_scaled_mask( float* x, float* y, int dim_a, int dim_b, float scale )
{
    int const i = blockIdx.x * blockDim.x + threadIdx.x;
    int const r = static_cast<int>( i/dim_a );
    int const c = i % dim_b;
    if (i < dim_a * dim_b)
    {
        y[i] = (r >= c) * x[i] * scale - (r < c) * 1.0e10f;
    }
}

extern "C"
void cuda_apply_scaled_mask( float* x, int dim_a, int dim_b, float scale, cudaStream_t sm )
{
    int const threads_per_block = 256;
    int const num_blocks = (dim_a*dim_b + threads_per_block - 1) / threads_per_block;
    int const shared_memory = 0;
    cuda_kernel_scaled_mask<<<num_blocks, threads_per_block, shared_memory, sm>>>( x, x, dim_a, dim_b, scale );
}

extern "C"
void cuda_apply_scaled_mask_2( float* x, float* y, int dim_a, int dim_b, float scale, cudaStream_t sm )
{
    int const threads_per_block = 256;
    int const num_blocks = (dim_a*dim_b + threads_per_block - 1) / threads_per_block;
    int const shared_memory = 0;
    cuda_kernel_scaled_mask<<<num_blocks, threads_per_block, shared_memory, sm>>>( x, y, dim_a, dim_b, scale );
}


#include <cuda.h>

#include "../cuda_assert.hpp"

__device__ float intrinsic_tanh( float x )
{
    float const e2x = __expf( x+x );
    return __fdiv_rd( __fadd_rd(e2x, -1.0f), __fadd_rd(e2x, 1.0f) );
}

//
// never be the hot spot, no optimization needed
//
__global__ void kernel_gelu(float* input, float* output, int n )
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n )
    {
        float const scale = 0.79788456080286535588f;
        float const x = input[i];
        //float const cdf = 1.0f + intrinsic_tanh(scale*(x+0.044715*x*x*x));
        float const cdf = 1.0f + tanhf(scale*(x+0.044715*x*x*x));
        output[i] = x * cdf * 0.5f;
    }
}

extern "C"
void cuda_gelu( float* input, float* output, int n, cudaStream_t sm )
{
    int const threads_per_block = 256;
    int const num_blocks = (n + threads_per_block - 1) / threads_per_block;
    int const shared_memory = 0;
    kernel_gelu<<<num_blocks, threads_per_block, shared_memory, sm>>>( input, output, n );
}


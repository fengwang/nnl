#include <cuda.h>

#include "../cuda_assert.hpp"

__global__ void kernel_vocabulary_copy( float* wte, int n_vocab, int n_embd, int* max_prob_index, float* output )
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float* input = wte + max_prob_index[0] * n_embd;

    if ( i < n_embd )
    {
        output[i] = input[i];
    }
}

//
// wte[argmax(gpt2(input))[-1]]
//
extern "C"
void vocabulary_lookup( float* wte, int n_vocab, int n_embd, int* max_prob_index, float* output, cudaStream_t sm )
{
    int const threads_per_block = 256;
    int const num_blocks = (n_embd + threads_per_block - 1) / threads_per_block;
    int const shared_memory = 0;
    kernel_vocabulary_copy<<<num_blocks, threads_per_block, shared_memory, sm>>>( wte, n_vocab, n_embd, max_prob_index, output );
}


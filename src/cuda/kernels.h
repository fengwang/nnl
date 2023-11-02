#ifndef TNVVJCRRHWKCTRCBQCSKSGEALGOSEJNGQQJBBBYATCAIFCCFEQSUBSCMVFLIBPRGDHBKLYXLW
#define TNVVJCRRHWKCTRCBQCSKSGEALGOSEJNGQQJBBBYATCAIFCCFEQSUBSCMVFLIBPRGDHBKLYXLW

#include <cuda.h>
#include <cuda_runtime.h>

extern "C"
{
    void add_bias( float*y, float* b, int bs, int m, int on_1st_dimension, cudaStream_t sm );

    void cuda_gelu( float* input, float* output, int n, cudaStream_t sm );

    void cuda_layer_norm( float* input, float* output, int rows, int cols, float eps, float* mean, float* var, cudaStream_t sm );

    void cuda_scaled_offset( float* x, float* y, int dim_a, int dim_b, float* alpha, float* beta, cudaStream_t sm );

    void cuda_apply_scaled_mask( float* x, int dim_a, int dim_b, float scale, cudaStream_t sm );
    void cuda_apply_scaled_mask_2( float* x, float*y, int dim_a, int dim_b, float scale, cudaStream_t sm );

    void cuda_add( float* c, float* a, int a_row, int a_col, float* b, int b_row, int b_col, cudaStream_t sm );

    void cuda_argmax_1d( float* input, int n_input, float* cache, int n_cache, int* result, cudaStream_t sm );

    void vocabulary_lookup( float* wte, int n_vocab, int n_embd, int* max_prob_index, float* output, cudaStream_t sm );
}

#endif//TNVVJCRRHWKCTRCBQCSKSGEALGOSEJNGQQJBBBYATCAIFCCFEQSUBSCMVFLIBPRGDHBKLYXLW


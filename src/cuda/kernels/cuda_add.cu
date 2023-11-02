#include <cuda.h>

#include "../cuda_assert.hpp"

__global__ void kernel_add_r_c_r_c( float* c, int c_row, int c_col, float* a, float* b )
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < c_row * c_col )
    {
        c[i] = a[i] + b[i];
    }
}

__global__ void kernel_add_r_c_r_1( float* c, int c_row, int c_col, float* a, float* b )
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < c_row * c_col )
    {
        c[i] = a[i] + b[i%c_row];
    }
}

__global__ void kernel_add_r_c_1_c( float* c, int c_row, int c_col, float* a, float* b )
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < c_row * c_col )
    {
        c[i] = a[i] + b[i%c_col];
    }
}

__global__ void kernel_add_r_1_1_c( float* c, int c_row, int c_col, float* a, float* b )
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < c_row * c_col )
    {
        c[i] = a[i%c_row] + b[i%c_col];
    }
}

extern "C"
void cuda_add( float* c, float* a, int a_row, int a_col, float* b, int b_row, int b_col, cudaStream_t sm )
{
    if ( a_row < b_row ) // a_row is 1
    {
        return cuda_add( c, b, b_row, b_col, a, a_row, a_col, sm );
    }

    if ( (a_row == b_row) && (a_col < b_col) ) // a_col is 1
    {
        return cuda_add( c, b, b_row, b_col, a, a_row, a_col, sm );
    }

    int const c_row = (a_row >= b_row) ? a_row : b_row;
    int const c_col = (a_col >= b_col) ? a_col : b_col;
    int const threads_per_block = 256;
    int const num_blocks = (c_row * c_col + threads_per_block - 1) / threads_per_block;
    int const shared_memory = 0;

    // [ r, c] -- [r, c]
    if ( (a_row == c_row) && (a_col == c_col) && (a_row == b_row) && (a_col == b_col) )
    {
        kernel_add_r_c_r_c<<<num_blocks, threads_per_block, shared_memory, sm>>>( c, c_row, c_col, a, b );
        return;
    }

    // [ r, c] -- [r, 1]
    if ( (a_row == c_row) && (a_col == c_col) && (a_row == b_row) && (b_col == 1) )
    {
        kernel_add_r_c_r_1<<<num_blocks, threads_per_block, shared_memory, sm>>>( c, c_row, c_col, a, b );
        return;
    }

    // [ r, c] -- [1, c]
    if ( (a_row == c_row) && (a_col == c_col) && (a_col == b_col) && (b_row == 1) )
    {
        kernel_add_r_c_1_c<<<num_blocks, threads_per_block, shared_memory, sm>>>( c, c_row, c_col, a, b );
        return;
    }

    // [ r, 1] -- [1, c]
    if ( (a_row == c_row) && (b_col == c_col) && (a_col == 1) && (b_row == 1) )
    {
        kernel_add_r_1_1_c<<<num_blocks, threads_per_block, shared_memory, sm>>>( c, c_row, c_col, a, b );
        return;
    }

    printf( "cuda_add cannot broadcast with (a_row, a_col) : (%d, %d) and (b_row, b_col) : (%d, %d)", a_row, a_col, b_row, b_col );
    abort(); // should not reach here
}



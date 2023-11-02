#ifndef MCUDA_ASSERT_HPP_INCLUDED_OPINAFSDLKJNSAFLKJNSADFLJNSAFDLKJNSFDLKJANSFFD
#define MCUDA_ASSERT_HPP_INCLUDED_OPINAFSDLKJNSAFLKJNSADFLJNSAFDLKJNSFDLKJANSFFD

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdio>

#ifdef cuda_assert
#undef cuda_assert
#endif

struct cuda_result_assert
{
    void operator()( const cudaError_t& result, const char* const file, const unsigned long line ) const
    {
        if ( cudaSuccess == result ) { return; }

        report_error( result, file, line );
    }

    void report_error( const cudaError_t& result, const char* const file, const unsigned long line ) const
    {
        printf( "%s:%lu: cuda runtime error occured:\n[[ERROR]]: %s %s\n", file, line, cudaGetErrorName(result), cudaGetErrorString(result) );
        abort();
    }

};//struct cuda_result_assert

#define cuda_assert(result) cuda_result_assert()(result, __FILE__, __LINE__)

inline void cuda_assert_no_error()
{
    cuda_assert( cudaGetLastError() );
}

#endif//_CUDA_ASSERT_HPP_INCLUDED_OPINAFSDLKJNSAFLKJNSADFLJNSAFDLKJNSFDLKJANSFFD


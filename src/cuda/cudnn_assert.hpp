#ifndef CUDNN_ASSERT_HPP_INCLUDED_SDPOIJASDLKJASDLKIASFD908UASDLKJSADLKASJLSADJF
#define CUDNN_ASSERT_HPP_INCLUDED_SDPOIJASDLKJASDLKIASFD908UASDLKJSADLKASJLSADJF

#include <cuda.h>
#include <cuda_runtime.h>
#include <cudnn.h>

#include <cstdio>

#ifdef cudnn_assert
#undef cudnn_assert
#endif

struct cudnn_result_assert
{
    void operator()( const cudnnStatus_t& result, const char* const file, const unsigned long line ) const
    {
        if ( CUDNN_STATUS_SUCCESS == result ) { return; }

        report_error( result, file, line );
    }

    void report_error( const cudnnStatus_t& result, const char* const file, const unsigned long line ) const
    {
        printf( "%s:%lu: cudnn runtime error occured:\n[[ERROR]]: %s\n", file, line, cudnnGetErrorString(result) );
        abort();
    }

};//struct cudnn_result_assert

#define cudnn_assert(result) cudnn_result_assert()(result, __FILE__, __LINE__)

#endif//CUDNN_ASSERT_HPP_INCLUDED_SDPOIJASDLKJASDLKIASFD908UASDLKJSADLKASJLSADJF


#ifndef CUDA_STREAM_HPP_INCLUDED_DSOIJASLKFDJ4I8UASFLKJASLKASJFLASKJSALKJSALKJSF
#define CUDA_STREAM_HPP_INCLUDED_DSOIJASLKFDJ4I8UASFLKJASLKASJFLASKJSALKJSALKJSF

#include "./cuda_assert.hpp"
#include "../../include/direct_space/stream.hpp"
#include "./cublas.hpp"

namespace nnl
{

    struct stream<cuda_engine>::cuda_stream
    {
        cudaStream_t sm_;

        cudaStream_t stream()
        {
            return sm_;
        }

        cuda_stream()
        {
            cuda_assert( cudaStreamCreate( &sm_ ) );
        }

        ~cuda_stream()
        {
            cuda_assert( cudaStreamDestroy( sm_ ) );
        }
    };//stream<cuda_engine>::cuda_stream

}//namespace nnl

#endif//CUDA_STREAM_HPP_INCLUDED_DSOIJASLKFDJ4I8UASFLKJASLKASJFLASKJSALKJSALKJSF


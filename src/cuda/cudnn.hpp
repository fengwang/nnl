#ifndef UQQDLBWAVWGBGKUNLKOSUPEEOIOEBHQIRRTVMUTJAIFYCHUKAACOTOBMYYJPNCFMVWUGIMRBJ
#define UQQDLBWAVWGBGKUNLKOSUPEEOIOEBHQIRRTVMUTJAIFYCHUKAACOTOBMYYJPNCFMVWUGIMRBJ

#include "./cuda_stream.hpp"
#include "./cudnn_assert.hpp"
#include "../../include/utility/utility.hpp"
#include "../../include/direct_space/stream.hpp"

namespace nnl
{

    struct cudnn_handle
    {
        cudnnHandle_t handle_;

        cudnn_handle()
        {
            cudnn_assert( cudnnCreate(&handle_) );
        }

        ~cudnn_handle()
        {
            cudnn_assert( cudnnDestroy(handle_) );
        }

        cudnnHandle_t handle()
        {
            return handle_;
        }

        void set_stream( stream<cuda_engine>& sm )
        {
            cudnn_assert( cudnnSetStream( handle_, sm.stream_->stream() ) );
        }

    };//struct cudnn_handle

    inline cudnn_handle& get_default_cudnn_handle()
    {
        return singleton<cudnn_handle>::instance();
    }


    template< typename T > requires std::floating_point<T>
    void cudnn_softmax( T* input, T* output, std::int32_t batch_size, std::int32_t dim_0, std::int32_t dim_1, std::int32_t dim_2, stream<cuda_engine>& sm,
                  T alpha, T beta, bool channel_first )
    {
        //spdlog::info( "Calling cudnn_softmax with bs {}, dim_0 {}, dim _1 {}, dim_2 {}, alpha {}, beta {}, channel {}", batch_size, dim_0, dim_1, dim_2, alpha, beta, channel_first );

        if constexpr( std::is_same_v<T, float> )
        {
            cudnnTensorDescriptor_t srcTensorDescriptor = nullptr;
            cudnn_assert( cudnnCreateTensorDescriptor( &srcTensorDescriptor ) );
            cudnnTensorDescriptor_t dstTensorDescriptor = nullptr;
            cudnn_assert( cudnnCreateTensorDescriptor( &dstTensorDescriptor ) );

            if (channel_first)//NCHW
            {
                cudnn_assert( cudnnSetTensor4dDescriptor( srcTensorDescriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch_size, dim_0, dim_1, dim_2 ) );
                cudnn_assert( cudnnSetTensor4dDescriptor( dstTensorDescriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch_size, dim_0, dim_1, dim_2 ) );
            }
            else //NHWC
            {
                cudnn_assert( cudnnSetTensor4dDescriptor( srcTensorDescriptor, CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT, batch_size, dim_0, dim_1, dim_2 ) );
                cudnn_assert( cudnnSetTensor4dDescriptor( dstTensorDescriptor, CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT, batch_size, dim_0, dim_1, dim_2 ) );
            }

            auto& hd = get_default_cudnn_handle();
            hd.set_stream( sm );

            cudnn_assert( cudnnSoftmaxForward( hd.handle(), CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL, &alpha, srcTensorDescriptor, input, &beta, dstTensorDescriptor, output ) );
            //cudnn_assert( cudnnSoftmaxForward( hd.handle(), CUDNN_SOFTMAX_FAST, CUDNN_SOFTMAX_MODE_CHANNEL, &alpha, srcTensorDescriptor, input, &beta, dstTensorDescriptor, output ) );

            //sm.synchronize();
        }
        else
        {
            better_assert( false, "Only float32 is implemented." );
        }

    }

}//namespace nnl

#endif//UQQDLBWAVWGBGKUNLKOSUPEEOIOEBHQIRRTVMUTJAIFYCHUKAACOTOBMYYJPNCFMVWUGIMRBJ

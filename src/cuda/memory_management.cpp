#include "./cuda_assert.hpp"
#include "../../include/direct_space/stream.hpp"
#include "./cublas.hpp"

#include "./cuda_stream.hpp"

extern "C"
{
    void* cuda_device_alloc( std::size_t n )
    {
        void* ans;
        cuda_assert( cudaMalloc( &ans, n ) );
        //cuda_assert( cudaMemset( ans, 0, n ) );
        return ans;
    }

    void cuda_device_free( void* ptr )
    {
        cuda_assert( cudaFree( ptr ) );
    }

    void* cuda_host_alloc( std::size_t n )
    {
        void* ans;
        cuda_assert( cudaMallocHost( &ans, n ) );
        return ans;
    }

    void cuda_host_free( void* ptr )
    {
        cuda_assert( cudaFreeHost( ptr ) );
    }
}

namespace nnl
{

    stream<cuda_engine>::stream() : stream_{ std::make_unique<stream<cuda_engine>::cuda_stream>() } {}
    stream<cuda_engine>::~stream()
    {
        (*this).synchronize();
    }
    stream<cuda_engine>::stream( stream<cuda_engine>&&) = default;
    stream<cuda_engine>& stream<cuda_engine>::operator = ( stream<cuda_engine>&&) = default;

    void stream<cuda_engine>::synchronize()
    {
        cudaError_t const ret = cudaStreamQuery( ((*this).stream_)->stream() );
        if ( ret == cudaErrorNotReady )
        {
            cuda_assert( cudaStreamSynchronize( ((*this).stream_)->stream() ) );
            return;
        }
        cuda_assert( ret );
    }

    void host2device( std::byte* src, std::size_t n, std::byte* dst, stream<cuda_engine>& e )
    {
        cuda_assert( cudaMemcpyAsync( dst, src, n, cudaMemcpyHostToDevice, (e.stream_)->stream() ) );
    }

    void device2host( std::byte* src, std::size_t n, std::byte* dst, stream<cuda_engine>& e )
    {
        cuda_assert( cudaMemcpyAsync( dst, src, n, cudaMemcpyDeviceToHost, (e.stream_)->stream() ) );
    }

    void device2device( std::byte* src, std::size_t n, std::byte* dst, stream<cuda_engine>& e )
    {
        if( !( (((src+n) <= dst) || (src >= (dst + n))) &&  (((dst+n) <= src) || (dst >= (src + n))) ) ) // has overlaps
        {
            spdlog::warn( nnl::format("memory areas do not overlap, with received src={}, n={}, dst={}, switching to cudaMemcpy at a price of potentional performance loss.", src, n, dst) );
            cuda_assert( cudaMemcpy( dst, src, n, cudaMemcpyDeviceToDevice ) );
            return;
        }
        better_assert( (((src+n) <= dst) || (src >= (dst + n))), nnl::format("memory areas may not overlap, but received src={}, n={}, dst={}", src, n, dst) );
        better_assert( (((dst+n) <= src) || (dst >= (src + n))), nnl::format("memory areas may not overlap, but received src={}, n={}, dst={}", src, n, dst) );
        cuda_assert( cudaMemcpyAsync( dst, src, n, cudaMemcpyDeviceToDevice, (e.stream_)->stream() ) );
    }

    template<Engine E>
    std::tuple<std::int64_t, std::byte*> allocate_maximum_device_memory();

    template<>
    std::tuple<std::int64_t, std::byte*> allocate_maximum_device_memory<cuda_engine>()
    {
        // set up here
        //cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);

        void* ans;
        std::int64_t const step = 1LL*1024*1024*1024; // <- step 1 GB
        std::int64_t n = 1LL*1024*1024*1024; // <-- start test at 1GB

        // allocate all the way to maximum
        while ( 0 ==  cudaMalloc( &ans, n ) )
        {
            spdlog::info( "allocating maximum device memory:: OK with {} bytes.", n );
            n += step;
            cudaFree( ans );
        }

        // reset errors
        {
            auto err = cudaGetLastError();
            (void) err;
        }

        std::int64_t const max_mem_in_bytes = n - step;
        return std::make_tuple( max_mem_in_bytes, reinterpret_cast<std::byte*>(cuda_device_alloc(max_mem_in_bytes)) );
    }

    //defined in ./cublas.hpp
    void cublas_handle::set_stream( stream<cuda_engine>& sm )
    {
        cublas_assert( cublasSetStream( (*this).handle_, (sm.stream_)->stream() ) );
    }

    template< Engine E >
    void reset_device();

    template<>
    void reset_device<cuda_engine>()
    {
        cuda_assert( cudaDeviceReset() );
    }

}//namespace nnl



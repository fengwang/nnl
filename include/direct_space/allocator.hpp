#ifndef GIUSJDESPICIHFGWUVPWWOTJCDTFKENOICEYBJCELBKUENJPSWHSFHBQASOJBUPKWBXLBRNFC
#define GIUSJDESPICIHFGWUVPWWOTJCDTFKENOICEYBJCELBKUENJPSWHSFHBQASOJBUPKWBXLBRNFC

#include "../utility/utility.hpp"
#include "./engine.hpp"

extern "C"
{
    void* cuda_device_alloc( std::size_t n );
    void  cuda_device_free( void* ptr );
    void* cuda_host_alloc( std::size_t n );
    void  cuda_host_free( void* ptr );
}

namespace nnl
{
    namespace allocator_
    {
        template< typename A, typename T >
        struct allocator_interface
        {
            typedef std::uint64_t size_type;
            typedef std::int64_t  difference_type;
            typedef T value_type;
            typedef value_type* pointer;

            void construct( ... ) {}
            void destroy( ... ) {}
            std::uint64_t max_size() const { return -1; }

            [[nodiscard]] pointer allocate( std::uint64_t n ); //NOT to be implemented
            void deallocate( pointer p, std::uint64_t n ); //NOT to be implemented
        };//struct allocator_interface

    }//namespace allocator_

    template< Engine E, typename T >
    struct host_allocator : allocator_::allocator_interface< host_allocator<E, T>, T> {};

    template< Engine E, typename T >
    struct device_allocator : allocator_::allocator_interface< device_allocator<E, T>, T> {};

    template< typename T >
    struct host_allocator<cuda_engine, T >
    {
        typedef std::uint64_t size_type;
        typedef std::int64_t  difference_type;
        typedef T value_type;
        typedef value_type* pointer;

        template< typename U >
        struct rebind
        {
            typedef host_allocator<cuda_engine, U> other;
        };

        void construct( ... ) {}
        void destroy( ... ) {}
        std::uint64_t max_size() const { return -1; }

        [[nodiscard]] pointer allocate( std::uint64_t n )
        {
            return reinterpret_cast<pointer>( cuda_host_alloc( n * sizeof(value_type) ) );
        }

        void deallocate( pointer p, [[maybe_unused]]std::uint64_t n=0 )
        {
            //spdlog::info( format( "host_allocator::deallocating pointer at {} with length {}.", p, n ) );
            cuda_host_free( p );
        }
    };

    template<>
    struct host_allocator<cuda_engine, void >
    {
        typedef std::uint64_t size_type;
        typedef std::int64_t  difference_type;
        typedef void value_type;
        typedef value_type* pointer;

        template< typename U >
        struct rebind
        {
            typedef host_allocator<cuda_engine, U> other;
        };
    };

    // nnl::device_allocator<nnl::cuda_engine, int>
    template< typename T >
    struct device_allocator<cuda_engine, T >
    {
        typedef std::uint64_t size_type;
        typedef std::int64_t  difference_type;
        typedef T value_type;
        typedef value_type* pointer;

        template< typename U >
        struct rebind
        {
            typedef device_allocator<cuda_engine, U> other;
        };

        void construct( ... ) {}
        void destroy( ... ) {}
        std::uint64_t max_size() const { return -1; }

        [[nodiscard]] pointer allocate( std::uint64_t n )
        {
            return reinterpret_cast<pointer>( cuda_device_alloc( n * sizeof(value_type) ) );
        }

        void deallocate( pointer p, [[maybe_unused]]std::uint64_t n=0 )
        {
            //spdlog::info( format( "device_allocator::deallocating pointer at {} with length {}.", p, n ) );
            cuda_device_free( p );
        }
    };

    template<>
    struct device_allocator<cuda_engine, void >
    {
        typedef std::uint64_t size_type;
        typedef std::int64_t  difference_type;
        typedef void value_type;
        typedef value_type* pointer;

        template< typename U >
        struct rebind
        {
            typedef device_allocator<cuda_engine, U> other;
        };
    };


    //TODO: host/device allocators for rocm and opencl


}//namespace nnl

#endif//GIUSJDESPICIHFGWUVPWWOTJCDTFKENOICEYBJCELBKUENJPSWHSFHBQASOJBUPKWBXLBRNFC


#include "../../include/direct_space/graph.hpp"
#include "../../include/direct_space/node.hpp"
#include "../../include/direct_space/computation_table.hpp"
#include "../../include/direct_space/allocator.hpp"
#include "../../include/direct_space/engine.hpp"
#include "../../include/utility/wheels/cached_allocator.hpp"

TEST_CASE( "host-vector", "[host-vector]" )
{
    using namespace nnl;

    std::vector<int, host_allocator<cuda_engine, int>> vec;
    vec.resize( 1024 );

    for ( int i = 0; i < 1024; ++i )
        vec[i] = i;

    for ( int i = 0; i < 1024; ++i )
        REQUIRE( vec[i] == i );
}


TEST_CASE( "device-vector", "[device-vector]" )
{
    using namespace nnl;

    std::vector<int, device_allocator<cuda_engine, int>> vec;
    vec.resize( 1024 );

    // std::_Vector_base<int, nnl::device_allocator<nnl::cuda_engine, int> >::allocator_type {aka nnl::device_allocator<nnl::cuda_engine, int>}
    //int i = vec.get_allocator(); // type of the allocator

    device_allocator<nnl::cuda_engine, int> alloc;
    int* ptr = alloc.allocate( 10 );
    alloc.deallocate( ptr );


    REQUIRE( true );
}


#if 0
TEST_CASE( "cached-host-vector", "[cached-host-vector]" )
{
    using namespace nnl;

    std::vector<int, cached_allocator<int, host_allocator<cuda_engine, std::byte>>> vec;
    vec.resize( 1024 );

    for ( int i = 0; i < 1024; ++i )
        vec[i] = i;

    for ( int i = 0; i < 1024; ++i )
        REQUIRE( vec[i] == i );

    std::int64_t address = 0;
    {
        std::vector<int, cached_allocator<int, host_allocator<cuda_engine, std::byte>>> vec2 = vec;
        for ( int i = 0; i < 1024; ++i )
        {
            REQUIRE( 0 == vec2[i] );
        }
        address = (std::int64_t)std::addressof( vec2[0] );
    }

    std::vector<int, cached_allocator<int, host_allocator<cuda_engine, std::byte>>> vec2 = vec;
    for ( int i = 0; i < 1024; ++i )
    {
        REQUIRE( 0 == vec2[i] );
    }

    REQUIRE( address == (std::int64_t)std::addressof(vec2[0]) );

}
#endif
namespace nnl
{
    template< Engine E >
    std::tuple<std::int64_t, std::byte*> allocate_maximum_device_memory();
    template<>
    std::tuple<std::int64_t, std::byte*> allocate_maximum_device_memory<cuda_engine>();
}

TEST_CASE( "Max-avaliable-memory", "[max-cuda-memory]" )
{
    auto [max_mem, address] = nnl::allocate_maximum_device_memory<nnl::cuda_engine>();
    std::cout << "Max cuda available memory is " << max_mem << " bytes, and starting at " << std::int64_t(address) << std::endl;
    cuda_device_free( address );
    REQUIRE( true );
}




TEST_CASE( "device-memory-vector", "[device-memory-vector]" )
{
    std::cout << "Testing device memory vector." << std::endl;
    using namespace nnl;

    for ( std::size_t i = 16; i < 512+128; i += i>>1 )
    {
        std::cout << "Trying with " << i << " M intergers." << std::endl;
        std::vector<int, device_allocator<cuda_engine, int>> vec;
        vec.resize( 1024*1024*i );
        std::cout << "Allocated " << i*sizeof(int) << "MB memory.\n";
    }

    std::cout << "All test of device memory vector done." << std::endl;


    REQUIRE( true );
}



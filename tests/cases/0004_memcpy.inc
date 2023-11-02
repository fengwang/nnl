#include "../../include/direct_space/graph.hpp"
#include "../../include/direct_space/node.hpp"
#include "../../include/direct_space/computation_table.hpp"
#include "../../include/direct_space/allocator.hpp"
#include "../../include/direct_space/engine.hpp"
#include "../../include/direct_space/stream.hpp"
#include "../../include/utility/wheels/cached_allocator.hpp"

namespace nnl
{
    void host2device( std::byte* src, std::size_t n, std::byte* dst, stream<cuda_engine>& e );
    void device2host( std::byte* src, std::size_t n, std::byte* dst, stream<cuda_engine>& e );
    void device2device( std::byte* src, std::size_t n, std::byte* dst, stream<cuda_engine>& e );
}

TEST_CASE( "memcpyh2dd2hd2d", "[memcpy]" )
{
    std::cout << "Test case 4 of memcpy started" << std::endl;

    using namespace nnl;
    stream<cuda_engine> e;

    for ( unsigned long n = 1024; n <= 1024*1024; n <<= 1 )
    {
        std::vector<int, host_allocator<cuda_engine, int>> hvec;
        std::vector<int, host_allocator<cuda_engine, int>> hhvec;
        std::vector<int, device_allocator<cuda_engine, int>> dvec;
        std::vector<int, device_allocator<cuda_engine, int>> ddvec;

        hhvec.resize( n );
        hvec.resize( n );
        dvec.resize( n );
        ddvec.resize( n );

        for ( unsigned long i = 0; i < n; ++i )
            hvec[i] = i;

        host2device( reinterpret_cast<std::byte*>(hvec.data()), n*sizeof(int), reinterpret_cast<std::byte*>(dvec.data()), e );
        device2device( reinterpret_cast<std::byte*>(dvec.data()), n*sizeof(int), reinterpret_cast<std::byte*>(ddvec.data()), e );
        device2host( reinterpret_cast<std::byte*>(ddvec.data()), n*sizeof(int), reinterpret_cast<std::byte*>(hhvec.data()), e );
        //device2host( reinterpret_cast<std::byte*>(dvec.data()), n*sizeof(int), reinterpret_cast<std::byte*>(hhvec.data()), e );

        e.synchronize();

        for ( unsigned long i = 0; i < n; ++i )
        {
            CHECK( (hvec[i] == hhvec[i]) );
        }


    }

}


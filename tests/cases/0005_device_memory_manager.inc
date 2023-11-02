#include "../../include/direct_space/graph.hpp"
#include "../../include/direct_space/node.hpp"
#include "../../include/direct_space/computation_table.hpp"
#include "../../include/direct_space/allocator.hpp"
#include "../../include/direct_space/engine.hpp"
#include "../../include/direct_space/stream.hpp"
#include "../../include/utility/wheels/cached_allocator.hpp"
#include "../../include/direct_space/session.hpp"
#include "../../include/direct_space/device.hpp"

namespace nnl
{
    void host2device( std::byte* src, std::size_t n, std::byte* dst, stream<cuda_engine>& e );
    void device2host( std::byte* src, std::size_t n, std::byte* dst, stream<cuda_engine>& e );
    void device2device( std::byte* src, std::size_t n, std::byte* dst, stream<cuda_engine>& e );
}

TEST_CASE( "dmm", "[dmm]" )
{
    std::cout << "Test case 5 of dmm started" << std::endl;

    using namespace nnl;

    //session<cuda_engine>& ss = get_default_session<cuda_engine>();
    //device_memory_manager<cuda_engine>& dmm = ss.device_memory_manager_;
    device_memory_manager<cuda_engine> dmm;// = ss.device_memory_manager_;
    std::cout << "On dmm construction.\n";
    std::cout << dmm << std::endl;

    std::int64_t n = 1024;
    std::vector<device_memory> record;

    record.emplace_back( dmm.acquire_memory( n ) );
    std::cout << "After mallocing 1 block:\n";
    std::cout << dmm << std::endl;

    record.emplace_back( dmm.acquire_memory( n ) );
    record.emplace_back( dmm.acquire_memory( n ) );
    record.emplace_back( dmm.acquire_memory( n ) );

    std::cout << "After mallocing 4 blocks:\n";
    std::cout << dmm << std::endl;


    dmm.dismiss_memory( record[0] );
    std::cout << "After dismissing the first block:\n";
    std::cout << dmm << std::endl;

    REQUIRE( record[0] == dmm.acquire_memory(n) );
    std::cout << "After acquiring memory again:\n";
    std::cout << dmm << std::endl;

    dmm.dismiss_memory( record[1] );
    std::cout << "After dismissing the first two blocks:\n";
    std::cout << dmm << std::endl;

    record[1] = dmm.acquire_memory(n+n);
    std::cout << "After acquiring memory two times larger:\n";
    std::cout << dmm << std::endl;

    dmm.dismiss_memory( record[2] );
    dmm.dismiss_memory( record[3] );
    std::cout << "After dismissing more memory:\n";
    std::cout << dmm << std::endl;

    dmm.align_assigned_memory();
    std::cout << "After aligning assigned memory:\n";
    std::cout << dmm << std::endl;


    REQUIRE( true );
}


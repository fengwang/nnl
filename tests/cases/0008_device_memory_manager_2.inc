#include "../../include/direct_space/graph.hpp"
#include "../../include/direct_space/node.hpp"
#include "../../include/direct_space/computation_table.hpp"
#include "../../include/direct_space/allocator.hpp"
#include "../../include/direct_space/engine.hpp"
#include "../../include/direct_space/stream.hpp"
#include "../../include/utility/wheels/cached_allocator.hpp"
#include "../../include/direct_space/session.hpp"
#include "../../include/direct_space/device.hpp"


TEST_CASE( "dmm-2", "[dmm-2]" )
{
    std::cout << "Test case 8 of dmm-2 started." << std::endl;

    using namespace nnl;

    session<cuda_engine>& ss = get_default_session<cuda_engine>();
    device_memory_manager<cuda_engine>& dmm = ss.device_memory_manager_;
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
    record.emplace_back( dmm.acquire_memory( n ) );
    record.emplace_back( dmm.acquire_memory( n ) );
    record.emplace_back( dmm.acquire_memory( n ) );
    record.emplace_back( dmm.acquire_memory( n ) );

    std::cout << "After mallocing 8 blocks:\n";
    std::cout << dmm << std::endl;


    dmm.dismiss_memory( record[7] );
    dmm.dismiss_memory( record[6] );
    std::cout << "After dismissing the last two blocks:\n";
    std::cout << dmm << std::endl;

    dmm.dismiss_memory( record[4] );
    dmm.dismiss_memory( record[3] );
    dmm.dismiss_memory( record[2] );
    dmm.dismiss_memory( record[1] );
    dmm.dismiss_memory( record[0] );
    std::cout << "After dismissing the first 5 blocks:\n";
    std::cout << dmm << std::endl;


    dmm.align_assigned_memory();
    std::cout << "After aligning memory:\n";
    std::cout << dmm << std::endl;

    REQUIRE( true );
}


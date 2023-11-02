#include "../../include/direct_space/graph.hpp"
#include "../../include/direct_space/node.hpp"
#include "../../include/direct_space/computation_table.hpp"
#include "../../include/direct_space/allocator.hpp"
#include "../../include/direct_space/engine.hpp"
#include "../../include/direct_space/stream.hpp"
#include "../../include/utility/wheels/cached_allocator.hpp"
#include "../../include/direct_space/session.hpp"
#include "../../include/direct_space/device.hpp"
#include "../../include/direct_space/tensor.hpp"
#include "../../include/utility/wheels/view.hpp"
#include "../../include/direct_space/layer.hpp"
#include "../../include/direct_space/model.hpp"

TEST_CASE( "tensor-load-memory-38", "[tensor-load-memory-38]" )
{

    spdlog::info( "\nTest case of 38 started.\n" );

    std::this_thread::sleep_for( std::chrono::seconds( 1 ) );

    using namespace nnl;
    auto& sess = get_default_session<default_engine_type>();
    sess.clean();

    std::int64_t const n = 100;

    auto t = make_tensor<default_engine_type>( {100,}, "float32" );
    t.load_memory( "./tests/testdata/0038/data.bin" );

    float* dat = reinterpret_cast<float*>( t.data() );

    for ( auto idx : range( n ) )
    {
        CHECK( std::abs( dat[idx] - 1.0*idx ) < 1.0e-5 );
    }
}


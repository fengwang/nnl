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

TEST_CASE( "Node-Buffer", "[node-buffer]" )
{
    using namespace nnl;

    {
        auto n1 = make_node( "Linear", "linear_node" );

        n1.set_buffer( {128, 3, 3, 256}, "float32" );

        auto& sess = get_default_session<default_engine_type>();
        std::cout << "After making node, the session is\n" << sess << "\n";;

        n1.weight_preloading();
        std::cout << "After loading weights, the session is\n" << sess << "\n";;

        n1.weight_dismissing();
        std::cout << "After unloading weights, the session is\n" << sess << "\n";;
    }

    REQUIRE( true );
}


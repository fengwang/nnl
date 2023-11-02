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

TEST_CASE( "tensor-2", "[tensor-2]" )
{
    std::cout << "Test case 9 of tensor-2" << std::endl;

    using namespace nnl;

    {
        auto& sess = get_default_session<cuda_engine>();

        auto t = make_tensor<cuda_engine>( {1, 2, 3, 4, 5, 6, 7, 8,}, "float32" );
        [[maybe_unused]] auto const& dm = sess.tensor_to_device( t );
        std::cout << "After tensor_to_device for the first time, the session is\n" << sess << std::endl;
        t.reshape( {2, 3, 4, 5, 6, 7} );
        sess.tensor_to_device( t ); // Expect warning here

        std::cout << "After tensor_to_device for the second time, the session is\n" << sess << std::endl;

        REQUIRE( true );
    }

}


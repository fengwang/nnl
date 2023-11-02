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

TEST_CASE( "Node", "[node]" )
{
    std::cout << "Test case 7 of node started." << std::endl;

    using namespace nnl;

    {
        auto n1 = make_node( "Linear", "linear_node" );
        {
            auto conv_tsor = make_tensor<default_engine_type>( {32, 3, 3, 32}, "float32", "conv1_kernel" );
            float* data = reinterpret_cast<float*>(conv_tsor.data());
            for ( int i = 0; i < 32*3*3*32; ++i )
                data[i] = (i*i - 127*i) * 1.0f;
            n1.add_weight( conv_tsor.name() );
        }
        {
            auto bias_tsor = make_tensor<default_engine_type>( std::vector<std::int64_t>{{32,}}, "float32", "bias_weight" );
            float* data = reinterpret_cast<float*>(bias_tsor.data());
            for ( int i = 0; i < 32; ++i )
                data[i] = (i*i - 127*i) * 1.0f;
            n1.add_weight( bias_tsor.name() );
        }

        auto& sess = get_default_session<default_engine_type>();
        std::cout << "After making node, the session is\n" << sess << "\n";;

        n1.weight_preloading();
        std::cout << "After loading weights, the session is\n" << sess << "\n";;

        n1.weight_dismissing();
        std::cout << "After unloading weights, the session is\n" << sess << "\n";;
    }

    {
        auto n1 = make_node( "Linear", "linear_node" );
        {
            auto conv_tsor = make_tensor<default_engine_type>( {32, 3, 3, 64}, "float32", "conv1_kernel_se" );
            float* data = reinterpret_cast<float*>(conv_tsor.data());
            for ( int i = 0; i < 32*3*3*64; ++i )
                data[i] = (i*i - 127*i) * 1.0f;
            n1.add_weight( conv_tsor.name() );
        }
        {
            auto bias_tsor = make_tensor<default_engine_type>( std::vector<std::int64_t>{{64,}}, "float32", "bias_weight_se" );
            float* data = reinterpret_cast<float*>(bias_tsor.data());
            for ( int i = 0; i < 64; ++i )
                data[i] = (i*i - 127*i) * 1.0f;
            n1.add_weight( bias_tsor.name() );
        }

        auto& sess = get_default_session<default_engine_type>();
        std::cout << "After making node, the session is\n" << sess << std::endl;

        n1.weight_preloading();
        std::cout << "After loading weights, the session is\n" << sess << std::endl;


        sess.get_device_memory_manager().align_assigned_memory();
        std::cout << "After aligning memory, the session is\n" << sess << "\n";;

        n1.weight_dismissing();
        std::cout << "After unloading weights, the session is\n" << sess << "\n";;
    }

    REQUIRE( true );
}


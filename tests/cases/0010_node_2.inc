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

TEST_CASE( "Node-2", "[node-2]" )
{
    std::cout << "Test case 10 of node-2" << std::endl;
    using namespace nnl;

    {
        std::int64_t dim = 1024*16;

        // initialize nodes
        auto inp = make_node( "Input", "inp" );
        auto l1 = make_node( "Linear", "l1" );
        l1.add_attribute( make_attribute_dtype( "float32" ) );
        l1.attributes();
        auto l2 = make_node( "Linear", "l2" );
        auto inp_t = make_tensor<default_engine_type>( {1, dim}, "float32", "inp_t" );
        auto l1_w = make_tensor<default_engine_type>( {dim, dim}, "float32", "l1_w" );
        auto l1_b = make_tensor<default_engine_type>( {dim,}, "float32", "l1_b" );
        auto l1_o = make_tensor<default_engine_type>( {1, dim}, "float32", "l1_o" );
        auto l2_w = make_tensor<default_engine_type>( {dim, dim}, "float32", "l2_w" );
        auto l2_b = make_tensor<default_engine_type>( {dim,}, "float32", "l2_b" );
        auto l2_o = make_tensor<default_engine_type>( {1, dim}, "float32", "l2_o" );
        {
            inp.add_input( inp_t.name() );
        }
        {
            l1.add_input( inp_t.name() ); // also output of inp
            l1.add_weight( l1_w.name() );
            l1.add_weight( l1_b.name() );
            l1.add_output( l1_o.name() );
        }
        {
            l2.add_input( l1_o.name() );
            l2.add_weight( l2_w.name() );
            l2.add_weight( l2_b.name() );
            l2.add_output( l2_o.name() );
        }

        // create computation graph
        graph<node> g;
        {
            g.connect( inp, l1 );
            g.connect( l1, l2 );
        }

        auto computation_table = make_computation_table( g );

        std::cout << "Get computation table" << std::endl;
        for ( auto & ct : computation_table )
            std::cout << ct << std::endl;

        auto& sess = get_default_session<default_engine_type>();
        for ( auto & ct : computation_table )
        {
            std::cout << "Before computation runs, the session is\n" << sess << std::endl;
            auto& [ao, host2device_operations, device2host_operations, device2host_dismiss_operations,
                  wp, wd, op] = ct;
            {
                std::cout << "arithmetic operations" << std::endl;
                if ( ao )
                    ao.arithmetic_operation();

                std::cout << "host2device operations" << std::endl;
                for ( auto h2do : host2device_operations )
                    if (h2do)
                        h2do.host2device_operation();

                std::cout << "device2host operations" << std::endl;
                for ( auto d2ho : device2host_operations )
                    if (d2ho)
                        d2ho.device2host_operation();

                std::cout << "device2hostdismiss operations" << std::endl;
                for ( auto d2hdo : device2host_dismiss_operations )
                    if (d2hdo)
                        d2hdo.device2host_dismiss_operation();

                std::cout << "weight preloading operations" << std::endl;
                if (wp)
                    wp.weight_preloading();

                std::cout << "weight dismissing operations" << std::endl;
                if (wd)
                    wd.weight_dismissing();

                std::cout << "output preallocating" << std::endl;
                if (op)
                    op.output_preallocating();
            }

            sess.get_device_memory_manager().synchronize();
        }

        std::cout << "After inference, the session is\n" << sess << std::endl;

    }

    REQUIRE( true );
}


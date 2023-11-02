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

TEST_CASE( "Node-stacked-linear-20", "[node-stacked-linear-20]" )
{
    spdlog::info( "\nTest-Case of Node-Stacked-Linear-20\n" );

    using namespace nnl;
    auto& sess = get_default_session<default_engine_type>();
    sess.clean();

    {
        std::int64_t rows = 128;
        std::uint64_t urows = static_cast<std::uint64_t>(rows);
        std::int64_t cols = 16*1024;
        //std::int64_t cols = 13*1024;
        std::uint64_t ucols = static_cast<std::uint64_t>(cols);
        //float cst_2 = 1.234567f;
        float cst_2 = 0.0f;

        // initialize nodes
        auto inp = make_node( "Input", "inp" );

        // input layer
        auto inp_t = make_tensor<default_engine_type>( {rows, cols}, "float32", "inp_t" );
        {
            auto mat = view_2d{ reinterpret_cast<float*>( inp_t.data() ), urows, ucols } ;
            for ( auto c : range(cols) )
            {
                for ( auto r : range(rows) )
                {
                    mat[r][c] = c * 1.0f;
                }
            }
        }
        inp.add_input( inp_t );


        auto l0 = make_node( "Linear", "l0" );
        {
            auto l_w = make_tensor<default_engine_type>( {cols, cols}, "float32", "l_w" );
            {
                auto mat = view_2d{ reinterpret_cast<float*>( l_w.data() ), ucols, ucols };
                for ( auto r : range(cols) )
                    for ( auto c : range(cols) )
                        mat[r][c] = (r==c) ? 1.0f : 0.0f;
            }
            auto l_b = make_tensor<default_engine_type>( {cols,}, "float32", "l_b" );
            {
                auto mat = view_1d{ reinterpret_cast<float*>( l_b.data() ), ucols } ;
                for ( auto c : range(cols) )
                    mat[c] = 0.0f;
            }
            l0.set_weights( {l_w, l_b,} );
        }

        auto l1 = make_node( "Linear", "l1" );
        {
            auto l_w = make_tensor<default_engine_type>( {cols, cols}, "float32", "l_w" );
            {
                auto mat = view_2d{ reinterpret_cast<float*>( l_w.data() ), ucols, ucols };
                for ( auto r : range(cols) )
                    for ( auto c : range(cols) )
                        mat[r][c] = (r==c) ? 1.0f : 0.0f;
            }
            auto l_b = make_tensor<default_engine_type>( {cols,}, "float32", "l_b" );
            {
                auto mat = view_1d{ reinterpret_cast<float*>( l_b.data() ), ucols } ;
                for ( auto c : range(cols) )
                    mat[c] = -c * 2.0f + cst_2;
            }
            l1.set_weights( {l_w, l_b,} );
        }
        l1.set_output_node();

        // create computation graph
        graph<node> g;
        {
            g.connect( inp, l0 );
            g.connect( l0, l1 );
        }

        auto computation_table = make_computation_table( g );

        g.inference_io_shapes();

        std::cout << "Get computation table" << std::endl;
        for ( auto & ct : computation_table )
            std::cout << ct << std::endl;


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
            std::cout << "synchronize" << std::endl;

            //sess.get_device_memory_manager().synchronize();
            sess.synchronize();
        }


        sess.synchronize();
        std::vector<std::string> outputs = l1.outputs();
        better_assert( (1==outputs.size()), "Expecting one output tensor." );
        auto l1_o = sess.find_tensor_by_name( outputs[0] );

        CHECK( l1_o.synchronize_to_host() );
        {
            sess.synchronize();
            auto mat = view_2d{ reinterpret_cast<float*>( l1_o.data() ), urows, ucols } ;
            for ( auto c : range(cols) )
            {
                for ( auto r : range(rows) )
                {
                    if ( ! (std::abs(mat[r][c]-cst_2+c) <= 1.0e-2f) )
                    {
                        spdlog::error( nnl::format( "mat[{}][{}] = {}, but {} is expected.", r, c, mat[r][c], -c+cst_2 ) );
                    }
                    REQUIRE( std::abs(mat[r][c]-cst_2+c) <= 1.0e-2f ); // stop at the first error
                }
            }
        }

        std::cout << "After inference, the session is\n" << sess << std::endl;
    }

    REQUIRE( true );
}


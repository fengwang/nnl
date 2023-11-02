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

TEST_CASE( "Node-2-11", "[node-2-11]" )
{
    std::cout << "Test case 11 of node-2-11" << std::endl;
    using namespace nnl;

    {
        std::int64_t bs = 128;
        std::uint64_t ubs = 128;
        std::int64_t dim = 1024*16;
        std::uint64_t udim = static_cast<std::uint64_t>(dim);
        float cst = 1.234567f;

        // initialize nodes
        auto inp = make_node( "Input", "inp" );
        auto l1 = make_node( "Linear", "l1" );
        l1.set_output_node();
        auto inp_t = make_tensor<default_engine_type>( {bs, dim}, "float32", "inp_t" );
        {
            auto mat = view_2d{ reinterpret_cast<float*>( inp_t.data() ), ubs, udim } ;
            for ( auto idx : range(dim) )
            {
                for ( auto jdx : range(bs) )
                {
                    mat[jdx][idx] = idx * 1.0f;
                }
            }
        }
        auto l1_w = make_tensor<default_engine_type>( {dim, dim}, "float32", "l1_w" );
        {
            auto mat = view_2d{ reinterpret_cast<float*>( l1_w.data() ), udim, udim };
            for ( auto r : range(dim) )
                for ( auto c : range(dim) )
                    mat[r][c] = (r==c) ? 1.0f : 0.0f;
        }
        auto l1_b = make_tensor<default_engine_type>( {dim,}, "float32", "l1_b" );
        if (1)
        {
            auto mat = view_1d{ reinterpret_cast<float*>( l1_b.data() ), udim } ;
            for ( auto idx : range(dim) )
                mat[idx] = -idx *2.0f + cst;
        }
        auto l1_o = make_tensor<default_engine_type>( {bs, dim}, "float32", "l1_o" );
        {
            inp.add_input( inp_t );
        }
        {
            l1.add_input( inp_t ); // also output of inp
            l1.add_weight( l1_w );
            l1.add_weight( l1_b );
            l1.add_output( l1_o );
        }

        // create computation graph
        graph<node> g;
        {
            g.connect( inp, l1 );
        }

        //std::vector<node> ordered_nodes = g.computation_order();
        //auto const& computation_table = make_computation_table( ordered_nodes, g.edges() );

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

        CHECK( l1_o.synchronize_to_host() );
        {
            sess.synchronize();
            auto mat = view_2d{ reinterpret_cast<float*>( l1_o.data() ), ubs, udim } ;
            for ( auto idx : range(dim) )
            {
                for ( auto jdx : range(bs) )
                {
                    CHECK( std::abs(mat[jdx][idx]-cst+idx) <= 1.0e-2f );
                }
            }
        }

        std::cout << "After inference, the session is\n" << sess << std::endl;
    }

    REQUIRE( true );
}


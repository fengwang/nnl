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

TEST_CASE( "Node-scaled_offset-16", "[node-scaled_offset-16]" )
{
    using namespace nnl;

    std::cout << "Testing Node-scaled_offset-16" << std::endl;

    {
        //std::int64_t bs = 16;
        //std::int64_t dim = 8;
        std::int64_t bs = 10;
        std::int64_t dim = 768;
        std::uint64_t ubs = static_cast<std::uint64_t>(bs);
        std::uint64_t udim = static_cast<std::uint64_t>(dim);

        // initialize nodes
        auto inp = make_node( "Input", "inp" );
        auto alpha_inp = make_node( "Input", "alpha_inp" );
        auto beta_inp = make_node( "Input", "beta_inp" );
        auto l1 = make_node( "ScaledOffset", "l1" );
        l1.set_output_node();
        auto inp_t = random<default_engine_type>( {bs, dim}, -1.0f, 1.0f, "float32", "inp_t" );
        auto alpha = random<default_engine_type>( {dim,}, -1.0f, 1.0f, "float32", "inp_alpha" );
        auto beta = random<default_engine_type>( {dim,}, -1.0f, 1.0f, "float32", "inp_beta" );

        auto gt_t = make_tensor<default_engine_type>( {bs, dim}, "float32", "gt_t" );
        {
            auto mat = view_2d{ reinterpret_cast<float*>( inp_t.data() ), ubs, udim } ;
            auto nat = view_2d{ reinterpret_cast<float*>( gt_t.data() ), ubs, udim } ;
            auto a = view_2d{ reinterpret_cast<float*>( alpha.data() ), 1LL, udim } ;
            auto b = view_2d{ reinterpret_cast<float*>( beta.data() ), 1LL, udim } ;

            for ( auto jdx : range(bs) )
            {
                for ( auto idx : range(dim) )
                {
                    nat[jdx][idx] = mat[jdx][idx] * a[0][idx] + b[0][idx];
                }
            }
        }

        auto l1_o = make_tensor<default_engine_type>( {bs, dim}, "float32", "l1_o" );
        {
            inp.add_input( inp_t );
            alpha_inp.add_input( alpha );
            beta_inp.add_input( beta );
        }
        {
            l1.add_input( inp_t );
            l1.add_input( alpha );
            l1.add_input( beta );
            l1.add_output( l1_o );
        }

        // create computation graph
        graph<node> g;
        {
            g.connect( inp, l1 );
            g.connect( alpha_inp, l1 );
            g.connect( beta_inp, l1 );
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
            sess.get_device_memory_manager().synchronize();
            auto mat = view_2d{ reinterpret_cast<float*>( l1_o.data() ), ubs, udim } ;
            auto nat = view_2d{ reinterpret_cast<float*>( gt_t.data() ), ubs, udim } ;
            /*
            std::cout << "Output:\n";
            for ( auto jdx : range(bs) )
            {
                for ( auto idx : range(dim) )
                {
                    std::cout << mat[jdx][idx] << " -- " << nat[jdx][idx] << "\t";
                }
                std::cout << "\n";
            }
            */
            for ( auto jdx : range(bs) )
            {
                for ( auto idx : range(dim) )
                {
                    CHECK( (std::abs(mat[jdx][idx]-nat[jdx][idx]) < 1.0e-5f) );
                }
            }
        }

        std::cout << "After inference, the session is\n" << sess << std::endl;
    }

    REQUIRE( true );
}


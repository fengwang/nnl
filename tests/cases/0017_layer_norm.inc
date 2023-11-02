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

TEST_CASE( "Node-layer_norm-17", "[node-layer_norm-17]" )
{
    using namespace nnl;

    {
        std::int64_t bs = 16;
        std::int64_t dim = 8;
        std::uint64_t ubs = static_cast<std::uint64_t>(bs);
        std::uint64_t udim = static_cast<std::uint64_t>(dim);

        // initialize nodes
        auto inp = make_node( "Input", "inp" );
        auto l1 = make_node( "LayerNorm", "l1" );
        l1.set_output_node();
        auto inp_t = random<default_engine_type>( {bs, dim}, -1.0f, 1.0f, "float32", "inp_t_17" );
        {
            auto mat = view_2d{ reinterpret_cast<float*>( inp_t.data() ), ubs, udim } ;
            std::cout << "LayerNorm input:\n";
            for ( auto jdx : range(bs) )
            {
                for ( auto idx : range(dim) )
                {
                    std::cout << mat[jdx][idx] << "\t";
                }
                std::cout << "\n";
            }
        }
        auto alpha = random<default_engine_type>( {dim,}, -1.0f, 1.0f, "float32", "alpha_17" );
        auto beta = random<default_engine_type>( {dim,}, -1.0f, 1.0f, "float32", "beta_17" );

        auto gt_t = make_tensor<default_engine_type>( {bs, dim}, "float32", "gt_t_17" );
        {
            std::cout << "Expected LayerNorm output:\n";
            auto mat = view_2d{ reinterpret_cast<float*>( inp_t.data() ), ubs, udim } ;
            auto nat = view_2d{ reinterpret_cast<float*>( gt_t.data() ), ubs, udim } ;
            auto _alpha = view_2d{ reinterpret_cast<float*>( alpha.data() ), 1, udim } ;
            auto _beta = view_2d{ reinterpret_cast<float*>( beta.data() ), 1, udim } ;
            for ( auto jdx : range(bs) )
            {
                float mean = 0.0f;
                for ( auto idx : range(dim) )
                {
                    mean += mat[jdx][idx] / dim;
                }

                float var = 0.0f;
                for ( auto idx : range(dim) )
                {
                    var += std::pow(mat[jdx][idx]-mean, 2) / dim;
                }

                for ( auto idx : range(dim) )
                {
                    nat[jdx][idx] = (mat[jdx][idx] - mean) / std::sqrt( var+1.0e-10 );
                    nat[jdx][idx] = nat[jdx][idx] * _alpha[0][idx] + _beta[0][idx];
                    std::cout << nat[jdx][idx] << "\t";
                }
                std::cout << "\n";
            }
        }

        auto l1_o = make_tensor<default_engine_type>( {bs, dim}, "float32", "l1_o_17" );
        auto l1_buffer = make_tensor<default_engine_type>( {bs, 2}, "float32", "l1_buffer_17" );
        {
            inp.add_input( inp_t );
        }
        {
            l1.add_input( inp_t );
            l1.add_output( l1_o );
            l1.add_buffer( l1_buffer );
            l1.add_weight( alpha );
            l1.add_weight( beta );
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
            sess.get_device_memory_manager().synchronize();
            auto mat = view_2d{ reinterpret_cast<float*>( l1_o.data() ), ubs, udim } ;
            auto nat = view_2d{ reinterpret_cast<float*>( gt_t.data() ), ubs, udim } ;
            std::cout << "Output:\n";
            for ( auto jdx : range(bs) )
            {
                for ( auto idx : range(dim) )
                {
                    std::cout << mat[jdx][idx] << " -- " << nat[jdx][idx] << "\t";
                }
                std::cout << "\n";
            }
            for ( auto jdx : range(bs) )
            {
                for ( auto idx : range(dim) )
                {
                    CHECK( (std::abs(mat[jdx][idx]-nat[jdx][idx]) < 1.0e-3f) );
                }
            }
        }

        std::cout << "After inference, the session is\n" << sess << std::endl;
    }

    REQUIRE( true );
}


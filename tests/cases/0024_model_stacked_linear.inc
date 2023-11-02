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
#include "../../include/direct_space/model.hpp"

TEST_CASE( "Node-stacked-linear-24", "[node-stacked-linear-24]" )
{
    spdlog::info( "\nTest-Case of Model-Stacked-Linear-24\n" );

    using namespace nnl;
    auto& sess = get_default_session<default_engine_type>();
    sess.clean();

    {
        std::int64_t rows = 128;
        std::uint64_t urows = static_cast<std::uint64_t>(rows);
        std::int64_t cols = 16*1024;
        std::uint64_t ucols = static_cast<std::uint64_t>(cols);
        float cst_2 = 0.0f;

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

        auto l_w2 = make_tensor<default_engine_type>( {cols, cols}, "float32", "l_w2" );
        {
            auto mat = view_2d{ reinterpret_cast<float*>( l_w2.data() ), ucols, ucols };
            for ( auto r : range(cols) )
                for ( auto c : range(cols) )
                    mat[r][c] = (r==c) ? 1.0f : 0.0f;
        }
        auto l_b2 = make_tensor<default_engine_type>( {cols,}, "float32", "l_b2" );
        {
            auto mat = view_1d{ reinterpret_cast<float*>( l_b2.data() ), ucols } ;
            for ( auto c : range(cols) )
                mat[c] = -c * 2.0f + cst_2;
        }

        spdlog::info( "All weights prepared." );

        layer inp = Input( "inp" );
        spdlog::info( format("inp layer: {}", inp) );
        layer l0 = Dense( l_w, l_b, "linear_0" )( inp );
        spdlog::info( format("l0 layer: {}", l0) );
        layer l1 = Dense( l_w2, l_b2, "linear_1" )( l0 );
        spdlog::info( format("l1 layer: {}", l1) );

        spdlog::info( "All layers prepared." );

        model m{ inp, l1 };
        spdlog::info( "Model is ready." );

        auto pred = m.predict( {inp_t, } );
        spdlog::info( "Prediction is done." );

        auto l1_o = pred[0];
        spdlog::info( "Verifying results." );


        {
            auto mat = view_2d{ reinterpret_cast<float*>( l1_o.data() ), urows, ucols } ;
            for ( auto c : range(cols) )
            {
                for ( auto r : range(rows) )
                {
                    if ( ! (std::abs(mat[r][c]-cst_2+c) <= 1.0e-2f) )
                    {
                        spdlog::error( format( "mat[{}][{}] = {}, but {} is expected.", r, c, mat[r][c], -c+cst_2 ) );
                    }
                    REQUIRE( std::abs(mat[r][c]-cst_2+c) <= 1.0e-2f ); // stop at the first error
                }
            }
        }

        std::cout << "After inference, the session is\n" << sess << std::endl;
    }

    REQUIRE( true );
}


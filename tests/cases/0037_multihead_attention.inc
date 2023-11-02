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

TEST_CASE( "multihead-attention-model-37", "[multihead-attention-model-37]" )
{

    spdlog::info( "\nTest case of 37 started.\n" );

    std::this_thread::sleep_for( std::chrono::seconds( 1 ) );

    using namespace nnl;
    auto& sess = get_default_session<default_engine_type>();
    sess.clean();


    // define all tensors

    std::int64_t const n_seq=10;
    std::int64_t const n_embd=768;
    std::int64_t const n_heads=12;
    std::int64_t const n_dim= n_embd / n_heads;
    auto t_input = make_tensor<default_engine_type>( {n_seq, n_embd,}, "float32", "input" );
    //tests/testdata/0037/b_p.bin
    t_input.load_memory( "./tests/testdata/0037/d_input.bin" );
    auto t_gt = make_tensor<default_engine_type>( {n_seq, n_embd,}, "float32", "gt" );
    //t_gt.import_from( d_gt );
    t_gt.load_memory( "./tests/testdata/0037/d_gt.bin" );
    auto t_b_p = make_tensor<default_engine_type>( {n_embd,}, "float32", "b_p" );
    //t_b_p.import_from( b_p );
    t_b_p.load_memory( "./tests/testdata/0037/b_p.bin" );

    std::vector<tensor<default_engine_type>> w_qs;
    {
        for ( [[maybe_unused]] auto idx : range(n_heads) )
            w_qs.emplace_back( make_tensor<default_engine_type>( { n_embd, n_dim, }, "float32" ) );
        w_qs[0].load_memory( "./tests/testdata/0037/w_qs_0.bin" );
        w_qs[1].load_memory( "./tests/testdata/0037/w_qs_1.bin" );
        w_qs[2].load_memory( "./tests/testdata/0037/w_qs_2.bin" );
        w_qs[3].load_memory( "./tests/testdata/0037/w_qs_3.bin" );
        w_qs[4].load_memory( "./tests/testdata/0037/w_qs_4.bin" );
        w_qs[5].load_memory( "./tests/testdata/0037/w_qs_5.bin" );
        w_qs[6].load_memory( "./tests/testdata/0037/w_qs_6.bin" );
        w_qs[7].load_memory( "./tests/testdata/0037/w_qs_7.bin" );
        w_qs[8].load_memory( "./tests/testdata/0037/w_qs_8.bin" );
        w_qs[9].load_memory( "./tests/testdata/0037/w_qs_9.bin" );
        w_qs[10].load_memory( "./tests/testdata/0037/w_qs_10.bin" );
        w_qs[11].load_memory( "./tests/testdata/0037/w_qs_11.bin" );
    }
    std::vector<tensor<default_engine_type>> b_qs;
    {
        for ( [[maybe_unused]] auto idx : range(n_heads) )
            b_qs.emplace_back( make_tensor<default_engine_type>( { n_dim, }, "float32" ) );
        b_qs[0].load_memory( "./tests/testdata/0037/b_qs_0.bin" );
        b_qs[1].load_memory( "./tests/testdata/0037/b_qs_1.bin" );
        b_qs[2].load_memory( "./tests/testdata/0037/b_qs_2.bin" );
        b_qs[3].load_memory( "./tests/testdata/0037/b_qs_3.bin" );
        b_qs[4].load_memory( "./tests/testdata/0037/b_qs_4.bin" );
        b_qs[5].load_memory( "./tests/testdata/0037/b_qs_5.bin" );
        b_qs[6].load_memory( "./tests/testdata/0037/b_qs_6.bin" );
        b_qs[7].load_memory( "./tests/testdata/0037/b_qs_7.bin" );
        b_qs[8].load_memory( "./tests/testdata/0037/b_qs_8.bin" );
        b_qs[9].load_memory( "./tests/testdata/0037/b_qs_9.bin" );
        b_qs[10].load_memory( "./tests/testdata/0037/b_qs_10.bin" );
        b_qs[11].load_memory( "./tests/testdata/0037/b_qs_11.bin" );
    }

    std::vector<tensor<default_engine_type>> w_ks;
    {
        for ( [[maybe_unused]] auto idx : range(n_heads) )
            w_ks.emplace_back( make_tensor<default_engine_type>( { n_embd, n_dim, }, "float32" ) );
        w_ks[0].load_memory( "./tests/testdata/0037/w_ks_0.bin" );
        w_ks[1].load_memory( "./tests/testdata/0037/w_ks_1.bin" );
        w_ks[2].load_memory( "./tests/testdata/0037/w_ks_2.bin" );
        w_ks[3].load_memory( "./tests/testdata/0037/w_ks_3.bin" );
        w_ks[4].load_memory( "./tests/testdata/0037/w_ks_4.bin" );
        w_ks[5].load_memory( "./tests/testdata/0037/w_ks_5.bin" );
        w_ks[6].load_memory( "./tests/testdata/0037/w_ks_6.bin" );
        w_ks[7].load_memory( "./tests/testdata/0037/w_ks_7.bin" );
        w_ks[8].load_memory( "./tests/testdata/0037/w_ks_8.bin" );
        w_ks[9].load_memory( "./tests/testdata/0037/w_ks_9.bin" );
        w_ks[10].load_memory( "./tests/testdata/0037/w_ks_10.bin" );
        w_ks[11].load_memory( "./tests/testdata/0037/w_ks_11.bin" );
    }
    std::vector<tensor<default_engine_type>> b_ks;
    {
        for ( [[maybe_unused]] auto idx : range(n_heads) )
            b_ks.emplace_back( make_tensor<default_engine_type>( { n_dim, }, "float32" ) );
        b_ks[0].load_memory( "./tests/testdata/0037/b_ks_0.bin" );
        b_ks[1].load_memory( "./tests/testdata/0037/b_ks_1.bin" );
        b_ks[2].load_memory( "./tests/testdata/0037/b_ks_2.bin" );
        b_ks[3].load_memory( "./tests/testdata/0037/b_ks_3.bin" );
        b_ks[4].load_memory( "./tests/testdata/0037/b_ks_4.bin" );
        b_ks[5].load_memory( "./tests/testdata/0037/b_ks_5.bin" );
        b_ks[6].load_memory( "./tests/testdata/0037/b_ks_6.bin" );
        b_ks[7].load_memory( "./tests/testdata/0037/b_ks_7.bin" );
        b_ks[8].load_memory( "./tests/testdata/0037/b_ks_8.bin" );
        b_ks[9].load_memory( "./tests/testdata/0037/b_ks_9.bin" );
        b_ks[10].load_memory( "./tests/testdata/0037/b_ks_10.bin" );
        b_ks[11].load_memory( "./tests/testdata/0037/b_ks_11.bin" );
    }

    std::vector<tensor<default_engine_type>> w_vs;
    {
        for ( [[maybe_unused]] auto idx : range(n_heads) )
            w_vs.emplace_back( make_tensor<default_engine_type>( { n_embd, n_dim, }, "float32" ) );
        w_vs[0].load_memory( "./tests/testdata/0037/w_vs_0.bin" );
        w_vs[1].load_memory( "./tests/testdata/0037/w_vs_1.bin" );
        w_vs[2].load_memory( "./tests/testdata/0037/w_vs_2.bin" );
        w_vs[3].load_memory( "./tests/testdata/0037/w_vs_3.bin" );
        w_vs[4].load_memory( "./tests/testdata/0037/w_vs_4.bin" );
        w_vs[5].load_memory( "./tests/testdata/0037/w_vs_5.bin" );
        w_vs[6].load_memory( "./tests/testdata/0037/w_vs_6.bin" );
        w_vs[7].load_memory( "./tests/testdata/0037/w_vs_7.bin" );
        w_vs[8].load_memory( "./tests/testdata/0037/w_vs_8.bin" );
        w_vs[9].load_memory( "./tests/testdata/0037/w_vs_9.bin" );
        w_vs[10].load_memory( "./tests/testdata/0037/w_vs_10.bin" );
        w_vs[11].load_memory( "./tests/testdata/0037/w_vs_11.bin" );
    }
    std::vector<tensor<default_engine_type>> b_vs;
    {
        for ( [[maybe_unused]] auto idx : range(n_heads) )
            b_vs.emplace_back( make_tensor<default_engine_type>( { n_dim, }, "float32" ) );
        b_vs[0].load_memory( "./tests/testdata/0037/b_vs_0.bin" );
        b_vs[1].load_memory( "./tests/testdata/0037/b_vs_1.bin" );
        b_vs[2].load_memory( "./tests/testdata/0037/b_vs_2.bin" );
        b_vs[3].load_memory( "./tests/testdata/0037/b_vs_3.bin" );
        b_vs[4].load_memory( "./tests/testdata/0037/b_vs_4.bin" );
        b_vs[5].load_memory( "./tests/testdata/0037/b_vs_5.bin" );
        b_vs[6].load_memory( "./tests/testdata/0037/b_vs_6.bin" );
        b_vs[7].load_memory( "./tests/testdata/0037/b_vs_7.bin" );
        b_vs[8].load_memory( "./tests/testdata/0037/b_vs_8.bin" );
        b_vs[9].load_memory( "./tests/testdata/0037/b_vs_9.bin" );
        b_vs[10].load_memory( "./tests/testdata/0037/b_vs_10.bin" );
        b_vs[11].load_memory( "./tests/testdata/0037/b_vs_11.bin" );
    }

    std::vector<tensor<default_engine_type>> w_ps;
    {
        for ( [[maybe_unused]] auto idx : range(n_heads) )
            w_ps.emplace_back( make_tensor<default_engine_type>( { n_dim, n_embd, }, "float32" ) );
        w_ps[0].load_memory( "./tests/testdata/0037/w_ps_0.bin" );
        w_ps[1].load_memory( "./tests/testdata/0037/w_ps_1.bin" );
        w_ps[2].load_memory( "./tests/testdata/0037/w_ps_2.bin" );
        w_ps[3].load_memory( "./tests/testdata/0037/w_ps_3.bin" );
        w_ps[4].load_memory( "./tests/testdata/0037/w_ps_4.bin" );
        w_ps[5].load_memory( "./tests/testdata/0037/w_ps_5.bin" );
        w_ps[6].load_memory( "./tests/testdata/0037/w_ps_6.bin" );
        w_ps[7].load_memory( "./tests/testdata/0037/w_ps_7.bin" );
        w_ps[8].load_memory( "./tests/testdata/0037/w_ps_8.bin" );
        w_ps[9].load_memory( "./tests/testdata/0037/w_ps_9.bin" );
        w_ps[10].load_memory( "./tests/testdata/0037/w_ps_10.bin" );
        w_ps[11].load_memory( "./tests/testdata/0037/w_ps_11.bin" );
    }


    auto input_layer = Input( "InputLayer" );
    auto output_layer = MultiHeadAttention( w_qs, b_qs, w_ks, b_ks, w_vs, b_vs, w_ps, t_b_p, "MultiHead_Attention_Layer_model_37" )( input_layer );
    auto m = model( input_layer, output_layer );
    auto outputs = m.predict( t_input );
    auto output = outputs[0];

    output.save_txt( "./output.txt" );
    t_gt.save_txt( "./gt.txt" );

    auto mat = view_2d{ reinterpret_cast<float*>( t_gt.data() ), static_cast<std::uint64_t>(n_seq), static_cast<std::uint64_t>(n_embd) };
    auto nat = view_2d{ reinterpret_cast<float*>( output.data() ), static_cast<std::uint64_t>(n_seq), static_cast<std::uint64_t>(n_embd) };
    for ( auto r : range( n_seq ) )
        for ( auto c : range( n_embd ) )
        {
            if ( std::abs(mat[r][c]-nat[r][c]) > 1.0e-2 )
            {
                spdlog::error( "gt[{}][{}]={}, pred[{}][{}]={}", r, c, mat[r][c], r, c, nat[r][c] );
            }
            REQUIRE( std::abs(mat[r][c]-nat[r][c]) < 1.0e-2 );
        }

}


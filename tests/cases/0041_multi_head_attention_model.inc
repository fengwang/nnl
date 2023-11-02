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

TEST_CASE( "Multi-head-attention-model-41", "[multi-head-attention-model-41]" )
{
    spdlog::info( "\nTest case of 41 started.\n" );

    std::this_thread::sleep_for( std::chrono::seconds( 1 ) );

    using namespace nnl;
    auto& sess = get_default_session<default_engine_type>();
    sess.clean();

    std::int64_t const n_seq = 10;
    std::int64_t const n_head = 25;
    std::int64_t const n_embd = 1600;

    auto input = make_tensor<default_engine_type>( {n_seq, n_embd}, "float32", "tensor_input" );
    input.load_memory( "./tests/testdata/0041/d_input.bin" );
    auto w_att = make_tensor<default_engine_type>( {n_embd, n_embd*3}, "float32", "tensor_w_att" );
    w_att.load_memory( "./tests/testdata/0041/d_attn_w.bin" );
    auto b_att = make_tensor<default_engine_type>( {n_embd*3,}, "float32", "tensor_b_att" );
    b_att.load_memory( "./tests/testdata/0041/d_attn_b.bin" );
    auto w_proj = make_tensor<default_engine_type>( {n_embd, n_embd}, "float32", "tensor_w_proj" );
    w_proj.load_memory( "./tests/testdata/0041/d_proj_w.bin" );
    auto b_proj = make_tensor<default_engine_type>( {n_embd,}, "float32", "tensor_b_proj" );
    b_proj.load_memory( "./tests/testdata/0041/d_proj_b.bin" );
    auto gt = make_tensor<default_engine_type>( {n_seq, n_embd}, "float32", "tensor_gt" );
    gt.load_memory( "./tests/testdata/0041/d_gt.bin" );


    auto input_layer = Input( "InputLayer" );
    auto output_layer = MultiHeadAttention( w_att, b_att, w_proj, b_proj, n_head, "Attention_Layer_model_41" )( input_layer );
    auto m = model( input_layer, output_layer );
    auto outputs = m.predict( input );
    auto output = outputs[0];

    output.save_txt( "./output.txt" );

    auto mat = view_2d{ reinterpret_cast<float*>( gt.data() ), static_cast<std::uint64_t>(n_seq), static_cast<std::uint64_t>(n_embd) };
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


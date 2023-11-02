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

extern "C" void cuda_device_synchronize();

TEST_CASE( "Multi-head-attention-model-44", "[multi-head-attention-model-44]" )
{
    spdlog::info( "\nTest case of 44 started.\n" );

    using namespace nnl;
    auto& sess = get_default_session<default_engine_type>();
    sess.clean();

    std::int64_t const n_seq = 127;
    std::int64_t const n_embd = 768;
    std::int64_t const n_head = 12;

    auto input = make_tensor<default_engine_type>( {n_seq, n_embd}, "float32", "input_tensor" );

    auto mlp_c_fc_w = make_tensor<default_engine_type>( {n_embd, n_embd}, "float32" );
    auto mlp_c_fc_b = make_tensor<default_engine_type>( {n_embd,}, "float32" );
    auto mlp_c_proj_w = make_tensor<default_engine_type>( {n_embd, n_embd}, "float32" );
    auto mlp_c_proj_b = make_tensor<default_engine_type>( {n_embd,}, "float32" );

    auto mha_att_w = make_tensor<default_engine_type>( {n_embd, n_embd*3}, "float32" );
    auto mha_att_b = make_tensor<default_engine_type>( {n_embd*3,}, "float32" );
    auto mha_proj_w = make_tensor<default_engine_type>( {n_embd, n_embd}, "float32" );
    auto mha_proj_b = make_tensor<default_engine_type>( {n_embd,}, "float32" );

    auto ln_1_alpha = make_tensor<default_engine_type>( {n_embd,}, "float32" );
    auto ln_1_beta = make_tensor<default_engine_type>( {n_embd,}, "float32" );
    auto ln_2_alpha = make_tensor<default_engine_type>( {n_embd,}, "float32" );
    auto ln_2_beta = make_tensor<default_engine_type>( {n_embd,}, "float32" );

    auto gt = make_tensor<default_engine_type>( {n_seq, n_embd,}, "float32" );


    //input.import_from( d_input );
    input.load_memory( "./tests/testdata/0044/d_input.bin" );
    mlp_c_fc_w.load_memory( "./tests/testdata/0044/d_mlp_c_fc_w.bin" );
    mlp_c_fc_b.load_memory( "./tests/testdata/0044/d_mlp_c_fc_b.bin" );
    mlp_c_proj_w.load_memory( "./tests/testdata/0044/d_mlp_c_proj_w.bin" );
    mlp_c_proj_b.load_memory( "./tests/testdata/0044/d_mlp_c_proj_b.bin" );

    mha_att_w.load_memory( "./tests/testdata/0044/d_mha_att_w.bin" );
    mha_att_b.load_memory( "./tests/testdata/0044/d_mha_att_b.bin" );
    mha_proj_w.load_memory( "./tests/testdata/0044/d_mha_proj_w.bin" );
    mha_proj_b.load_memory( "./tests/testdata/0044/d_mha_proj_b.bin" );

    ln_1_alpha.load_memory( "./tests/testdata/0044/d_ln_1_alpha.bin" );
    ln_1_beta.load_memory( "./tests/testdata/0044/d_ln_1_beta.bin" );
    ln_2_alpha.load_memory( "./tests/testdata/0044/d_ln_2_alpha.bin" );
    ln_2_beta.load_memory( "./tests/testdata/0044/d_ln_2_beta.bin" );

    //gt.load_memory( "./tests/testdata/0044/stage_1.bin" );
    gt.load_memory( "./tests/testdata/0044/d_gt.bin" );

    auto input_layer = Input( "InputLayer" );
    auto output_layer = Transformer( mlp_c_fc_w, mlp_c_fc_b, mlp_c_proj_w, mlp_c_proj_b,
                                     mha_att_w, mha_att_b, mha_proj_w, mha_proj_b,
                                     ln_1_alpha, ln_1_beta,
                                     ln_2_alpha, ln_2_beta,
                                     n_head,
                                     "Transformer_44"
                        )( input_layer );
    auto m = model( input_layer, output_layer );
    auto outputs = m.predict( input );

    //cuda_device_synchronize();

    auto output = outputs[0];
    output.save_txt( "./output.txt" );

    //debug
    //input.synchronize_to_host();
    //input.save_txt( "./debug_input.txt" );

    #if 1
    auto mat = view_2d{ reinterpret_cast<float*>( gt.data() ), static_cast<std::uint64_t>(n_seq), static_cast<std::uint64_t>(n_embd) };
    auto nat = view_2d{ reinterpret_cast<float*>( output.data() ), static_cast<std::uint64_t>(n_seq), static_cast<std::uint64_t>(n_embd) };
    for ( auto r : range( n_seq ) )
    {
        for ( auto c : range( n_embd ) )
        {
            float const md = std::abs( nat[r][c] ) + 1.0e-10;
            CHECK( ((std::abs(mat[r][c]-nat[r][c])/md < 1.0e-1) || (std::abs(mat[r][c]-nat[r][c]) < 1.0e-1))  );
            if ( std::abs(mat[r][c]-nat[r][c])/md > 1.0e-1 )
            {
                spdlog::error( "gt[{}][{}]={}, pred[{}][{}]={}", r, c, mat[r][c], r, c, nat[r][c] );
                break;
            }
        }
    }
    #endif
}


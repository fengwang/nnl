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

TEST_CASE( "Vocabulary-projection-46", "[vocabulary-projection]" )
{
    spdlog::info( "\nTest case of 46 started.\n" );

    using namespace nnl;
    auto& sess = get_default_session<default_engine_type>();
    sess.clean();

    std::int64_t const n_seq = 4;
    std::int64_t const n_embd = 1600;
    std::int64_t const n_vocab = 50257;

    auto ans = make_tensor_from_memory<default_engine_type>( {n_embd,}, "float32",
                                                             "./tests/testdata/0046/ans.bin", "ans_tensor" );
    auto input = make_tensor_from_memory<default_engine_type>( {n_seq, n_embd,}, "float32",
                                                                 "./tests/testdata/0046/vocabulary_projection_input.bin", "input_tensor" );
    auto ln_t = make_tensor_from_memory<default_engine_type>( {1, n_embd,}, "float32",
                                                              "./tests/testdata/0046/ln.bin", "ln_tensor" );
    auto gt = make_tensor_from_memory<default_engine_type>( {n_vocab,}, "float32",
                                                  "./tests/testdata/0046/vocabulary_projection_output.bin", "output_tensor" );
    auto wte = make_tensor_from_memory<default_engine_type>( {n_vocab, n_embd,}, "float32",
                                                             "examples/gpt2-1558M/assets/wte.bin", "wte_tensor" );
    auto ln_f_b = make_tensor_from_memory<default_engine_type>( {n_embd,}, "float32",
                                                                "examples/gpt2-1558M/assets/ln_f.b.bin", "ln_f.b_tensor" );
    auto ln_f_g = make_tensor_from_memory<default_engine_type>( {n_embd,}, "float32",
                                                                "examples/gpt2-1558M/assets/ln_f.g.bin", "ln_f.g_tensor" );
    auto input_layer = Input( "InputLayer" );
    auto output_layer = VocabularyProjection( ln_f_g, ln_f_b, wte, "VocabularyProjection" )(input_layer);
    auto m = model( input_layer, output_layer );
    auto outputs = m.predict( input );


    auto output = outputs[0];
    output.save_txt( "./output.txt" );


    #if 1
    auto mat = view_2d{ reinterpret_cast<float*>( ans.data() ), 1, static_cast<std::uint64_t>(n_embd) };
    auto nat = view_2d{ reinterpret_cast<float*>( output.data() ), 1, static_cast<std::uint64_t>(n_embd) };
    for ( auto r : range( 1 ) )
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

    {
        auto argmax = outputs[1];
        auto argmax_vec = reinterpret_cast<int*>( argmax.data() );
        CHECK( 45351 == argmax_vec[0] );
    }



}


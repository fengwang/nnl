#include "../../include/direct_space/model.hpp"
#include "./gpt2_tokenizer.hpp"
#include "./lyra.hpp"

using namespace nnl;

// gpt-2 layer
#if 0
{
    "attn": {
        "c_attn": {"b": [3*n_embd], "w": [n_embd, 3*n_embd]},
        "c_proj": {"b": [n_embd], "w": [n_embd, n_embd]},
    },
    "ln_1": {"b": [n_embd], "g": [n_embd]},
    "ln_2": {"b": [n_embd], "g": [n_embd]},
    "mlp": {
        "c_fc": {"b": [4*n_embd], "w": [n_embd, 4*n_embd]},
        "c_proj": {"b": [n_embd], "w": [4*n_embd, n_embd]},
    },
},
#endif

struct gpt2_1558m
{
    std::int64_t heads_;
    std::int64_t blocks_;
    std::int64_t n_embd_;
    std::int64_t n_vocab_;
    std::int64_t n_token_;

    std::vector<std::vector<tensor<default_engine_type>>> weight_blocks_;
    tensor<default_engine_type> wte_; // [n_vocab_, n_embd_,]
    tensor<default_engine_type> wpe_;
    tensor<default_engine_type> ln_f_b_;
    tensor<default_engine_type> ln_f_g_;
    gpt2_tokenizer tokenizer_;


    std::string predict( std::string const& input, std::int64_t max_len )
    {
        //convert input to token
        std::vector<std::int64_t> encoding = tokenizer_.encode( input );
        spdlog::info( nnl::format( "encoding is [{}]", encoding ) );
        //better_assert( false, "stop here to debug" );
        std::int64_t n_input = encoding.size();
        //max_len = std::max( max_len + n_input - static_cast<std::int64_t>(input.size()), n_input ); // adjust max_len
        //max_len = std::max( max_len, n_input ); // adjust max_len
        max_len += n_input;
        max_len = std::min( n_token_, max_len );
        spdlog::info( "Making prediction with input __{}__ and max_len {}", input, max_len );
        spdlog::info( nnl::format( "The initial encoding is: {}", encoding ) );

        // prepare input token
        auto input_tensor = make_tensor<default_engine_type>({n_input, n_embd_,}, "float32", "InputToken");
        //auto input_tensor = make_tensor<default_engine_type>({max_len, n_embd_,}, "float32", "InputToken");
        spdlog::info( nnl::format( "Input tensor generated: {}", input_tensor ) );
        {
            // reserve memory for later use
            input_tensor.reserve( make_vector( max_len, n_embd_) );
            //input_tensor.reshape( make_vector( n_input, n_embd_ ) );
            auto wte_m = view_2d{ reinterpret_cast<float*>(wte_.data()), n_vocab_, n_embd_ };
            auto input_m = view_2d{ reinterpret_cast<float*>(input_tensor.data()), n_input, n_embd_ };
            auto wpe_m = view_2d{ reinterpret_cast<float*>(wpe_.data()), n_token_, n_embd_ };
            // x = wte[inputs] + wpe[range(len(inputs))]
            for ( auto idx : range( n_input ) )
            {
                for ( auto jdx : range( n_embd_ ) )
                {
                    input_m[idx][jdx] = wte_m[encoding[idx]][jdx] + wpe_m[idx][jdx];
                }
            }

            /*
            for ( auto idx : range( n_input ) )
            {
                std::copy( wte_m[encoding[idx]], wte_m[encoding[idx]]+n_embd_, input_m[idx] );
                spdlog::info( "Copying {} float number from wte {} to input {}. The first value is <{}:{}>", n_embd_, encoding[idx], idx, wte_m[encoding[idx]][0], input_m[idx][0] );
            }
            */
            spdlog::info( "Input token prepared." );
            spdlog::info( nnl::format( "Input tensor updated: {}", input_tensor ) );
        }

        if constexpr ( debug_mode )
        {
            input_tensor.save_txt( "./input.txt" );
            spdlog::info( "Initial tensor is saved to ./input.txt." );
        }



        // iteratively predict
        auto&& gpt2_model = create_model();
        {
            for ( [[maybe_unused]] auto idx : range( max_len - n_input ) )
            {
                if ( idx < 0 ) break; // <-- no negative indices, just in case

                spdlog::info( "Making prediction at index {}.", idx );
                auto outputs = gpt2_model.predict( input_tensor ); // two outputs generated [ vector and max arg ]
                // copy the max idx to encoding
                encoding.push_back( *(reinterpret_cast<int*>(outputs[1].data())) );
                ++n_input;
                // enlarge input tensor 1st dimension by 1
                // Note: no memory reallocation will be executed here, as we have reserved memory
                input_tensor.reshape( make_vector( n_input, n_embd_ ) );
                // copy the embeding for the predicted token to the last row of the input tensor
                #if 0
                std::copy_n( outputs[0].data(), n_embd_*sizeof(float), input_tensor.data()+(encoding.size()-1)*n_embd_*sizeof(float) );
                #else
                {
                    auto output_m = view_2d{ reinterpret_cast<float*>(outputs[0].data()), 1, n_embd_ };
                    auto input_m = view_2d{ reinterpret_cast<float*>(input_tensor.data()), n_input, n_embd_ };
                    auto wpe_m = view_2d{ reinterpret_cast<float*>(wpe_.data()), n_token_, n_embd_ };
                    for ( auto jdx : range( n_embd_ ) )
                    {
                        input_m[n_input-1][jdx] = output_m[0][jdx] + wpe_m[n_input-1][jdx];
                    }
                }
                #endif
                spdlog::info( "Prediction at index {} has been finished. New encoding: {}", idx, encoding.back() );
                spdlog::info( nnl::format("The input tensor has been updated to {}", input_tensor) );

                if constexpr (debug_mode)
                {
                    input_tensor.save_txt( nnl::format("./input_{}.txt", idx) );
                    spdlog::info( nnl::format("Initial tensor is saved to ./input_{}.txt.", idx) );
                }
            }
        }

        std::cout << "Encoding:\n";
        for ( auto e : encoding )
        {
            std::cout << e << ", ";
        }
        std::cout << std::endl;

        return tokenizer_.decode( encoding );
    }

    model create_model()
    {
        layer input_layer = Input( "InputLayer" );
        //layer output_layer = input_layer;
        std::vector<layer> output_layers;
        output_layers.push_back( input_layer );

        for ( [[maybe_unused]] std::int64_t idx : range( blocks_ ) )
        {
            std::string const& layer_name = std::string{ "Block_" } + std::to_string( idx );
            spdlog::info( "create_model: preparing layer {} of {}/{}", layer_name, idx, blocks_ );
            auto output_layer = Transformer
            (
                    weight_blocks_[idx][0], // mlp_c_fc_w
                    weight_blocks_[idx][1], // mlp_c_fc_b
                    weight_blocks_[idx][2], // mlp_c_proj_w
                    weight_blocks_[idx][3], // mlp_c_proj_b
                    weight_blocks_[idx][4], // mha_att_w
                    weight_blocks_[idx][5], // mha_att_b
                    weight_blocks_[idx][6], // mha_proj_w
                    weight_blocks_[idx][7], // mha_proj_b
                    weight_blocks_[idx][8], // ln_1_alpha
                    weight_blocks_[idx][9], // ln_1_beta
                    weight_blocks_[idx][10], // ln_2_alpha
                    weight_blocks_[idx][11], // ln_2_beta
                    heads_,
                    layer_name
            )( output_layers.back() );
            output_layers.push_back( output_layer );
            spdlog::info( "create_model: prepared layer {} of {}/{}", layer_name, idx, blocks_ );
        }
        spdlog::info( "create_model: All transformer layers have been built." );

        //auto output_layer = VocabularyProjection( ln_f_b_, ln_f_g_, wte_, "Projection" )( output_layers.back() );
        auto output_layer = VocabularyProjection( ln_f_g_, ln_f_b_, wte_, "Projection" )( output_layers.back() );
        spdlog::info( "GPT2-1558M model has been built." );
        return model{ input_layer, output_layer };
    }



    gpt2_1558m( std::string const& model_folder ) :
        heads_{ 25 },
        blocks_{ 48 },
        n_embd_{ 1600 },
        n_vocab_{ 50257 },
        n_token_{ 1024 },
        wte_{ make_tensor<default_engine_type>({n_vocab_, n_embd_,}, "float32", "wte") },
        wpe_{ make_tensor<default_engine_type>({n_token_, n_embd_,}, "float32", "wpe") },
        ln_f_b_{ make_tensor<default_engine_type>({n_embd_,}, "float32", "ln_f_b") },
        ln_f_g_{ make_tensor<default_engine_type>({n_embd_,}, "float32", "ln_f_g") }
    {

        std::string const& sep = std::string{"/"};
        weight_blocks_.resize( blocks_ );
        {
            for ( auto block_index : range( blocks_ ) )
            {
                std::string const& sidx = std::to_string( block_index);
                { // mlp
                    weight_blocks_[block_index].emplace_back( make_tensor<default_engine_type>({n_embd_, n_embd_*4}, "float32", "mlp.c_fc.w_"+sidx) );
                    weight_blocks_[block_index].back().load_memory(model_folder+sep+sidx+std::string{".mlp.c_fc.w.bin"} );
                    weight_blocks_[block_index].emplace_back( make_tensor<default_engine_type>({n_embd_*4,}, "float32", "mlp.c_fc.b_"+sidx) );
                    weight_blocks_[block_index].back().load_memory(model_folder+sep+sidx+std::string{".mlp.c_fc.b.bin"} );

                    weight_blocks_[block_index].emplace_back( make_tensor<default_engine_type>({n_embd_*4, n_embd_}, "float32", "mlp.c_proj.w_"+sidx) );
                    weight_blocks_[block_index].back().load_memory(model_folder+sep+sidx+std::string{".mlp.c_proj.w.bin"} );
                    weight_blocks_[block_index].emplace_back( make_tensor<default_engine_type>({n_embd_,}, "float32", "mlp.c_proj.b_"+sidx) );
                    weight_blocks_[block_index].back().load_memory(model_folder+sep+sidx+std::string{".mlp.c_proj.b.bin"} );
                }
                { // attn
                    weight_blocks_[block_index].emplace_back( make_tensor<default_engine_type>({n_embd_, n_embd_*3}, "float32", "attn.c_attn.w_"+sidx) );
                    weight_blocks_[block_index].back().load_memory(model_folder+sep+sidx+std::string{".attn.c_attn.w.bin"} );
                    weight_blocks_[block_index].emplace_back( make_tensor<default_engine_type>({n_embd_*3,}, "float32", "attn.c_attn.b_"+sidx) );
                    weight_blocks_[block_index].back().load_memory(model_folder+sep+sidx+std::string{".attn.c_attn.b.bin"} );

                    weight_blocks_[block_index].emplace_back( make_tensor<default_engine_type>({n_embd_, n_embd_}, "float32", "attn.c_proj.w_"+sidx) );
                    weight_blocks_[block_index].back().load_memory(model_folder+sep+sidx+std::string{".attn.c_proj.w.bin"} );
                    weight_blocks_[block_index].emplace_back( make_tensor<default_engine_type>({n_embd_,}, "float32", "attn.c_proj.b_"+sidx) );
                    weight_blocks_[block_index].back().load_memory(model_folder+sep+sidx+std::string{".attn.c_proj.b.bin"} );
                }
                { // ln_1
                    weight_blocks_[block_index].emplace_back( make_tensor<default_engine_type>({n_embd_,}, "float32", "ln_1.g_"+sidx) );
                    weight_blocks_[block_index].back().load_memory(model_folder+sep+sidx+std::string{".ln_1.g.bin"} );
                    weight_blocks_[block_index].emplace_back( make_tensor<default_engine_type>({n_embd_,}, "float32", "ln_1.b_"+sidx) );
                    weight_blocks_[block_index].back().load_memory(model_folder+sep+sidx+std::string{".ln_1.b.bin"} );
                }
                { // ln_2
                    weight_blocks_[block_index].emplace_back( make_tensor<default_engine_type>({n_embd_,}, "float32", "ln_2.g_"+sidx) );
                    weight_blocks_[block_index].back().load_memory(model_folder+sep+sidx+std::string{".ln_2.g.bin"} );
                    weight_blocks_[block_index].emplace_back( make_tensor<default_engine_type>({n_embd_,}, "float32", "ln_2.b_"+sidx) );
                    weight_blocks_[block_index].back().load_memory(model_folder+sep+sidx+std::string{".ln_2.b.bin"} );
                }
            }
        }

        wte_.load_memory( model_folder+sep+std::string{"wte.bin"} );
        wpe_.load_memory( model_folder+sep+std::string{"wpe.bin"} );

        ln_f_b_.load_memory( model_folder+sep+std::string{"ln_f.b.bin"} );
        ln_f_g_.load_memory( model_folder+sep+std::string{"ln_f.g.bin"} );

        auto _tokenizer = make_gpt2_tokenizer( model_folder+sep+std::string{"vocab.json"}, model_folder+sep+std::string{"merges.txt"} );
        assert( _tokenizer.has_value() );
        tokenizer_= _tokenizer.value();
    }

}; // gpt2_1558m



// Ex: gpt2_1558m --max_len 40 "This is a prompt example."
int main( int argc, char const** argv)
{
    std::string prompt = "For decades holidaymakers have poured into resorts and islands in southern Europe for a relaxing break in the summer sun.\nBut the scenes of tourists fleeing wildfires in Greece, or trapped indoors unable to enjoy baking beaches in Spain, may give some people second thoughts.\nBack-to-back heatwaves brought sweltering temperatures in the 40s to parts of Europe in July, which is expected to break records for the world's hottest month ever recorded.";
    std::int64_t max_len = 40;
    if ( argc > 2 )
    {
        auto cli = lyra::opt( max_len, "-max_len" )["--max_len"] | lyra::arg( prompt, "prompt" );
        if ( cli.parse( {argc, argv} ) )
        {
            std::cout << "GPT2-1558M start with\n";
            std::cout << "max_len: " << max_len << std::endl;
            std::cout << "prompt: " << prompt << std::endl;
        }
    }

    spdlog::set_level(spdlog::level::err);
    gpt2_1558m gpt2_application{ "./examples/gpt2-1558M/assets" };
    std::string const& prediction = gpt2_application.predict( prompt, max_len );

    std::cout << "Prediction:\n" << prediction << std::endl;

    return 0;
}


#include "../../../include/direct_space/node.hpp"

#include "../cublas.hpp"
#include "../kernels.h"
#include "../cuda_stream.hpp"
#include "../cudnn.hpp"


namespace nnl::node_
{


void vocabulary_projection::implement_arithmetic_operation
(
    std::vector<std::byte*> const& d_w, std::vector<std::vector<std::int64_t>> const& w_s,
    std::vector<std::byte*> const& d_i, std::vector<std::vector<std::int64_t>> const& i_s,
    std::vector<std::byte*> const& d_o, std::vector<std::vector<std::int64_t>> const& o_s,
    [[maybe_unused]] std::byte* d_b, [[maybe_unused]] std::int64_t b_s,
    stream<default_engine_type>& sm
)
{
    better_assert( 3 == d_w.size() );
    better_assert( 3 == w_s.size() );
    better_assert( 1 == w_s[0].size() ); // layer_norm scaling, [n_embd,]
    better_assert( 1 == w_s[1].size() ); // layer_norm offset, [n_embd,]
    better_assert( 2 == w_s[2].size() ); // wte, [n_vocab, n_embd]
    better_assert( 1 == d_i.size() );
    better_assert( 1 == i_s.size() );
    better_assert( 2 == i_s[0].size() ); // input, [n_seq, n_embd]
    /*
    better_assert( 1 == d_o.size() );
    better_assert( 1 == o_s.size() );
    better_assert( 1 == o_s[0].size() ); // output, [n_embd,]
    */
    better_assert( 2 == d_o.size() );
    better_assert( 2 == o_s.size() );
    better_assert( 1 == o_s[0].size() ); // output, [n_embd,]
    better_assert( 1 == o_s[1].size() ); // output, [1,]

    auto const [n_seq, n_embd] = std::make_pair( i_s[0][0], i_s[0][1] );
    auto const [n_vocab, n_embd_] = std::make_pair( w_s[2][0], w_s[2][1] );
    better_assert( n_seq > 0 );
    better_assert( n_vocab > 0 );
    better_assert( n_embd == n_embd_ );
    better_assert( n_embd == w_s[0][0] ); // check layer norml scaling
    better_assert( n_embd == w_s[1][0] ); // check layer norml offset
    better_assert( n_embd == w_s[2][1] ); // check wte
    better_assert( n_embd == o_s[0][0] ); // check output shape
    better_assert( b_s >= static_cast<std::int64_t>(sizeof(float)*(2 + n_embd)) ); // check buffer size

    // apply layer norm
    float* x = reinterpret_cast<float*>(d_i[0]) + (n_seq-1) * n_embd; // only processing the last row
    std::int64_t const bfsize = std::max( n_embd, n_vocab );
    float* output = reinterpret_cast<float*>(d_o[0]);
    float* y = reinterpret_cast<float*>(d_b);
    float* y_ = y + bfsize;
    float* mean = y_ + bfsize;
    float* var = mean  + 1;
    int const rows = 1;
    int const cols = static_cast<int>(n_embd);

    cuda_layer_norm( x, y, rows, cols, 1.0e-5f, mean, var, sm.stream_->stream() );

    cuda_scaled_offset( y, y_, rows, cols, reinterpret_cast<float*>(d_w[0]), reinterpret_cast<float*>(d_w[1]), sm.stream_->stream() );

    float* wte = reinterpret_cast<float*>(d_w[2]);
    // gemv: output = y_ * wte^T, where y is of [1, n_embd], wte is of [n_vocab, n_embd]
    cuda_gemm( y_, false,
               wte, true,
               1, n_embd, n_vocab,
               //output,
               y, // <- [1, n_vocab]
               1.0f, 0.0f,
               sm );

    // TODO: softmax and then multinomial

    {
        // first extract argmax, storing it in result
        float* input = y;
        int n_input = static_cast<int>( n_vocab );
        float* cache = y_;
        int n_cache = static_cast<int>( bfsize );
        //int* result = reinterpret_cast<int*>(mean);
        int* result = reinterpret_cast<int*>(d_o[1]);
        cuda_argmax_1d( input, n_input, cache, n_cache, result, sm.stream_->stream() );

        // then copy wte[argmax(...)[-1]] to output
        int* max_prob_index = result;
        vocabulary_lookup( wte, n_vocab, n_embd, max_prob_index, output, sm.stream_->stream() );
    }
}// vocabulary_projection::implement_arithmetic_operation

} // namespace nnl::node_


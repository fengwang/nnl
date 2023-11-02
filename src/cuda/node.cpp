#include "../../include/direct_space/node.hpp"

#include "./cublas.hpp"
#include "./kernels.h"
#include "./cuda_stream.hpp"
#include "./cudnn.hpp"


namespace nnl::node_
{




void add::implement_arithmetic_operation
(
    std::vector<std::byte*> const& d_w, std::vector<std::vector<std::int64_t>> const& w_s,
    std::vector<std::byte*> const& d_i, std::vector<std::vector<std::int64_t>> const& i_s,
    std::vector<std::byte*> const& d_o, std::vector<std::vector<std::int64_t>> const& o_s,
    [[maybe_unused]] std::byte* d_b, [[maybe_unused]] std::int64_t b_s,
    stream<default_engine_type>& sm
)
{
    // size check
    better_assert( d_w.size() == 0, format("Expecting 0 weights, but got {}", d_w.size()) );
    better_assert( w_s.size() == 0, format("Expecting 0 weights shape, but got {}", w_s.size()) );
    better_assert( d_i.size() == 2, format("Expecting 2 input, but got {}", d_i.size()) );
    better_assert( i_s.size() == 2, format("Expecting 2 input shape, but got {}", i_s.size()) );
    better_assert( d_o.size() == 1, format("Expecting 1 output, but got {}", d_o.size()) );
    better_assert( o_s.size() == 1, format("Expecting 1 output shape, but got {}", o_s.size()) );


    //void cuda_add( float* c, float* a, int a_row, int a_col, float* b, int b_row, int b_col, cudaStream_t sm )

    float* a = reinterpret_cast<float*>( d_i[0] );
    float* b = reinterpret_cast<float*>( d_i[1] );
    float* c = reinterpret_cast<float*>( d_o[0] );

    std::vector<std::int64_t> const& a_shape = i_s[0];
    std::int64_t const a_col = *(a_shape.rbegin());
    std::int64_t const a_row = std::accumulate(a_shape.begin(), a_shape.end(), std::int64_t{1}, [](std::int64_t a, std::int64_t b){ return a*b; } ) / a_col;

    std::vector<std::int64_t> const& b_shape = i_s[1];
    std::int64_t const b_col = *(b_shape.rbegin());
    std::int64_t const b_row = std::accumulate(b_shape.begin(), b_shape.end(), std::int64_t{1}, [](std::int64_t a, std::int64_t b){ return a*b; } ) / b_col;

    spdlog::info( "calling add::implement_arithmetic_operation with a_row {}, a_col {}, b_row {} and b_col {}.", a_row, a_col, b_row, b_col );
    // c = a + b
    cuda_add( c, a, a_row, a_col, b, b_row, b_col, sm.stream_->stream() );
}



//
// This is the default implementation for fp32, other cases will be addressed in the future.
//
void gemm::implement_arithmetic_operation
(
    std::vector<std::byte*> const& d_w, std::vector<std::vector<std::int64_t>> const& w_s,
    std::vector<std::byte*> const& d_i, std::vector<std::vector<std::int64_t>> const& i_s,
    std::vector<std::byte*> const& d_o, std::vector<std::vector<std::int64_t>> const& o_s,
    [[maybe_unused]] std::byte* d_b, [[maybe_unused]] std::int64_t b_s,
    stream<default_engine_type>& sm
)
{
    // size check
    better_assert( d_w.size() == 1, format("Expecting 1 weights, but got {}", d_w.size()) );
    better_assert( w_s.size() == 1, format("Expecting 1 weights shape, but got {}", w_s.size()) );
    better_assert( d_i.size() == 1, format("Expecting 1 input, but got {}", d_i.size()) );
    better_assert( i_s.size() == 1, format("Expecting 1 input shape, but got {}", i_s.size()) );
    better_assert( d_o.size() == 1, format("Expecting 1 output, but got {}", d_o.size()) );
    better_assert( o_s.size() == 1, format("Expecting 1 output shape, but got {}", o_s.size()) );

    // d_imension check x W + b -> y
    better_assert( w_s[0].size() == 2 ); // W
    better_assert( i_s[0].size() == 2 ); // x
    better_assert( o_s[0].size() == 2 ); // y
    better_assert( w_s[0][0] == i_s[0][1] );
    better_assert( w_s[0][1] == o_s[0][1] );
    better_assert( i_s[0][0] == o_s[0][0] );

    float* x = reinterpret_cast<float*>(d_i[0]);
    float* W = reinterpret_cast<float*>(d_w[0]);
    float* y = reinterpret_cast<float*>(d_o[0]);

    // y = xw
    cuda_gemm( x, false, W, false, i_s[0][0], i_s[0][1], w_s[0][1], y, 1.0f, 0.0f, sm );
}


//
// Thi_s i_s the default implementation for fp32, other cases will be addressed in the future.
//
void linear::implement_arithmetic_operation
(
    std::vector<std::byte*> const& d_w, std::vector<std::vector<std::int64_t>> const& w_s,
    std::vector<std::byte*> const& d_i, std::vector<std::vector<std::int64_t>> const& i_s,
    std::vector<std::byte*> const& d_o, std::vector<std::vector<std::int64_t>> const& o_s,
    [[maybe_unused]] std::byte* d_b, [[maybe_unused]] std::int64_t b_s,
    stream<default_engine_type>& sm
)
{
    // size check
    better_assert( d_w.size() == 2, format("Expecting 2 weights, but got {}", d_w.size()) );
    better_assert( w_s.size() == 2, format("Expecting 2 weights shape, but got {}", w_s.size()) );
    better_assert( d_i.size() == 1, format("Expecting 1 input, but got {}", d_i.size()) );
    better_assert( i_s.size() == 1, format("Expecting 1 input shape, but got {}", i_s.size()) );
    better_assert( d_o.size() == 1, format("Expecting 1 output, but got {}", d_o.size()) );
    better_assert( o_s.size() == 1, format("Expecting 1 output shape, but got {}", o_s.size()) );

    // d_imension check x W + b -> y
    better_assert( w_s[0].size() == 2 ); // W
    better_assert( w_s[1].size() == 1 ); // b
    better_assert( i_s[0].size() == 2 ); // x
    better_assert( o_s[0].size() == 2 ); // y
    better_assert( w_s[0][0] == i_s[0][1] );
    better_assert( w_s[0][1] == w_s[1][0] );
    better_assert( w_s[0][1] == o_s[0][1] );
    better_assert( i_s[0][0] == o_s[0][0], format( "Expecting input and output share share the same batch size, but for input it is {}, for output it is {}", i_s[0][0], o_s[0][0] ) );


    float* x = reinterpret_cast<float*>(d_i[0]);
    float* W = reinterpret_cast<float*>(d_w[0]);
    [[maybe_unused]] float* b = reinterpret_cast<float*>(d_w[1]);
    float* y = reinterpret_cast<float*>(d_o[0]);

    //
    // y = b
    //
    add_bias( y, b, i_s[0][0], w_s[1][0], 0, sm.stream_->stream() );

    //
    // y = xw + y
    //
    cuda_gemm( x, false, W, false, i_s[0][0], i_s[0][1], w_s[0][1], y, 1.0f, 1.0f, sm );

}

void softmax::implement_arithmetic_operation
(
    std::vector<std::byte*> const& d_w, std::vector<std::vector<std::int64_t>> const& w_s,
    std::vector<std::byte*> const& d_i, std::vector<std::vector<std::int64_t>> const& i_s,
    std::vector<std::byte*> const& d_o, std::vector<std::vector<std::int64_t>> const& o_s,
    [[maybe_unused]] std::byte* d_b, [[maybe_unused]] std::int64_t b_s,
    stream<default_engine_type>& sm
)
{
    better_assert( d_w.size() == 0, format("Unexpected weight size {}, should be empty.", d_w.size()) );
    better_assert( w_s.size() == 0, format("Unexpected weight shape length {}, should be zero.", w_s.size()) );
    better_assert( d_i.size() == 1, format("Expect a single input, but got {}.", d_i.size()) );
    better_assert( i_s.size() == 1 && i_s[0].size()>=2, "Expect input has more than 2 dimensions." );
    better_assert( d_o.size() == 1, format("Expect a single output, but got {}.", d_o.size()) );
    better_assert( o_s.size() == 1 && i_s[0] == o_s[0], "Expect same shape with input and output tensors" );

    float* input = reinterpret_cast<float*>(d_i[0]);
    float* output = reinterpret_cast<float*>(d_o[0]);
    std::uint64_t batch_size = i_s[0][0];
    std::uint64_t dim_0 = i_s[0][1];
    std::uint64_t dim_1 = i_s[0].size() > 2 ? i_s[0][2] : 1;
    std::uint64_t dim_2 = i_s[0].size() > 3 ? i_s[0][3] : 1;
    bool channel_first = true;


    cudnn_softmax( input, output, batch_size, dim_0, dim_1, dim_2, sm, 1.0f, 0.0f, channel_first );

}//softmax::implement_arithmetic_operation

void gelu::implement_arithmetic_operation
(
    std::vector<std::byte*> const& d_w, std::vector<std::vector<std::int64_t>> const& w_s,
    std::vector<std::byte*> const& d_i, std::vector<std::vector<std::int64_t>> const& i_s,
    std::vector<std::byte*> const& d_o, std::vector<std::vector<std::int64_t>> const& o_s,
    [[maybe_unused]] std::byte* d_b, [[maybe_unused]] std::int64_t b_s,
    stream<default_engine_type>& sm
)
{
    better_assert( d_w.size() == 0, format("Unexpected weight size {}, should be empty.", d_w.size()) );
    better_assert( w_s.size() == 0, format("Unexpected weight shape length {}, should be zero.", w_s.size()) );
    better_assert( d_i.size() == 1, format("Expect a single input, but got {}.", d_i.size()) );
    better_assert( i_s.size() == 1 && i_s[0].size()>=2, "Expect input has more than 2 dimensions." );
    better_assert( d_o.size() == 1, format("Expect a single output, but got {}.", d_o.size()) );
    better_assert( o_s.size() == 1 && i_s[0] == o_s[0], "Expect same shape with input and output tensors" );

    float* input = reinterpret_cast<float*>(d_i[0]);
    float* output = reinterpret_cast<float*>(d_o[0]);
    std::uint64_t const n = std::accumulate( i_s[0].begin(), i_s[0].end(), 1LL, []( std::uint64_t a, std::uint64_t b ){ return a*b; } );

    cuda_gelu( input, output, n, sm.stream_->stream() );
}//gelu::implement_arithmetic_operation

void layer_norm::implement_arithmetic_operation
(
    std::vector<std::byte*> const& d_w, std::vector<std::vector<std::int64_t>> const& w_s,
    std::vector<std::byte*> const& d_i, std::vector<std::vector<std::int64_t>> const& i_s,
    std::vector<std::byte*> const& d_o, std::vector<std::vector<std::int64_t>> const& o_s,
    std::byte* d_b, std::int64_t b_s,
    stream<default_engine_type>& sm
)
{
    better_assert( (d_w.size() == 0)||(d_w.size()==2) );
    better_assert( (w_s.size() == 0)||(w_s.size()==2) );
    better_assert( d_i.size() == 1 );
    better_assert( d_o.size() == 1 );
    better_assert( i_s.size() == 1 );
    better_assert( o_s.size() == 1 );
    better_assert( i_s[0] == o_s[0] );
    better_assert( i_s[0].size() >= 2 );

    better_assert( (b_s >= static_cast<std::int64_t>(sizeof(float) * i_s[0][0] * 2)), nnl::format( "Cache size not large enought: allocated {} bytes, but at least {} bytes required.",  b_s, static_cast<std::int64_t>(sizeof(float) * i_s[0][0] * 2 )) );

    //spdlog::info( "layer_norm has {} bytes buffer memory", b_s );

    std::byte* mean = d_b;
    std::byte* var = d_b + (i_s[0][0] * sizeof( float ));

    std::int64_t const rows = i_s[0][0];
    std::int64_t const cols = std::accumulate( i_s[0].begin()+1, i_s[0].end(), 1LL, []( std::int64_t a, std::int64_t b ){ return a*b; } );

    cuda_layer_norm( reinterpret_cast<float*>(d_i[0]), // <-- input
                     reinterpret_cast<float*>(d_o[0]), // <-- output
                     static_cast<int>(rows),
                     static_cast<int>(cols),
                     static_cast<float>(1.0e-5f),
                     reinterpret_cast<float*>(mean),
                     reinterpret_cast<float*>(var),
                     sm.stream_->stream() );

    if (d_w.size()==0) //<- stop here if no weights
        return;

    // verify alpha/beta shape here
    better_assert( w_s.size() == 2 );
    better_assert( w_s[0][0] == cols );
    better_assert( w_s[1][0] == cols );

    // continue with scaled_offset
    cuda_scaled_offset( reinterpret_cast<float*>( d_o[0] ), // <- input
                        reinterpret_cast<float*>( d_o[0] ), // <- output
                        static_cast<int>( rows ),           // <- dim_a
                        static_cast<int>( cols ),           // <- dim_b
                        reinterpret_cast<float*>( d_w[0] ), // <- alpha
                        reinterpret_cast<float*>( d_w[1] ), // <- beta
                        sm.stream_->stream() );

}//layer_norm::implement_arithmetic_operation

// y = x * \alpha + \beta
// example diemension: x/y [10, 768], \alpha/\beta [768,]
//  void cuda_scaled_offset( float* x, float* y, int dim_a, int dim_b, float* alpha, float* beta, cudaStream_t sm );
void scaled_offset::implement_arithmetic_operation
(
    std::vector<std::byte*> const& d_w, std::vector<std::vector<std::int64_t>> const& w_s,
    std::vector<std::byte*> const& d_i, std::vector<std::vector<std::int64_t>> const& i_s,
    std::vector<std::byte*> const& d_o, std::vector<std::vector<std::int64_t>> const& o_s,
    [[maybe_unused]]std::byte* d_b, [[maybe_unused]]std::int64_t b_s,
    stream<default_engine_type>& sm
)
{
    float* alpha = nullptr;
    float* beta = nullptr;
    if ( d_w.size() == 0 ) // three inputs, no weight
    {
        better_assert( w_s.size() == 0 );
        better_assert( d_i.size() == 3 ); // x, alpha, beta
        // i_s exmaple: [ [10, 768], [768], [768] ]
        better_assert( i_s.size() == 3 );
        better_assert( d_o.size() == 1 ); // y
        // o_s exmaple: [ [10, 768] ]
        better_assert( o_s.size() == 1 );
        better_assert( i_s[0] == o_s[0] );
        better_assert( i_s[1] == i_s[2] );
        better_assert( i_s[0].size() == 2 );
        better_assert( i_s[1].size() == 1 );
        better_assert( i_s[0][1] == i_s[1][0] );

        alpha = reinterpret_cast<float*>( d_i[1] );
        beta = reinterpret_cast<float*>( d_i[2] );
    }
    else // one input with two weights
    {
        better_assert( d_w.size() == 2 ); // alpha, beta
        better_assert( w_s.size() == 2 ); // [768, ] [768, ]
        better_assert( d_i.size() == 1 ); // x, alpha, beta
        // i_s exmaple: [ [10, 768]
        better_assert( i_s.size() == 1 );
        better_assert( d_o.size() == 1 ); // y
        // o_s exmaple: [ [10, 768] ]
        better_assert( o_s.size() == 1 );
        better_assert( i_s[0] == o_s[0] );
        better_assert( w_s[0] == w_s[1] );
        better_assert( i_s[0].size() == 2 );
        better_assert( i_s[0][1] == w_s[0][0] );

        alpha = reinterpret_cast<float*>( d_w[0] );
        beta = reinterpret_cast<float*>( d_w[1] );
    }

    float* y = reinterpret_cast<float*>( d_o[0] );
    float* x = reinterpret_cast<float*>( d_i[0] );
    int dim_a = static_cast<int>( i_s[0][0] );
    int dim_b = static_cast<int>( i_s[0][1] );

    cuda_scaled_offset( x, y, dim_a, dim_b, alpha, beta, sm.stream_->stream() );
}//scaled_offset::implement_arithmetic_operation


void attention::implement_arithmetic_operation
(
    std::vector<std::byte*> const& d_w, std::vector<std::vector<std::int64_t>> const& w_s,
    std::vector<std::byte*> const& d_i, std::vector<std::vector<std::int64_t>> const& i_s,
    std::vector<std::byte*> const& d_o, std::vector<std::vector<std::int64_t>> const& o_s,
    std::byte* d_b, std::int64_t b_s,
    stream<default_engine_type>& sm
)
{
#if 0
    Attention( q, k, v ): // dimension [[n_q, d_k], [n_k, d_k], [n_k, d_v]]        <-- i_s
        softmax( [ q k^T / sqrt(n) + mask ] v ) // dimension [[n_q, d_v],]         <-- o_s
#endif
    better_assert( 0 == d_w.size() );
    better_assert( 0 == w_s.size() );
    better_assert( 3 == d_i.size() );
    better_assert( 3 == i_s.size() );
    better_assert( 1 == d_o.size() );
    better_assert( 1 == o_s.size() );
    better_assert( 2 == i_s[0].size() );
    better_assert( 2 == i_s[1].size() );
    better_assert( 2 == i_s[2].size() );
    better_assert( 2 == o_s[0].size() );
    better_assert( i_s[0][1] == i_s[1][1] );
    better_assert( i_s[1][0] == i_s[2][0] );
    better_assert( o_s[0][0] == i_s[0][0] );
    better_assert( o_s[0][1] == i_s[2][1] );
    better_assert( i_s[0][0] == i_s[1][0] ); // n_q == n_k

    std::int64_t const n_q = i_s[0][0];
    std::int64_t const d_k = i_s[0][1];
    std::int64_t const n_k = i_s[1][0];
    //std::int64_t const d_k = i_s[1][1];
    //std::int64_t const n_k = i_s[2][0];
    std::int64_t const d_v = i_s[2][1];



    {
        // buffer = q k^T
        //[[n_q, d_k], [n_k, d_k]] -> [n_q, n_k]
        better_assert( b_s >= i_s[0][0] * i_s[1][0] ); // <-- check the buffer size
        cuda_gemm(  reinterpret_cast<float*>( d_i[0] ),
                    false,
                    reinterpret_cast<float*>( d_i[1] ),
                    true,
                    n_q, //i_s[0][0],
                    d_k, //i_s[0][1],
                    n_k, //i_s[1][0],
                    reinterpret_cast<float*>( d_b ),
                    1.0f,
                    0.0f,
                    sm,
                    1 );
    }

    {
        // qk = qk / sqrt(n) + mask( n )
        cuda_apply_scaled_mask( reinterpret_cast<float*>( d_b ), // view as [n_q, n_k]
                                n_q, //i_s[0][0], // n_q
                                n_q, //i_s[1][0], // n_k
                                static_cast<float>( 1.0f / std::sqrt(1.0f*d_k) ), // \frac{1}{\sqrt{d_k}}
                                sm.stream_->stream() );
    }

    {
        //softmax
        cudnn_softmax( reinterpret_cast<float*>( d_b ), // input
                       reinterpret_cast<float*>( d_b ), // output
                       n_q, //i_s[0][0], // batch size
                       n_k, // i_s[2][1], // dim_0 // did I make a mistake here?
                       1, // dim_1
                       1, // dim_2
                       sm,  1.0f, 0.0f,  true );
    }

    {
        // o = qk * v
        cuda_gemm( reinterpret_cast<float*>( d_b ), // shape [n_q, n_k]
                   false,
                   reinterpret_cast<float*>( d_i[2] ), // shape [n_k, d_v]
                   false,
                   n_q, //i_s[0][0], // n_q
                   n_k, //i_s[1][0], // n_k
                   d_v, //i_s[2][1], // d_v
                   reinterpret_cast<float*>( d_o[0] ),
                   1.0f,
                   0.0f,
                   sm );
    }

}//attention::implement_arithmetic_operation


// Used in dot-product attention
void query_product::implement_arithmetic_operation
(
    std::vector<std::byte*> const& d_w, std::vector<std::vector<std::int64_t>> const& w_s,
    std::vector<std::byte*> const& d_i, std::vector<std::vector<std::int64_t>> const& i_s,
    std::vector<std::byte*> const& d_o, std::vector<std::vector<std::int64_t>> const& o_s,
    std::byte* d_b, std::int64_t b_s,
    stream<default_engine_type>& sm
)
{
#if 0
    QueryProduct( q, k ): // dimension [[n_q, d_k], [n_k, d_k] ]        <-- i_s
        q k^T // dimension [[n_q, n_k],]                                <-- o_s
#endif
    better_assert( 0 == d_w.size() ); // no weights
    better_assert( 0 == w_s.size() ); // weights shape empty
    better_assert( 2 == d_i.size() ); // two inputs
    better_assert( 2 == i_s.size() ); // two shapes for inputs
    better_assert( 1 == d_o.size() ); // one output
    better_assert( 1 == o_s.size() ); // one shape for input
    better_assert( 2 == i_s[0].size() ); // the first input is a matrix
    better_assert( 2 == i_s[1].size() ); // the second input is a matrix
    better_assert( 2 == o_s[0].size() ); // the output is a matrix
    better_assert( i_s[0][1] == i_s[1][1] ); // dimension check for the inputs
    better_assert( o_s[0][0] == i_s[0][0] ); // dimension check for the first dimension of the output matrix
    better_assert( o_s[0][1] == i_s[1][0] ); // dimension check for the second dimension of the output matrix
    better_assert( d_b == nullptr );
    better_assert( b_s == 0 );

    {
        cuda_gemm(  reinterpret_cast<float*>( d_i[0] ),
                    false,
                    reinterpret_cast<float*>( d_i[1] ),
                    true,
                    i_s[0][0],
                    i_s[0][1],
                    i_s[1][0],
                    reinterpret_cast<float*>( d_o[0] ),
                    1.0f,
                    0.0f,
                    sm,
                    1 );

        spdlog::info( nnl::format( "QueryProduct called with input shapes [{}, {}] and output shape [{}]", i_s[0], i_s[1], o_s[0] ) );
    }

}//query_product::implement_arithmetic_operation

void scaled_mask::implement_arithmetic_operation
(
    std::vector<std::byte*> const& d_w, std::vector<std::vector<std::int64_t>> const& w_s,
    std::vector<std::byte*> const& d_i, std::vector<std::vector<std::int64_t>> const& i_s,
    std::vector<std::byte*> const& d_o, std::vector<std::vector<std::int64_t>> const& o_s,
    std::byte* d_b, std::int64_t b_s,
    stream<default_engine_type>& sm
)
{
#if 0
    calculate x/sqrt(n) + mask
#endif
    better_assert( 0 == d_w.size() ); // no weights
    better_assert( 0 == w_s.size() ); // weights shape empty
    better_assert( 1 == d_i.size() ); // single input
    better_assert( 1 == i_s.size() ); // single input shape
    better_assert( 1 == d_o.size() ); // single output
    better_assert( 1 == o_s.size() ); // single output shape
    better_assert( 2 == i_s[0].size() ); // input is a matrix
    better_assert( 2 == o_s[0].size() ); // output is a matrix
    better_assert( i_s[0][0] == o_s[0][0] ); // input-output of same shape
    better_assert( i_s[0][1] == o_s[0][1] ); // input-output of same shape
    better_assert( d_b == nullptr );
    better_assert( b_s == 0 );
    better_assert( i_s[0][0] == i_s[0][1] ); // n_q equals to n_k

    {
        // qk = qk / sqrt(n) + mask( n )
        cuda_apply_scaled_mask_2(   reinterpret_cast<float*>( d_i[0] ), // view as [n_q, n_k]
                                    reinterpret_cast<float*>( d_o[0] ),
                                    i_s[0][0], // n_q
                                    i_s[0][1], // n_k
                                    static_cast<float>( 1.0f / std::sqrt(1.0f*i_s[0][1]) ), // updated
                                    sm.stream_->stream() );
    }
}//scaled_mask::implement_arithmetic_operation

void multiply::implement_arithmetic_operation
(
    std::vector<std::byte*> const& d_w, std::vector<std::vector<std::int64_t>> const& w_s,
    std::vector<std::byte*> const& d_i, std::vector<std::vector<std::int64_t>> const& i_s,
    std::vector<std::byte*> const& d_o, std::vector<std::vector<std::int64_t>> const& o_s,
    std::byte* d_b, std::int64_t b_s,
    stream<default_engine_type>& sm
)
{
#if 0
    Calculate A \dot B, where A and B are two matrices
#endif
    better_assert( 0 == d_w.size() ); // no weights
    better_assert( 0 == w_s.size() ); // weights shape empty
    better_assert( 2 == d_i.size() ); // two inputs
    better_assert( 2 == i_s.size() ); // two shapes for inputs
    better_assert( 1 == d_o.size() ); // one output
    better_assert( 1 == o_s.size() ); // one shape for input
    better_assert( 2 == i_s[0].size() ); // the first input is a matrix
    better_assert( 2 == i_s[1].size() ); // the second input is a matrix
    better_assert( 2 == o_s[0].size() ); // the output is a matrix
    better_assert( i_s[0][1] == i_s[1][0] ); // dimension check for the inputs
    better_assert( o_s[0][0] == i_s[0][0] ); // dimension check for the first dimension of the output matrix
    better_assert( o_s[0][1] == i_s[1][1] ); // dimension check for the second dimension of the output matrix
    better_assert( d_b == nullptr );
    better_assert( b_s == 0 );

    {
        cuda_gemm(  reinterpret_cast<float*>( d_i[0] ),
                    false,
                    reinterpret_cast<float*>( d_i[1] ),
                    false,
                    i_s[0][0],
                    i_s[0][1],
                    i_s[1][1],
                    reinterpret_cast<float*>( d_o[0] ),
                    1.0f,
                    0.0f,
                    sm,
                    1 );
    }

}//multiply::implement_arithmetic_operation

void multi_head_attention::implement_attributed_arithmetic_operation
(
    [[maybe_unused]] std::vector<std::byte*> const& d_w, [[maybe_unused]]std::vector<std::vector<std::int64_t>> const& w_s,
    [[maybe_unused]] std::vector<std::byte*> const& d_i, [[maybe_unused]] std::vector<std::vector<std::int64_t>> const& i_s,
    [[maybe_unused]] std::vector<std::byte*> const& d_o, [[maybe_unused]] std::vector<std::vector<std::int64_t>> const& o_s,
    [[maybe_unused]] std::byte* d_b, [[maybe_unused]] std::int64_t b_s,
    [[maybe_unused]] stream<default_engine_type>& sm,
    [[maybe_unused]] std::vector<attribute> const& attributes
)
{
    // dw: w_att, b_att, w_proj, b_proj
    // di:
    better_assert( 4 == d_w.size () );
    better_assert( 4 == w_s.size () );
    better_assert( 2 == w_s[0].size () );
    better_assert( 1 == w_s[1].size () );
    better_assert( 2 == w_s[2].size () );
    better_assert( 1 == w_s[3].size () );
    better_assert( 1 == d_i.size () );
    better_assert( 1 == i_s.size () );
    better_assert( 2 == i_s[0].size () );
    better_assert( 1 == d_o.size () );
    better_assert( 1 == o_s.size () );
    better_assert( 2 == o_s[0].size () );

    // check weights and output dimensions
    std::int64_t const n_seq = i_s[0][0];
    std::int64_t const n_embd = i_s[0][1];
    better_assert( 1 * n_embd == w_s[0][0] );
    better_assert( 3 * n_embd == w_s[0][1] );
    better_assert( 3 * n_embd == w_s[1][0] );
    better_assert( 1 * n_embd == w_s[2][0] );
    better_assert( 1 * n_embd == w_s[2][1] );
    better_assert( 1 * n_embd == w_s[3][0] );
    better_assert( 1 * n_seq  == o_s[0][0] );
    better_assert( 1 * n_embd == o_s[0][1] );

    // check heads
    std::int64_t heads = -1;
    for ( auto const& att : attributes )
        if ( att.name() == "heads" )
            heads = att.heads().value();
    better_assert( 0 < heads );
    std::int64_t const n_div = static_cast<std::int64_t>( n_embd * 1.0 / heads );
    better_assert( n_div*heads == n_embd );

    float* x      = reinterpret_cast<float*>( d_i[0] ); // shape [n_seq, n_embd]
    float* w_attn = reinterpret_cast<float*>( d_w[0] ); // shape [n_embd, n_embd*3]
    float* b_attn = reinterpret_cast<float*>( d_w[1] ); // shape [n_embd*3]
    float* w_proj = reinterpret_cast<float*>( d_w[2] ); // shape [n_embd, n_embd]
    float* b_proj = reinterpret_cast<float*>( d_w[3] ); // shape [n_embd]
    float* output = reinterpret_cast<float*>( d_b ); // shape [n_embd*3, n_seq ]
    float* result = reinterpret_cast<float*>( d_o[0] ); // shape [n_seq, n_embd]

    // output = w_attn^T * x^T
    add_bias( output, b_attn, n_embd*3, n_seq, 1, sm.stream_->stream() );
    cuda_gemm( w_attn, true, x, true, n_embd*3, n_embd, n_seq, output, 1.0f, 1.0f, sm );

    float* q     = output;                  // shape [n_seq, n_embd]
    float* k     = q + n_embd * n_seq;      // shape [n_seq, n_embd]
    float* v     = k + n_embd * n_seq;      // shape [n_seq, n_embd]
    float* buff  = output + n_embd*3*n_seq; // [n_seq, n_seq]
    float* out   = buff + n_seq * n_seq * heads;    // [n_seq, n_embd]

    // TODO: parallel here using events
    for ( std::int64_t idx = 0; idx < heads; ++idx )
    {
        float* _q = q + n_seq * n_div * idx; // shape [n_seq, n_div]
        float* _k = k + n_seq * n_div * idx; // shape [n_seq, n_div]
        float* _v = v + n_seq * n_div * idx; // shape [n_seq, n_div]
        float* _b = buff + n_seq * n_seq * idx; // shape [n_seq, n_seq]
        float* _o = out + n_seq * n_div * idx;     // shape [n_seq, n_div]

        // QK = Q * K^T -> QK = q^T * k
        cuda_gemm( _q, true, _k, false, n_seq, n_div, n_seq, _b, 1.0f, 0.0f, sm );

        // QK = QK / sqrt(n) + mask(n)
        cuda_apply_scaled_mask( _b, n_seq, n_seq, static_cast<float>( 1.0f / std::sqrt(1.0f*n_div) ), sm.stream_->stream() );

        // softmax(QK)
        cudnn_softmax( _b, _b, n_seq, n_seq, 1, 1, sm, 1.0f, 0.0f, true );

        // output^T = v * softmax(QT)^T
        cuda_gemm( _v, false, _b, true, n_div, n_seq, n_seq, _o, 1.0f, 0.0f, sm );
    }

    add_bias( result, b_proj, n_seq, n_embd, 0, sm.stream_->stream() );
    cuda_gemm( out, true, w_proj, false, n_seq, n_embd, n_embd, result, 1.0f, 1.0f, sm ); // failed
}

}//namespace nnl::node_



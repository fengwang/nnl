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

TEST_CASE( "Multi-head-attention-model-42", "[multi-head-attention-model-42]" )
{
    spdlog::info( "\nTest case of 42 started.\n" );

    using namespace nnl;
    auto& sess = get_default_session<default_engine_type>();
    sess.clean();

    std::int64_t const n_inp = 2;
    std::int64_t const n_dim_1 = 3;
    std::int64_t const n_dim_2 = 5;
    std::int64_t const n_dim_3 = 7;

    auto input = make_tensor<default_engine_type>( {n_inp, n_dim_1}, "float32", "tensor_input" );
    auto w_c_fc = make_tensor<default_engine_type>( {n_dim_1, n_dim_2}, "float32", "tensor_w_fc" );
    auto b_c_fc = make_tensor<default_engine_type>( {n_dim_2,}, "float32", "tensor_b_fc" );
    auto w_c_proj = make_tensor<default_engine_type>( {n_dim_2, n_dim_3}, "float32", "tensor_w_proj" );
    auto b_c_proj = make_tensor<default_engine_type>( {n_dim_3,}, "float32", "tensor_b_proj" );
    auto gt = make_tensor<default_engine_type>( {n_inp, n_dim_3}, "float32", "tensor_gt" );


    // shape is (2, 3)
    [[maybe_unused]] float d_input[]=
    {
    	-0.5534178416929223,	0.046326682801352215,	0.10140291316236216,	-0.9087960997343874,	-0.27854232930445466,	-0.5538381166174224,
    };

    // shape is (3, 5)
    [[maybe_unused]] float d_w_c_fc[]=
    {
    	0.377452323642713,	-0.6725371499982016,	-0.8593502663889994,	0.882021720491672,	0.12736276046965056,	-0.8440153212200259,	0.4452810222496797,	-0.6830956529536671,	-0.49943738651195324,	-0.413025487795605,	0.39322142841718977,	-0.07147182391969653,	-0.569875712571321,	-0.10634747584068482,	-0.7582424969220862,
    };

    // shape is (5,)
    [[maybe_unused]] float d_b_c_fc[]=
    {
    	0.83537401548945,	0.13289265743839107,	-0.05596136851614553,	-0.37086715070557696,	-0.9135685405267477,
    };

    // shape is (5, 7)
    [[maybe_unused]] float d_w_c_proj[]=
    {
    	0.5048898339199461,	0.08653808774263605,	0.5343808514294013,	0.5808975324315737,	-0.8555572412507397,	0.31943166881569507,	-0.8934013233115059,	0.3700489054454015,	-0.1132980759734703,	-0.06400770524007782,	0.32211866952991897,	0.3799332638532238,	-0.4733410311088724,	-0.9770421324796503,	0.5425888932495304,	0.6322846525776988,	-0.7207167925034084,	0.15601349114968222,	-0.3186514559116014,	-0.535569025061867,	-0.6027124928297907,	-0.4570184534226387,	0.953705528021386,	-0.5691420434372769,	-0.28718516587580756,	-0.9632001067387674,	0.2335805391900283,	-0.5391892244378311,	0.4847034721316992,	0.168500416789773,	-0.15979289082048598,	0.41850566438972425,	0.9440910518067389,	0.6200782353145564,	-0.13868964067662737,
    };

    // shape is (7,)
    [[maybe_unused]] float d_b_c_proj[]=
    {
    	0.9154571996229015,	-0.9044837977902083,	0.46556360178337663,	0.35133289856278926,	0.05401033558831192,	0.1339749037447604,	0.445890498494282,
    };

    // shape is (2, 7)
    [[maybe_unused]] float d_gt[]=
    {
    	1.3973097014011353,	-0.9581874240695021,	0.6576656228803843,	0.7523448000215591,	-0.25195359523243144,	-0.13410409043450267,	-0.3340111107365086,	1.8700849918106457,	-0.4156330616850512,	-0.05070399894727873,	0.868361554884096,	-0.40154665181164256,	-0.7054922853608465,	-0.9034884992640484,
    };



    input.import_from( d_input );
    w_c_fc.import_from( d_w_c_fc );
    b_c_fc.import_from( d_b_c_fc );
    w_c_proj.import_from( d_w_c_proj );
    b_c_proj.import_from( d_b_c_proj );
    gt.import_from( d_gt );

    auto input_layer = Input( "InputLayer" );
    auto output_layer = PositionwiseFeedForward( w_c_fc, b_c_fc, w_c_proj, b_c_proj, "PositionwiseFeedForward_42" )( input_layer );
    auto m = model( input_layer, output_layer );
    auto outputs = m.predict( input );
    auto output = outputs[0];

    output.save_txt( "./output.txt" );

    auto mat = view_2d{ reinterpret_cast<float*>( gt.data() ), static_cast<std::uint64_t>(n_inp), static_cast<std::uint64_t>(n_dim_3) };
    auto nat = view_2d{ reinterpret_cast<float*>( output.data() ), static_cast<std::uint64_t>(n_inp), static_cast<std::uint64_t>(n_dim_3) };
    for ( auto r : range( n_inp ) )
    {
        for ( auto c : range( n_dim_3 ) )
        {
            if ( std::abs(mat[r][c]-nat[r][c]) > 1.0e-2 )
            {
                spdlog::error( "gt[{}][{}]={}, pred[{}][{}]={}", r, c, mat[r][c], r, c, nat[r][c] );
            }
            REQUIRE( std::abs(mat[r][c]-nat[r][c]) < 1.0e-2 );
        }
    }
}


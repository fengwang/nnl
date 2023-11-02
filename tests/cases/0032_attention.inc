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

TEST_CASE( "Node-attention-32", "[node-attention-32]" )
{
    using namespace nnl;

    auto& sess = get_default_session<default_engine_type>();
    sess.clean();

    {
        auto inp_q = make_node( "Input", "inp_q" );
        auto inp_k = make_node( "Input", "inp_k" );
        auto inp_v = make_node( "Input", "inp_v" );
        auto l1 = make_node( "Attention", "attention" );
        l1.set_output_node();

        std::int64_t n_q = 16;
        std::int64_t d_k = 16;
        std::int64_t n_k = 16;
        std::int64_t d_v = 16;

        n_q = 3;
        d_k = 5;
        n_k = n_q;
        d_v = 7;



    // shape is (3, 3)
    [[maybe_unused]] float attention_query_product[]=
    {
    	0.012098540800983014,	0.869690588166359,	-2.932854663183885,	3.117252399281158,	0.44855903258528684,	1.431434863170286,
    	0.6273548772764069,	-0.057556832760215476,	-0.09258567893289538,
    };

    // shape is (3, 3)
    [[maybe_unused]] float attention_scaled_mask[]=
    {
    	0.005410631931910555,	-9999999999.611063,	-10000000001.311613,	1.3940776535633972,	0.20060169775644893,	-9999999999.359842,
    	0.2805616303212168,	-0.02574019812428573,	-0.041405574367384845,
    };

    // shape is (3, 3)
    [[maybe_unused]] float attention_softmax[]=
    {
    	1.0,	0.0,	0.0,	0.7673621596673126,	0.23263784033268747,	0.0,
    	0.4063576527283134,	0.29914603377283244,	0.2944963134988543,
    };

    // shape is (3, 7)
    [[maybe_unused]] float attention_projection[]=
    {
    	0.49087183250951594,	1.8927422187767202,	-0.6209797023155086,	-0.4537523812416193,	0.21745165945019368,	0.5143288551447859,
    	0.3972413264075336,	0.02473144914223191,	1.2746755440519897,	-0.4529571412552762,	-0.4220004775288783,	0.43168351676684225,
    	0.3197383491838275,	0.21872495705849848,	0.2916119918549111,	0.5165264712692113,	-0.5107325725319722,	-0.27999816866998034,
    	0.3604423997120315,	0.32201890801421185,	0.18415683719413828,
    };

    // shape is (3, 7)
    [[maybe_unused]] float gt[]=
    {
    	0.49087183250951594,	1.8927422187767202,	-0.6209797023155086,	-0.4537523812416193,	0.21745165945019368,	0.5143288551447859,
    	0.3972413264075336,	0.02473144914223191,	1.2746755440519897,	-0.4529571412552762,	-0.4220004775288783,	0.43168351676684225,
    	0.3197383491838275,	0.21872495705849848,	0.2916119918549111,	0.5165264712692113,	-0.5107325725319722,	-0.27999816866998034,
    	0.3604423997120315,	0.32201890801421185,	0.18415683719413828,
    };

    // shape is (3, 5)
    [[maybe_unused]] float d_tq[]=
    {
    	-0.712390662050588,	0.753766378659703,	-0.044503078338053455,	0.45181233874578974,	1.3451017084510097,	0.5323378882945463,
    	1.3501878997225267,	0.8612113741693206,	1.4786857374358966,	-1.0453771305385342,	-0.7889890249515489,	-1.261605945319069,
    	0.5628467852810314,	-0.24332625188556253,	0.9137407048596775,
    };

    // shape is (3, 5)
    [[maybe_unused]] float d_tk[]=
    {
    	0.31735092273633597,	0.12730328020698067,	2.1503829673811126,	0.6062886568962988,	-0.026771649986440726,	-0.9841607817725814,
    	1.190705272505982,	0.9528306110451948,	-1.0871815907509188,	-0.1452113325860197,	0.23785783828501061,	-1.639093411201915,
    	-0.27813451650877447,	1.3992384201681904,	-1.6151079632521659,
    };

    // shape is (3, 7)
    [[maybe_unused]] float d_tv[]=
    {
    	0.49087183250951594,	1.8927422187767202,	-0.6209797023155086,	-0.4537523812416193,	0.21745165945019368,	0.5143288551447859,
    	0.3972413264075336,	-1.5128451152607716,	-0.7640339696578627,	0.1012697856437548,	-0.31726597100271015,	1.1383330474921514,
    	-0.3221236569774085,	-0.37011607810475444,	1.8496125662057936,	-0.08165155616424355,	-0.9802743185258672,	-0.002389025600514556,
    	-0.2324258720274623,	0.7109747948581455,	0.4531586132388156,
    };

        auto t_q = make_tensor<default_engine_type>( {n_q, d_k}, "float32", "t_q" );
        t_q.import_from( d_tq );
        auto t_k = make_tensor<default_engine_type>( {n_k, d_k}, "float32", "t_k" );
        t_k.import_from( d_tk );
        auto t_v = make_tensor<default_engine_type>( {n_k, d_v}, "float32", "t_v" );
        t_v.import_from( d_tv );
        auto t_b = make_tensor<default_engine_type>( {n_q, n_k}, "float32", "t_b" );
        auto t_o = make_tensor<default_engine_type>( {n_q, d_v}, "float32", "t_o" );


        layer input_q = Input( "input_q" );
        layer input_k = Input( "input_k" );
        layer input_v = Input( "input_v" );
        layer output = Attention( "attention" )( input_q, input_k, input_v );

        model m{ {input_q, input_k, input_v, }, output };
        auto results = m.predict( {t_q, t_k, t_v,} );
        //results[0].save_txt( "./0032_attation_result.txt" );

        float* prediction = reinterpret_cast<float*>( results[0].data() );
        for ( auto idx : range( n_q * d_v ) )
        {
            CHECK( std::abs(prediction[idx] - gt[idx]) < 1.0e-2 );
        }

    }

    REQUIRE( true );
}


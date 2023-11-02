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

TEST_CASE( "Node-attention-19-no-buffer", "[node-attention-19-no-buffer]" )
{
    std::cout << "Test case 19 of node-attention-19-no-buffer" << std::endl;

    using namespace nnl;

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

        auto t_q = make_tensor<default_engine_type>( {n_q, d_k}, "float32", "t_q" );
        auto t_k = make_tensor<default_engine_type>( {n_k, d_k}, "float32", "t_k" );
        auto t_v = make_tensor<default_engine_type>( {n_k, d_v}, "float32", "t_v" );
        //auto t_b = make_tensor<default_engine_type>( {n_q, n_k}, "float32", "t_b" );
        auto t_o = make_tensor<default_engine_type>( {n_q, d_v}, "float32", "t_o" );

        {
            float start = 0.0f;
            float step = 0.05f;

            {
                std::uint64_t dim_a = static_cast<std::uint64_t>( n_q );
                std::uint64_t dim_b = static_cast<std::uint64_t>( d_k );
                auto mat = view_2d{ reinterpret_cast<float*>( t_q.data() ), dim_a, dim_b  };
                for ( auto r : range( dim_a ) )
                    for ( auto c : range( dim_b ) )
                    {
                        mat[r][c] = std::sin(start);
                        //mat[r][c] = (start);
                        start += step;
                    }
            }
            {
                std::uint64_t dim_a = static_cast<std::uint64_t>( n_k );
                std::uint64_t dim_b = static_cast<std::uint64_t>( d_k );
                auto mat = view_2d{ reinterpret_cast<float*>( t_k.data() ), dim_a, dim_b  };
                for ( auto r : range( dim_a ) )
                    for ( auto c : range( dim_b ) )
                    {
                        mat[r][c] = std::sin(start);
                        //mat[r][c] = (start);
                        start += step;
                    }
            }
            {
                std::uint64_t dim_a = static_cast<std::uint64_t>( n_k );
                std::uint64_t dim_b = static_cast<std::uint64_t>( d_v );
                auto mat = view_2d{ reinterpret_cast<float*>( t_v.data() ), dim_a, dim_b  };
                std::cout << "V matrix:\n";
                for ( auto r : range( dim_a ) )
                {
                    for ( auto c : range( dim_b ) )
                    {
                        mat[r][c] = std::sin(start);
                        //mat[r][c] = (start);
                        start += step;
                        std::cout << mat[r][c] << "\t";
                    }
                    std::cout << "\n";
                }
            }
        }

        {
            inp_q.add_input( t_q );
            inp_k.add_input( t_k );
            inp_v.add_input( t_v );
        }
        {
            l1.add_input( t_q );
            l1.add_input( t_k );
            l1.add_input( t_v );
            l1.add_output( t_o );
            //l1.add_buffer( t_b );
        }

        // create computation graph
        graph<node> g;
        {
            g.connect( inp_q, l1 );
            g.connect( inp_k, l1 );
            g.connect( inp_v, l1 );
        }


        auto computation_table = make_computation_table( g );


        std::cout << "Get computation table" << std::endl;
        for ( auto & ct : computation_table )
            std::cout << ct << std::endl;


        auto& sess = get_default_session<default_engine_type>();
        for ( auto & ct : computation_table )
        {
            std::cout << "Before computation runs, the session is\n" << sess << std::endl;
            auto& [ao, host2device_operations, device2host_operations, device2host_dismiss_operations,
                  wp, wd, op] = ct;
            {
                std::cout << "arithmetic operations" << std::endl;
                if ( ao )
                    ao.arithmetic_operation();

                std::cout << "host2device operations" << std::endl;
                for ( auto h2do : host2device_operations )
                    if (h2do)
                        h2do.host2device_operation();

                std::cout << "device2host operations" << std::endl;
                for ( auto d2ho : device2host_operations )
                    if (d2ho)
                        d2ho.device2host_operation();

                std::cout << "device2hostdismiss operations" << std::endl;
                for ( auto d2hdo : device2host_dismiss_operations )
                    if (d2hdo)
                        d2hdo.device2host_dismiss_operation();

                std::cout << "weight preloading operations" << std::endl;
                if (wp)
                    wp.weight_preloading();

                //std::cout << "skip weight dismissing operations" << std::endl;
                std::cout << "weight dismissing operations" << std::endl;
                if (wd)
                    wd.weight_dismissing();

                std::cout << "output preallocating" << std::endl;
                if (op)
                    op.output_preallocating();
            }

            sess.get_device_memory_manager().synchronize();
        }

        float gt[] = { 0.4504405942753893,0.49449938310466274,0.5373221810064719,0.5788019532877502,0.6188350221200393,0.6573213256800776,0.6941646682522445,0.729272960667749,0.7625584504796027,0.7939379412980213,0.8233330007380816,0.8506701564598345,0.8758810798108894,0.8989027566124672,0.9196776446620198,0.9381538175587758,0.8546974179188622,0.8744349588427265,0.8919868677664072,0.9073092740584532,0.9203638796828885,0.9311180549243594,0.9395449199454281,0.9456234119721684,0.9493383379401316,0.9506804124690933,0.94964628107167,0.9462385285377909,0.9404656724740657,0.9323421420142096,0.9218882417537192,0.9091301009989597,0.8677616023019475,0.8731454180232477,0.8763468249248333,0.8773578211567109,0.876175879754817,0.8728039549571143,0.8672504748195305,0.859529320150214,0.8496597898147388,0.8376665524989914,0.8235795850503046,0.8074340975509481,0.7892704453112628,0.7691340280024013,0.7470751761807923,0.7231490254879619,0.7441347192019135,0.7421492360221426,0.7383087662561665,0.7326229090783231,0.7251058761704192,0.7157764561999308,0.7046579678581135,0.691778201575426,0.6771693500599234,0.6608679278322509,0.6429146799583628,0.6233544802080705,0.6022362188939949,0.5796126806712444,0.5555404126032647,0.5300795828236312,-0.13630796364600223,-0.17059441352739602,-0.20445446621882457,-0.2378034892225335,-0.2705581273488476,-0.30263651106073924,-0.33395846110502797,-0.3644456889187587,-0.3940219923098382,-0.4226134459228285,-0.4501485860138401,-0.4765585890726697,-0.5017774438457331,-0.5257421163298232,-0.5483927073242891,-0.5696726021478455,-0.8463807519145914,-0.8643566412348184,-0.880172089100195,-0.8937875651275836,-0.905169037717719,-0.9142880591164823,-0.9211218365194711,-0.9256532890421402,-0.9278710904131182,-0.9277696972839852,-0.9253493630847606,-0.9206161373904563,-0.913581850800292,-0.9042640853673584,-0.8926861306526369,-0.8788769255132339,-0.8463633698294752,-0.8515715984809735,-0.8546513416261851,-0.8555949015111449,-0.8543999197275378,-0.8510693831074885,-0.845611616258037,-0.8380402607539579,-0.8283742410409242,-0.8166377171342525,-0.8028600242314471,-0.7870755993894809,-0.7693238954500901,-0.7496492824282134,-0.7281009366100645,-0.7047327176380282,-0.4944385470162925,-0.492561244254556,-0.48945279490312044,-0.4851209684665143,-0.47957659225485805,-0.47283352432122583,-0.4649086188237305,-0.4558216838989043,-0.44559543215167424,-0.4342554238856795,-0.42183000321582353,-0.4083502272227475,-0.39384978832630235,-0.3783649300720457,-0.36193435654125455,-0.34459913561087957,0.6172993920550325,0.6315030561931749,0.6441282915712697,0.6551435416759651,0.6645212741186325,0.6722380494519924,0.67827457975653,0.6826157768502492,0.6852507900012671,0.6861730330489949,0.685380200866108,0.6828742751201636,0.6786615193204664,0.672752463162555,0.6651618762094473,0.6559087309754246,0.8547828141927478,0.8649506283237526,0.8729565163415277,0.8787804676954138,0.8824079255400825,0.8838298231200648,0.8830426064319201,0.8800482431073957,0.8748542174953674,0.8674735119548701,0.8579245744059578,0.8462312722195106,0.8324228325612425,0.816533769339001,0.79860379693598,0.7786777309454362,0.8419790823541802,0.8460468466834739,0.8479999345087387,0.8478334641275607,0.8455478516291968,0.8411488098545679,0.8346473341171343,0.8260596747203367,0.8154072963403004,0.8027168243753251,0.7880199783962518,0.7713534928640589,0.7527590253128391,0.7322830522276592,0.7099767528775526,0.6858958813939962,0.5488475679907334,0.5457264133327087,0.5412412268501741,0.5354032191734966,0.528226982281492,0.5197304530290779,0.5099348683144779,0.4988647119980377,0.48654765370532244,0.4730144796674637,0.45829901577161525,0.44243804301385065,0.4254712055658299,0.40744091168501545,0.38839222771611365,0.36837276544868275,-0.5833906468488498,-0.5981304519281115,-0.6113752423778703,-0.6230919131197565,-0.6332511785788384,-0.6418276458823109,-0.6487998783284544,-0.6541504489672219,-0.6578659841585397,-0.6599371969994424,-0.6603589105364941,-0.6591300707054776,-0.6562537489660025,-0.6517371346244584,-0.6455915168644908,-0.6378322565299245,-0.8529041568485303,-0.862456391944836,-0.8698529352198862,-0.8750752991675381,-0.878110430597677,-0.8789507432623933,-0.8775941368176522,-0.874044002073058,-0.868309212516598,-0.8604041021355429,-0.8503484295889432,-0.8381673288212724,-0.8238912462406545,-0.8075558646187015,-0.7892020139021668,-0.7688755691593401,-0.8251766560285858,-0.8290797292642093,-0.8309105349530486,-0.8306644970343472,-0.8283422304747674,-0.8239495397312947,-0.8174974042431185,-0.8090019509887523,-0.7984844141769863,-0.7859710821724223,-0.7714932317882538,-0.7550870501105224,-0.7367935440492479,-0.7166584378425114,-0.6947320587696684,-0.6710692113593631,-0.3389457745141629,-0.33550022658418505,-0.3312161028128875,-0.32610411127857,-0.32017702929779407,-0.3134496714887464,-0.30593885274241023,-0.2976633461940925,-0.2886438363003642,-0.2789028671386891,-0.2684647860589718,-0.2573556828278649,-0.24560332441793703,-0.23323708560470363,-0.22028787554498472,-0.20678806052011095 };

        CHECK( t_o.synchronize_to_host() );
        {
            sess.synchronize();
            auto mat = view_2d{ reinterpret_cast<float*>( t_o.data() ), static_cast<std::uint64_t>( n_q ), static_cast<std::uint64_t>( d_v ) };
            auto nat = view_2d{ gt, static_cast<std::uint64_t>( n_q ), static_cast<std::uint64_t>( d_v ) };

            std::cout << "Output:\n";
            for ( auto jdx : range(n_q) )
            {
                for ( auto idx : range(d_v) )
                {
                    std::cout << mat[jdx][idx] << "\t";
                }
                std::cout << "\n";
            }

            for ( auto jdx : range(n_q) )
            {
                for ( auto idx : range(d_v) )
                {
                    CHECK( std::abs(mat[jdx][idx]-nat[jdx][idx]) < 1.0e-3f );
                }
            }
        }

        std::cout << "After inference, the session is\n" << sess << std::endl;

        sess.clean();
        std::cout << "After cleaning, the session is\n" << sess << std::endl;
    }

    REQUIRE( true );
}


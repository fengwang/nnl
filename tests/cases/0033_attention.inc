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

TEST_CASE( "Node-attention-33", "[node-attention-33]" )
{
    using namespace nnl;

    spdlog::info( "Starting test case attention-33." );

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



    // shape is (13, 13)
    [[maybe_unused]] float attention_query_product[]=
    {
    	-2.3089101177997917,	-1.0157575081223391,	2.7317553926350446,	-2.4289140226827994,	1.8036157519498819,	-2.0525693275838,
    	5.179883588565844,	-1.170869593900846,	0.8514176730400864,	0.17171553579965737,	-4.072092634033126,	-0.8550310804338157,
    	-4.780736572429341,	-1.633668278448795,	-3.061397562680069,	-7.424613714631217,	-0.03848403549004925,	8.728242071157164,
    	4.493577697768049,	5.556585886079799,	2.874439673923956,	-1.334022181131233,	5.081557982104428,	-0.40030239781546073,
    	3.465332266257257,	7.625478879360275,	-0.18080471303533724,	-2.7787869345593683,	8.396486034820517,	-1.7641555447108768,
    	-1.2259411201260213,	-8.89188975842835,	4.212319383798791,	3.7137151929311596,	-2.0285063855411454,	-8.251073194394804,
    	-0.06930019602682425,	-2.029028389334754,	-9.970232685366174,	5.790697398291659,	1.1116880423695037,	-5.347326853478983,
    	-0.3411896688232393,	2.150506156415752,	-1.9769070634882338,	-2.536849463761952,	1.5634696220903175,	1.6337930240737129,
    	2.9779059084560746,	4.361048220663376,	-1.4064598405813904,	4.815700729914828,	-6.301679904006761,	4.157380386293454,
    	-0.789285081547633,	-2.7860088773347886,	-3.38799595740029,	1.5522929812230775,	-3.721567914232831,	0.12102079285996481,
    	3.149402344248527,	-1.5103054160289506,	-3.25403590464085,	-3.8443250819718155,	0.323674425382908,	-0.639625866126658,
    	0.8423020388786344,	-6.335501904953268,	-4.859317748119939,	-0.4089386392422738,	2.88877971596525,	-3.271418856304642,
    	2.1342654707075495,	0.07481324859431132,	-0.4935920412650201,	1.536143266804662,	-0.4109851296059759,	6.008732246578375,
    	0.6507028775754552,	-1.7986469588546021,	2.4633842678861155,	-0.11947140804405752,	6.630414414085129,	-5.707705274521347,
    	4.748052526343051,	0.41673668912390377,	-0.587857110661939,	-1.0864007714172181,	-0.5592089717899638,	4.612709189817611,
    	-1.43553567188268,	-0.33988591240121074,	0.3122416784329332,	-3.1579213127584995,	-3.359818654513254,	-3.956062446352468,
    	-1.8463023071257911,	-6.696046069963235,	0.07872095057111506,	-1.526257634694054,	0.27140643668745223,	2.8730580085338477,
    	-4.345111439624589,	-0.28360906201637026,	-1.6465519150914412,	-4.323476115343753,	-1.0444813881184871,	-0.7685109894040542,
    	-3.0147413674733663,	1.2916475674281125,	1.4815876012198426,	-0.7055891197876889,	-1.5488006354208392,	0.5344708552143689,
    	1.6098349664233609,	2.671796584922629,	-0.2598514101600883,	1.5429710168998865,	0.0809435131357335,	3.5597280913277496,
    	7.041226666027763,	-1.1548885289203812,	1.7152059027520732,	0.6065185702856191,	-2.3562724254844643,	-0.2620309482771504,
    	-3.819775510219623,	-4.8645995257974555,	6.657745063221573,	3.3983736686057044,	7.015042168647388,	1.3754803055763651,
    	-2.6844344263084112,	-2.1485602037183984,	0.5187214567399906,	0.7953792236264365,	-2.17771051008687,	-0.49323842183569666,
    	4.4324954353526715,	-4.3264591129602845,	-1.1108185036241214,	0.3357565375160426,	0.8254377826974864,	-5.198042506201834,
    	1.2758842489836741,	3.0352372960464997,	1.4673946138929308,	1.2044093469381885,	2.2508092381155618,	-4.480954582500789,
    	-1.4975100547456979,	-1.3475435456708575,	-3.998159010602512,	-2.3984304372565566,	-0.24346424711917086,	0.9477493928498684,
    	-1.7458668946179605,	0.9025679977094814,	-2.266544174038513,	0.7377758878318884,	-2.360384251761666,	4.259826667024908,
    	-3.691746465097058,	-2.3613397165244594,	1.4776670333734223,	5.00789367484582,	3.4909177608560045,	0.8105634878680468,
    	4.021862447976983,
    };

    // shape is (13, 13)
    [[maybe_unused]] float attention_scaled_mask[]=
    {
    	-0.5961580289418932,	-10000000000.262268,	-9999999999.294664,	-10000000000.627142,	-9999999999.534307,	-10000000000.52997,
    	-9999999998.66256,	-10000000000.302317,	-9999999999.780165,	-9999999999.955664,	-10000000001.05141,	-10000000000.220768,
    	-10000000001.23438,	-0.42181133571063495,	-0.7904494517586589,	-10000000001.917027,	-10000000000.009937,	-9999999997.746378,
    	-9999999998.839764,	-9999999998.565296,	-9999999999.257822,	-10000000000.344442,	-9999999998.687948,	-10000000000.103357,
    	-9999999999.105255,	-9999999998.03111,	-0.046683576166778144,	-0.7174797013471461,	2.167963371968201,	-10000000000.455503,
    	-10000000000.316536,	-10000000002.295876,	-9999999998.912384,	-9999999999.041122,	-10000000000.523758,	-10000000002.130419,
    	-10000000000.017893,	-10000000000.523893,	-10000000002.574303,	1.4951516391006807,	0.2870366182850009,	-1.380673856683454,
    	-0.08809479368336198,	-9999999999.444742,	-10000000000.510435,	-10000000000.655012,	-9999999999.596313,	-9999999999.578156,
    	-9999999999.231108,	-9999999998.873981,	-10000000000.363146,	-9999999998.756592,	-1.6270867547565426,	1.0734309999975935,
    	-0.20379253174959636,	-0.7193443989535772,	-0.874776794668625,	-9999999999.5992,	-10000000000.960905,	-9999999999.968752,
    	-9999999999.186829,	-10000000000.38996,	-10000000000.840189,	-10000000000.992601,	-9999999999.916428,	-0.1651506884874694,
    	0.21748145127023355,	-1.635819557849958,	-1.2546704474932435,	-0.10558750262706995,	0.745879715386347,	-10000000000.844677,
    	-9999999999.448935,	-9999999999.980682,	-10000000000.127445,	-9999999999.603369,	-10000000000.106115,	-9999999998.448551,
    	0.16801076054526543,	-0.46440864782336605,	0.6360430829888184,	-0.030847384913505707,	1.711965640280354,	-1.4737231648854272,
    	1.2259418907629793,	-9999999999.892399,	-10000000000.151785,	-10000000000.280508,	-10000000000.144386,	-9999999998.809004,
    	-10000000000.370653,	-0.08775816522269347,	0.08062045470417346,	-0.8153717768631421,	-0.8675014463471229,	-1.0214509314186453,
    	-0.4767132058375013,	-1.7289116609603488,	0.020325662037303058,	-10000000000.394077,	-9999999999.929924,	-9999999999.258179,
    	-10000000001.121902,	-10000000000.073227,	-0.42513787638767203,	-1.116316732830126,	-0.26968393477376706,	-0.19842868422261908,
    	-0.7784028739564615,	0.33350196785456004,	0.38254427369812305,	-0.18218232734685794,	-0.39989860450535836,	-9999999999.862,
    	-9999999999.584343,	-9999999999.310144,	-10000000000.067093,	0.3983934034755989,	0.02089952522388114,	0.9191178409826058,
    	1.818036907626473,	-0.2981909359489746,	0.4428642597783625,	0.15660242145878234,	-0.6083869242019391,	-0.06765609992455603,
    	-0.9862617958221025,	-10000000001.256035,	-9999999998.280977,	-9999999999.122543,	1.811276099474273,	0.35514748776890337,
    	-0.6931179884718892,	-0.5547558591550248,	0.13393330421829563,	0.2053660324683049,	-0.5622824358951537,	-0.1273536128986188,
    	1.1444654002174193,	-1.1170869395028329,	-0.286812104346351,	-9999999999.913307,	-9999999999.786873,	-1.3421288039598644,
    	0.32943189653347515,	0.7836948999583816,	0.3788796601281189,	0.31097715618054367,	0.5811564463140917,	-1.1569774982091574,
    	-0.38665543352054965,	-0.3479342473781683,	-1.0323202175701767,	-0.6192720760354411,	-0.06286219829929839,	-9999999999.755293,
    	-0.45078089383668135,	0.23304205492990634,	-0.5852191892996403,	0.19049291512041966,	-0.6094485931815458,	1.099882515941181,
    	-0.9532048385160671,	-0.6096952931224916,	0.38153198743299893,	1.2930325868170214,	0.901351090051666,	0.2092865926371162,
    	1.0384404187834566,
    };

    // shape is (13, 13)
    [[maybe_unused]] float attention_softmax[]=
    {
    	1.0,	0.0,	0.0,	0.0,	0.0,	0.0,
    	0.0,	0.0,	0.0,	0.0,	0.0,	0.0,
    	0.0,	0.5911298583132182,	0.4088701416867817,	0.0,	0.0,	0.0,
    	0.0,	0.0,	0.0,	0.0,	0.0,	0.0,
    	0.0,	0.0,	0.09372530994958951,	0.047921878052258574,	0.8583528119981518,	0.0,
    	0.0,	0.0,	0.0,	0.0,	0.0,	0.0,
    	0.0,	0.0,	0.0,	0.6408461257418427,	0.1914591277723517,	0.03612423171762388,
    	0.13157051476818168,	0.0,	0.0,	0.0,	0.0,	0.0,
    	0.0,	0.0,	0.0,	0.0,	0.040586350552464796,	0.6042267682205879,
    	0.16846466757118153,	0.10060220466614926,	0.0861200089896164,	0.0,	0.0,	0.0,
    	0.0,	0.0,	0.0,	0.0,	0.0,	0.15196310024800094,
    	0.2227989665564479,	0.034916826946824915,	0.05111704334425078,	0.16128950464367495,	0.37791455826080067,	0.0,
    	0.0,	0.0,	0.0,	0.0,	0.0,	0.0,
    	0.08543398200882749,	0.04539148484939373,	0.13642517009650407,	0.07002734383453808,	0.40009371900469787,	0.016543776883901733,
    	0.2460845233221371,	0.0,	0.0,	0.0,	0.0,	0.0,
    	0.0,	0.18169494741998546,	0.21501499882637776,	0.08776963132343518,	0.08331144099549449,	0.0714241752915637,
    	0.12314642901818908,	0.035204561842773235,	0.20243381528218105,	0.0,	0.0,	0.0,
    	0.0,	0.0,	0.08845897632303437,	0.044316631865161514,	0.10333672612970934,	0.11096869032026517,
    	0.062132792093424136,	0.1888926273591697,	0.19838727412080376,	0.1127862516626413,	0.09072003012579069,	0.0,
    	0.0,	0.0,	0.0,	0.0902793603891789,	0.06189348903419202,	0.15196242778802402,
    	0.3733634609467829,	0.04498479427889714,	0.09438476981988447,	0.07088918514504755,	0.0329874946451985,	0.056648151897801535,
    	0.022606866054993,	0.0,	0.0,	0.0,	0.3672609758591541,	0.08562215669218937,
    	0.030014440422720768,	0.03446832349110916,	0.06863001022898059,	0.07371178078248207,	0.034209868868630915,	0.052849221870481335,
    	0.18853091560693275,	0.0196427764932505,	0.04505952968406832,	0.0,	0.0,	0.02179583153443826,
    	0.11596395823861136,	0.18264464627302027,	0.12184225320744196,	0.11384349993408628,	0.1491576779635177,	0.026229114735826252,
    	0.056667004375612916,	0.05890425293490804,	0.0297112964066016,	0.04490618183864408,	0.0783342825572913,	0.0,
    	0.03222426527696278,	0.0638503904483775,	0.028170672466213926,	0.061190598457283855,	0.027496316477729325,	0.15192441972931997,
    	0.01949768623865997,	0.02748953397473398,	0.07407163237456418,	0.18429416971736526,	0.12456799764947954,	0.062351469178411764,
    	0.14287084801089786,
    };

    // shape is (13, 17)
    [[maybe_unused]] float attention_projection[]=
    {
    	0.1486981280629037,	1.4460230634069287,	-0.0051374179562285,	-0.15956287406705108,	1.436570669708498,	-0.5795082893291696,
    	-0.16240072992892313,	1.3391315836256719,	1.762714296497166,	-1.0740807545618982,	-0.19163168438548278,	0.4838288394495532,
    	1.2751258373944747,	0.4609946505605043,	-0.5956856051072797,	0.7958769911591804,	-2.233052848438117,	0.14597041628511606,
    	1.158188608649178,	0.45139821034909167,	0.08957400273934778,	0.2643533143990825,	0.042811717340945354,	-0.314805383628312,
    	0.6039426921878943,	1.512818784979983,	-1.4434282629138937,	0.8020374249149499,	0.6540235536131588,	1.1172637477085148,
    	0.2102381804770166,	-0.06402664910414996,	0.7210856784027759,	-1.222033094792522,	-1.2771049154451606,	0.5527968203804647,
    	0.27013338959201516,	0.8141258350878042,	0.42428835471222304,	0.22223002422955573,	0.10395590988015521,	-1.1745286106604074,
    	-1.3720789897085486,	-0.2901250823207824,	-0.5154799651390942,	0.47930470943620196,	-0.8856514958061596,	-0.44910073926230415,
    	0.5840953783878248,	0.10829721977569806,	-0.9473729328090188,	0.14101991530308475,	1.1199940550991736,	0.08287845504757821,
    	-0.06582411232247731,	0.6881180075338273,	-0.20703599956887048,	-0.23528949099464644,	0.6459726598277208,	0.9286340200572938,
    	-1.007570302157928,	0.3208856839869932,	0.4807426926909062,	0.7064019384880136,	0.19059844590233718,	-0.2250792313644572,
    	0.5970729566462332,	-1.432041210752585,	-0.29950302012742,	0.6591052107970544,	0.6569216611351426,	0.41372696241916146,
    	-0.6073162219720838,	0.5873440784224118,	-0.215209661440302,	-0.6358136223639356,	0.23845551305088714,	-1.2317190002828275,
    	1.3744087669017757,	0.5813524228048395,	0.15331835172136365,	-0.26972757357711574,	0.5556004691207683,	0.3079844867185123,
    	-0.035776187669319184,	-0.07417087878998435,	0.5078806475333452,	0.32118705359122757,	-0.08238491409193888,	0.8212649287553501,
    	0.4368556103418483,	-0.396745114991488,	0.17416050666644997,	-0.008977639708524887,	-0.5391716492441461,	0.886996439526585,
    	0.04459061592281677,	-0.08685684081084244,	-0.016723026242590755,	0.3379129521852675,	0.1527848615211276,	-0.08643738784491943,
    	-1.3932423793293218,	0.43396015573145386,	-0.035947848993742454,	0.3297207620980279,	0.3784103328599522,	-0.12296296079317573,
    	0.21203309229800554,	-0.7711017354729084,	-0.11187971704037908,	-0.3368277827071883,	0.6657871806740241,	-0.35467544122108147,
    	-0.033229467709039395,	-0.47810772652587197,	0.02416218415841684,	-0.22865523460392975,	-0.13449425254742925,	-0.29407357624685937,
    	0.5169869242018049,	-0.1108574482417457,	0.2187820798404105,	-0.03600643889065436,	-0.0014444607267574128,	-0.0705932937211339,
    	0.16006083129249377,	0.09788393959039343,	-0.91784410364096,	0.47043394397921123,	0.0202711893899212,	-0.07819846883745663,
    	-0.17600944627774517,	0.1927096196390917,	0.3463954902397084,	-0.3634375335195742,	-0.549590651856054,	0.22702997343374487,
    	-0.3653157391071281,	0.09320878471944402,	0.07588512773418897,	-0.18995064887147325,	-0.38381454899181866,	-0.14116870238617638,
    	-0.5138346543631216,	-0.41860180525747537,	0.31370686284708726,	-0.24589259241624337,	-0.16310352060681152,	-0.22471425849745433,
    	-0.05514865924079751,	0.13942944427633425,	-0.30887854377063995,	-0.2000958925708174,	0.33653889131880127,	-0.34706454701854433,
    	-0.04084928496361009,	0.2358430241151737,	-0.10652996344924738,	-0.29404129462044154,	-0.38735957131254334,	-1.1515714940522088,
    	-0.19176431798803076,	0.3300328104075846,	-0.010771817539083446,	-0.7167975551578608,	-0.2581294294543401,	0.03885497665225605,
    	-0.020152945026730498,	-0.34949788562000583,	-0.36357070487029086,	0.6068684311629184,	0.028167645885390852,	0.08588046728571508,
    	0.34060483912196676,	-0.25137491920659116,	-0.21393327766115885,	0.1504308593441472,	0.59158753236597,	-0.7961935077863357,
    	0.410210716311093,	0.36181338882062986,	0.40538269508106783,	0.046509418777399586,	-0.22836216789594377,	0.3461114261892982,
    	-0.7570475224245979,	-0.41062786089036996,	0.31774041807662085,	-0.015342651988549796,	0.04465888874439741,	0.07425723491437052,
    	0.09065999166290808,	-0.061288587244434804,	-0.45117537737335284,	-0.4204336864807966,	-0.5000048086363436,	0.46353573273394594,
    	0.22852642326961195,	-0.27684871743480255,	-0.32696067162966713,	0.25408749515717327,	0.07012181699670847,	-0.20943295374708076,
    	0.1854310228975223,	0.07744387577849178,	0.06533315223050318,	-0.04404010111153725,	0.03194826071509733,	-0.20986221380086206,
    	-0.06474439907652249,	-0.19018725419468369,	0.09527330165305306,	-0.8529975540380113,	0.7508406734755981,	0.3698328324906503,
    	0.16192504632074844,	-0.05825077460768121,	0.05072245692624566,	-0.2536894069744704,	-0.14438815284665324,
    };

    // shape is (13, 17)
    [[maybe_unused]] float gt[]=
    {
    	0.1486981280629037,	1.4460230634069287,	-0.0051374179562285,	-0.15956287406705108,	1.436570669708498,	-0.5795082893291696,
    	-0.16240072992892313,	1.3391315836256719,	1.762714296497166,	-1.0740807545618982,	-0.19163168438548278,	0.4838288394495532,
    	1.2751258373944747,	0.4609946505605043,	-0.5956856051072797,	0.7958769911591804,	-2.233052848438117,	0.14597041628511606,
    	1.158188608649178,	0.45139821034909167,	0.08957400273934778,	0.2643533143990825,	0.042811717340945354,	-0.314805383628312,
    	0.6039426921878943,	1.512818784979983,	-1.4434282629138937,	0.8020374249149499,	0.6540235536131588,	1.1172637477085148,
    	0.2102381804770166,	-0.06402664910414996,	0.7210856784027759,	-1.222033094792522,	-1.2771049154451606,	0.5527968203804647,
    	0.27013338959201516,	0.8141258350878042,	0.42428835471222304,	0.22223002422955573,	0.10395590988015521,	-1.1745286106604074,
    	-1.3720789897085486,	-0.2901250823207824,	-0.5154799651390942,	0.47930470943620196,	-0.8856514958061596,	-0.44910073926230415,
    	0.5840953783878248,	0.10829721977569806,	-0.9473729328090188,	0.14101991530308475,	1.1199940550991736,	0.08287845504757821,
    	-0.06582411232247731,	0.6881180075338273,	-0.20703599956887048,	-0.23528949099464644,	0.6459726598277208,	0.9286340200572938,
    	-1.007570302157928,	0.3208856839869932,	0.4807426926909062,	0.7064019384880136,	0.19059844590233718,	-0.2250792313644572,
    	0.5970729566462332,	-1.432041210752585,	-0.29950302012742,	0.6591052107970544,	0.6569216611351426,	0.41372696241916146,
    	-0.6073162219720838,	0.5873440784224118,	-0.215209661440302,	-0.6358136223639356,	0.23845551305088714,	-1.2317190002828275,
    	1.3744087669017757,	0.5813524228048395,	0.15331835172136365,	-0.26972757357711574,	0.5556004691207683,	0.3079844867185123,
    	-0.035776187669319184,	-0.07417087878998435,	0.5078806475333452,	0.32118705359122757,	-0.08238491409193888,	0.8212649287553501,
    	0.4368556103418483,	-0.396745114991488,	0.17416050666644997,	-0.008977639708524887,	-0.5391716492441461,	0.886996439526585,
    	0.04459061592281677,	-0.08685684081084244,	-0.016723026242590755,	0.3379129521852675,	0.1527848615211276,	-0.08643738784491943,
    	-1.3932423793293218,	0.43396015573145386,	-0.035947848993742454,	0.3297207620980279,	0.3784103328599522,	-0.12296296079317573,
    	0.21203309229800554,	-0.7711017354729084,	-0.11187971704037908,	-0.3368277827071883,	0.6657871806740241,	-0.35467544122108147,
    	-0.033229467709039395,	-0.47810772652587197,	0.02416218415841684,	-0.22865523460392975,	-0.13449425254742925,	-0.29407357624685937,
    	0.5169869242018049,	-0.1108574482417457,	0.2187820798404105,	-0.03600643889065436,	-0.0014444607267574128,	-0.0705932937211339,
    	0.16006083129249377,	0.09788393959039343,	-0.91784410364096,	0.47043394397921123,	0.0202711893899212,	-0.07819846883745663,
    	-0.17600944627774517,	0.1927096196390917,	0.3463954902397084,	-0.3634375335195742,	-0.549590651856054,	0.22702997343374487,
    	-0.3653157391071281,	0.09320878471944402,	0.07588512773418897,	-0.18995064887147325,	-0.38381454899181866,	-0.14116870238617638,
    	-0.5138346543631216,	-0.41860180525747537,	0.31370686284708726,	-0.24589259241624337,	-0.16310352060681152,	-0.22471425849745433,
    	-0.05514865924079751,	0.13942944427633425,	-0.30887854377063995,	-0.2000958925708174,	0.33653889131880127,	-0.34706454701854433,
    	-0.04084928496361009,	0.2358430241151737,	-0.10652996344924738,	-0.29404129462044154,	-0.38735957131254334,	-1.1515714940522088,
    	-0.19176431798803076,	0.3300328104075846,	-0.010771817539083446,	-0.7167975551578608,	-0.2581294294543401,	0.03885497665225605,
    	-0.020152945026730498,	-0.34949788562000583,	-0.36357070487029086,	0.6068684311629184,	0.028167645885390852,	0.08588046728571508,
    	0.34060483912196676,	-0.25137491920659116,	-0.21393327766115885,	0.1504308593441472,	0.59158753236597,	-0.7961935077863357,
    	0.410210716311093,	0.36181338882062986,	0.40538269508106783,	0.046509418777399586,	-0.22836216789594377,	0.3461114261892982,
    	-0.7570475224245979,	-0.41062786089036996,	0.31774041807662085,	-0.015342651988549796,	0.04465888874439741,	0.07425723491437052,
    	0.09065999166290808,	-0.061288587244434804,	-0.45117537737335284,	-0.4204336864807966,	-0.5000048086363436,	0.46353573273394594,
    	0.22852642326961195,	-0.27684871743480255,	-0.32696067162966713,	0.25408749515717327,	0.07012181699670847,	-0.20943295374708076,
    	0.1854310228975223,	0.07744387577849178,	0.06533315223050318,	-0.04404010111153725,	0.03194826071509733,	-0.20986221380086206,
    	-0.06474439907652249,	-0.19018725419468369,	0.09527330165305306,	-0.8529975540380113,	0.7508406734755981,	0.3698328324906503,
    	0.16192504632074844,	-0.05825077460768121,	0.05072245692624566,	-0.2536894069744704,	-0.14438815284665324,
    };

    // shape is (13, 15)
    [[maybe_unused]] float d_tq[]=
    {
    	-0.712390662050588,	0.753766378659703,	-0.044503078338053455,	0.45181233874578974,	1.3451017084510097,	0.5323378882945463,
    	1.3501878997225267,	0.8612113741693206,	1.4786857374358966,	-1.0453771305385342,	-0.7889890249515489,	-1.261605945319069,
    	0.5628467852810314,	-0.24332625188556253,	0.9137407048596775,	0.31735092273633597,	0.12730328020698067,	2.1503829673811126,
    	0.6062886568962988,	-0.026771649986440726,	-0.9841607817725814,	1.190705272505982,	0.9528306110451948,	-1.0871815907509188,
    	-0.1452113325860197,	0.23785783828501061,	-1.639093411201915,	-0.27813451650877447,	1.3992384201681904,	-1.6151079632521659,
    	0.49087183250951594,	1.8927422187767202,	-0.6209797023155086,	-0.4537523812416193,	0.21745165945019368,	0.5143288551447859,
    	0.3972413264075336,	-1.5128451152607716,	-0.7640339696578627,	0.1012697856437548,	-0.31726597100271015,	1.1383330474921514,
    	-0.3221236569774085,	-0.37011607810475444,	1.8496125662057936,	-0.08165155616424355,	-0.9802743185258672,	-0.002389025600514556,
    	-0.2324258720274623,	0.7109747948581455,	0.4531586132388156,	-0.33267577528589565,	-0.38445768500357647,	-0.530327407582726,
    	1.3503059959308692,	2.0152208250443837,	-0.057532387873538586,	0.5646542880869975,	0.35953229551797367,	-0.511965094194221,
    	-0.5231642148972904,	-0.48137141579474935,	-1.9241594490929896,	-0.757453225587536,	0.45348104026746144,	0.9223441482568703,
    	-0.20734969827652708,	-0.37591996120915105,	0.6398839732802036,	2.0536932362953815,	-0.928335225810039,	-0.7746600331288077,
    	-0.4371456556814915,	0.5479183096073379,	-0.8541429534648363,	0.46565797200403236,	-0.7209996734353815,	0.5622617060581323,
    	-0.3148580761325759,	-0.42989708186880404,	0.8149954430712237,	-1.185418809153406,	-0.7686270207784495,	0.7291654706806434,
    	0.7659110492252689,	0.20780005335113885,	-2.187115273247333,	0.7219650574076287,	0.3487505938690223,	-0.3386182541020083,
    	-1.9453906815188249,	0.8309056578021413,	1.749241790081012,	0.5011372949374676,	1.3303169236725105,	-0.6575172676978901,
    	0.18494595039221876,	-0.6510564789536804,	-1.1205068713840005,	-0.3466067875476209,	-0.911344781357355,	-0.7461352526666984,
    	-0.7709685285300425,	-0.41395371382728313,	-0.2918113875232261,	0.12677274472597902,	-1.247336156644393,	0.31717096074155054,
    	-1.81842142065901,	0.05569219339294779,	0.080866188394344,	-2.319847554388968,	-0.34112117132031883,	1.0898461110532707,
    	0.45053117920927527,	-0.49429864739786794,	-0.5465767417188724,	0.29202648981800744,	0.15420459268584072,	0.8173055190310792,
    	0.14375578038446712,	0.9051715331786675,	0.4006736681660438,	-0.764593898622952,	-1.01700202554413,	-0.5269892532991041,
    	0.40865864630465626,	-0.8677998149261569,	0.8917941660323877,	-1.5390641198540729,	-0.2642474112620271,	-0.9323592637202481,
    	0.31287871866605876,	0.13679032249522113,	-0.15170019141369395,	-1.2368377011017966,	-0.882490391596822,	-0.6606324986051187,
    	0.9929491259546886,	-0.6437970571220203,	0.5011676877836018,	0.9199851312920041,	-0.8444764142538497,	-0.5164428497021851,
    	-2.2068670954994913,	-0.4031289968469349,	0.7133254192320697,	-0.9466364609859,	0.35136152893488404,	0.788570479116926,
    	0.36003178759974036,	-0.43820143730669325,	0.2283352044566521,	-0.6753750576188053,	-0.4192603220087837,	2.386745800499031,
    	-0.5083432327925071,	-0.6862364377992013,	0.1549035962020594,	0.277836156413196,	0.7547050088790299,	-0.7598847788231394,
    	-1.054657177172913,	-0.9776990052219913,	0.747371595544824,	1.075681534164297,	-0.7574771053093196,	-0.5108262341824535,
    	-0.025645745936127125,	-0.9590398006025731,	-0.4979713554723796,	0.9384604688575977,	-1.2231117488973633,	0.3685950822771125,
    	1.1790347434852069,	-1.0760914658121563,	1.0942750991001489,	0.004613901175436712,	-0.5351833981652417,	0.8683775099508642,
    	0.07299114797407072,	-0.24006071463865328,	-0.5778416564893406,	-0.8416715368715004,	-0.5534026175135522,	-0.24150531238917375,
    	0.7622308225590424,	-0.25251758817880776,	0.29164051076463826,	-0.3896555250433101,	0.47518769779384906,	-0.07236315822392445,
    	1.2690580976875576,	0.2572490050399525,	-1.6642564360602479,
    };

    // shape is (13, 15)
    [[maybe_unused]] float d_tk[]=
    {
    	-0.1217464334245328,	-0.44151223119024857,	0.9552466924396553,	0.01997092245414585,	0.8628168935353443,	0.7435452461317569,
    	-0.6625811650314397,	-0.3051810275934747,	0.041447402165011596,	-1.0669628403806435,	3.4011057189806557,	0.5253295997343811,
    	-1.2768778466596493,	-1.4141544662915098,	0.12673002941390707,	-0.3848490502028002,	-1.5509569211976624,	-1.7207954551375861,
    	0.017982138806439393,	-0.4654414831035872,	0.4004773841168456,	-0.11723949907063196,	1.100836264556602,	-1.830354023073999,
    	0.6019355688249879,	-0.8135337616981977,	-0.935955221598663,	0.9630313786660784,	-1.9278777352364316,	-0.08809626952746996,
    	-0.36184796336959735,	1.1950253748410917,	-1.4615058094923734,	0.35021313455542263,	0.47347563519013397,	-0.018421809137576996,
    	0.10767464455391365,	-0.15579970816999872,	-1.4634777566193453,	-0.5334945027167617,	-1.9323560015091858,	0.9399792195744116,
    	0.9506069961060607,	-1.41124263188717,	1.2047437309657716,	-0.17830606215500272,	-1.5232135953524801,	-0.7320583853311774,
    	0.5751509658063075,	-0.10321862725194576,	-1.0790126956031663,	0.6558615323233493,	-0.22467908543381457,	-0.5136792046662925,
    	-1.257253734973249,	0.5022857779517222,	0.839193535320577,	-1.055658897711565,	-0.308375994286531,	-0.28410380510716565,
    	0.15003366373872756,	-1.0942595125916925,	1.982925182595953,	0.2676098967321498,	0.8784598910238555,	-0.33240947257420594,
    	2.0921686012929617,	0.44080738731842145,	-1.812024553173441,	0.9962094669993536,	-0.04995254339686802,	-1.0670398044692198,
    	-0.059260833836361,	-0.9041049314828216,	0.7027533792191322,	0.6085923102841594,	-1.393525225748726,	0.6586660668217251,
    	1.0461644964376338,	-2.4957528823141972,	1.3043577797963937,	0.6609032917196628,	0.8224723016354284,	1.4267264504853654,
    	-0.048697610640109204,	-0.8670155109522104,	0.3711592574799248,	-0.16127395672832917,	0.44951651436299817,	-2.3269262116155196,
    	-0.4158113660344991,	2.5920840674002923,	0.37476811336909965,	0.9995238627794718,	0.33912166991349607,	-0.34921913811594835,
    	0.843842055045005,	0.7419006788952703,	-0.3519968223496652,	-0.19049578315740576,	0.6722721224549026,	-0.7996871994715307,
    	-0.8446924471317016,	1.1722774270578469,	1.1490631884147902,	-0.19380492240705713,	1.5666402273744786,	0.8579051809684236,
    	0.40976548500449694,	-0.5090008972919337,	0.7718870234868406,	-0.5338435604421259,	-0.07115124422410989,	-0.6191847305942658,
    	1.2125948226530212,	0.1406138453747909,	0.019892207578106016,	0.6793203874098627,	2.4897075316883863,	1.0525318150652148,
    	-1.5340965702283584,	0.015174920002897885,	0.117052488790469,	-0.557684623006479,	-0.9570655488171265,	1.717175959710519,
    	0.7778483925521874,	0.946348007431648,	0.39254793773535773,	0.8624014081512591,	0.5232452162507775,	0.20156945957526934,
    	-0.1650355232064671,	-0.9626608287669728,	-0.7013680441849037,	-0.030667987475674858,	-0.8814625655313383,	0.3532107803002698,
    	-0.8912937922772961,	0.7909698216528814,	-1.391766084294036,	0.3303022703226134,	1.2795087343766578,	0.8672126651998152,
    	-0.5209590207114089,	1.3204048067975092,	0.04066528515761626,	0.3650517980270081,	-0.04520588860236072,	-1.7385178655572635,
    	0.5704933453211514,	0.4349760219816502,	0.7588944642911625,	-0.9738381295272108,	0.8048850955179162,	-0.644619180730499,
    	-1.1702444249284003,	-1.0259108732843625,	0.20581615250656984,	-0.404357671797137,	1.5418077084686126,	0.8001528559058803,
    	1.147309002011903,	-0.16224855290821893,	-1.0747468405930436,	-1.6003841181528236,	-0.22205672759554324,	1.2750575903893269,
    	1.2567430683363894,	-1.6495595298995447,	0.2754714469462586,	0.7085245177855963,	-0.9824603525381572,	0.017446916303085782,
    	-0.7200583582807348,	0.15647674659188437,	0.20425480457441636,	-0.3236790297502634,	-0.11145904212515595,	-0.6779772552780428,
    	0.1106188085540423,	-3.210430797268423,	0.4777660895774647,	1.0369968080399357,	-1.3189645536049885,	-0.19422133384463808,
    	0.5877885293013254,	-0.7068190812919728,	-0.6861465851073458,	-0.06434314961587694,	0.7387144235342626,	-1.3825474846365924,
    	0.23254262542334578,	0.5319765561553507,	-1.3703642428009506,
    };

    // shape is (13, 17)
    [[maybe_unused]] float d_tv[]=
    {
    	0.1486981280629037,	1.4460230634069287,	-0.0051374179562285,	-0.15956287406705108,	1.436570669708498,	-0.5795082893291696,
    	-0.16240072992892313,	1.3391315836256719,	1.762714296497166,	-1.0740807545618982,	-0.19163168438548278,	0.4838288394495532,
    	1.2751258373944747,	0.4609946505605043,	-0.5956856051072797,	0.7958769911591804,	-2.233052848438117,	0.14202678794857176,
    	0.7420478267454842,	1.1114411280385852,	0.44976720755395455,	-1.430396701578305,	0.9425397724411325,	-0.5351465926388975,
    	-0.45896716822932365,	1.1515287731761275,	-1.9774177085108056,	2.2386487591829325,	0.9000849971172967,	0.8890323725239309,
    	-0.1522965744944709,	0.704626894626221,	0.6129550677145995,	0.23966318182680454,	-1.5120214739877849,	0.4446977161029871,
    	0.2532205247585785,	0.9407870683897978,	0.41730231988738953,	0.26955855330903167,	0.1687210624313978,	-1.4889501692875957,
    	-1.8552666603782968,	-0.11032172211623395,	-0.7046047621786199,	0.4553185270261121,	-1.220671903715444,	-0.565046518309226,
    	0.7061888116045446,	0.005043898069076927,	-0.8732591793851744,	0.5560180812944249,	0.2673819755787573,	-1.0319348271510602,
    	-0.6359031894503993,	0.1997790971313106,	-0.19651664300377936,	-0.2648908941443411,	-0.5361687068333683,	-2.6939563172399312,
    	0.4813399523048028,	0.3080908624754918,	-0.13752826816152028,	-1.8003657016978516,	-0.41998581004543123,	-0.0285337341330573,
    	-0.23181317888607783,	-0.1165749870881772,	-2.236053777261358,	0.5833276501518617,	0.542544026931158,	0.6261672937955176,
    	1.2571257313814808,	0.18248209713031777,	1.3116180465656415,	-1.2548603914775587,	0.6350869000947228,	-0.2688785015946833,
    	1.3613709645591259,	-0.5226347611202428,	-0.5672394268341736,	-0.6847900441875194,	0.4403789105928676,	-0.8384801402483328,
    	0.7998766732572836,	0.6790266830356967,	-0.0012405611429540924,	0.08134367638997905,	-0.6871472120485602,	1.8366742207237288,
    	0.7571112741306737,	-1.2085723310812708,	0.9386034698816718,	-1.1466886495167958,	0.23082017281823,	0.5467579110457865,
    	-0.407616632784632,	-0.668307894421939,	0.2614395928205015,	0.46893464927171374,	0.11163027156564224,	0.2829905076675621,
    	-1.469634625209567,	-0.1463693097462854,	-1.0835914658230943,	-0.0001493927548569246,	-1.1527257504517598,	-0.913446285204153,
    	-1.0526677980507988,	-0.5586105579100125,	-0.43933304497920833,	-0.28528895825565204,	0.41194412664756364,	-1.1114344807014709,
    	1.4145035166500803,	-0.5462647972204403,	-0.9558693649714395,	0.10035528320794392,	-0.6176890658642286,	-0.6788185843968384,
    	-0.014644605617744137,	-1.4610908390266266,	0.7971698357380883,	-1.5715038322410304,	-0.8902148949351755,	0.8568100979970842,
    	0.9113751643607431,	0.14128931736396996,	-1.6158841884561967,	-0.5877654187602196,	-0.805312334511712,	-0.8441837586351917,
    	-0.5260683585992184,	0.16935027581489936,	0.6494568961145271,	0.03403109657625523,	-1.0549422538411257,	-0.16218430250232538,
    	0.054184073949185126,	0.6806681362126462,	-0.7959296749570147,	-0.6529646317490029,	-0.8242714579780461,	-1.2921348859801496,
    	0.20096395382796983,	-0.7277052749006709,	0.736058077736053,	0.6660163620525159,	-0.14659250672563906,	0.07848761359327407,
    	-0.5390081340365667,	0.05420653128649276,	-0.03571540661693727,	1.0039302837615773,	-0.5936396768276437,	0.9410600435834002,
    	0.5317235659329966,	0.23958160437685797,	-1.2695289278830393,	-1.1003869388765317,	-0.12411569290993438,	0.023663980551364306,
    	-1.3401742611452505,	1.6128983277731723,	0.07097352721985194,	-0.21141560941821472,	0.44732515056190925,	-0.8405369997384325,
    	-2.1490883096447986,	-0.19948202611107752,	-0.7855491842631765,	-0.04931982755314424,	0.11573722731850587,	-1.6326461309487728,
    	-0.8378122948744013,	0.8309427428150207,	1.3850943853355704,	-0.11066443898983998,	0.0955208342642143,	0.3200664601730596,
    	0.2713309790029247,	2.6113654665010273,	1.6834959298134051,	-0.6144833956432614,	-0.2894620115384941,	1.1701766375339,
    	0.5992695241557091,	0.9696621383994227,	0.6922148845648759,	-0.812720293514382,	-1.2115714159878144,	-1.4314687103196855,
    	-0.6603821943150443,	0.44452754755268203,	-0.5007806561331115,	2.0441428809847904,	-1.5605053092527066,	-0.23276339561067175,
    	1.0682009090489417,	1.900647880486306,	-0.9215710802026723,	0.3187895017249588,	0.7212519457943586,	-1.9497509961006325,
    	0.8140013785687984,	0.1846723505590757,	-0.39635786305557547,	0.9838660218731716,	0.09450087503790537,	-0.7328151697091639,
    	1.3594580312665707,	-0.6155332630831808,	1.2940562967121043,	-2.364696080657899,	0.921435479755555,	-0.2928802097119344,
    	0.3008427164039561,	0.19745598324840175,	1.0388537536498363,	-0.81775460145923,	-0.16482422338366787,
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


        {
            auto mat = view_2d{ reinterpret_cast<float*>(results[0].data()), static_cast<std::uint64_t>(n_q), static_cast<std::uint64_t>(d_v)};
            auto nat = view_2d{ gt, static_cast<std::uint64_t>(n_q), static_cast<std::uint64_t>(d_v)};
            for ( auto idx : range( n_q ) )
                for ( auto jdx : range( d_v ) )
                {
                    CHECK( std::abs( mat[idx][jdx] - nat[idx][jdx] ) < 1.0e-2 );
                    if ( std::abs( mat[idx][jdx] - nat[idx][jdx] ) > 1.0e-2 )
                        spdlog::error( "Element mismatch at [{}][{}] : {} v.s. {}", idx, jdx, mat[idx][jdx], nat[idx][jdx] );
                }
        }

    }

    REQUIRE( true );
}

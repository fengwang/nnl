#include "../../include/direct_space/graph.hpp"
#include "../../include/direct_space/node.hpp"

TEST_CASE( "alexnet", "[alexnet]" )
{
    using namespace nnl;

    graph<node> g;

    auto conv1 = make_node( "conv1" );
    auto conv2 = make_node( "conv2" );
    auto conv3 = make_node( "conv3" );
    auto conv4 = make_node( "conv4" );
    auto conv5 = make_node( "conv5" );
    auto dropout_1 = make_node( "drop_out_1" );
    auto dropout_2 = make_node( "dropout_2" );
    auto fc_1 = make_node( "fc_1" );
    auto fc_2 = make_node( "fc_2" );
    auto fc_3 = make_node( "fc_3" );
    auto input = make_node( "input" );
    auto max_pool_1 = make_node( "max_pool_1" );
    auto max_pool_2 = make_node( "max_pool_2" );
    auto max_pool_3 = make_node( "max_pool_3" );
    auto relu1 = make_node( "relu1" );
    auto relu2 = make_node( "relu2" );
    auto relu3 = make_node( "relu3" );
    auto relu4 = make_node( "relu4" );
    auto relu5 = make_node( "relu5" );
    auto relu6 = make_node( "relu6" );
    auto relu7 = make_node( "relu7" );
    auto softmax = make_node( "softmax" );
    g.connect( conv1, relu1 );
    g.connect( conv2, relu2 );
    g.connect( conv3, relu3 );
    g.connect( conv4, relu4 );
    g.connect( conv5, relu5 );
    g.connect( dropout_1, fc_1 );
    g.connect( dropout_2, fc_2 );
    g.connect( fc_1, relu6 );
    g.connect( fc_2, relu7 );
    g.connect( fc_3, softmax );
    g.connect( input, conv1 );
    g.connect( max_pool_1, conv2 );
    g.connect( max_pool_2, conv3 );
    g.connect( max_pool_3, dropout_1 );
    g.connect( relu1, max_pool_1 );
    g.connect( relu2, max_pool_2 );
    g.connect( relu3, conv4 );
    g.connect( relu4, conv5 );
    g.connect( relu5, max_pool_3 );
    g.connect( relu6, dropout_2 );
    g.connect( relu7, fc_3 );

    std::vector<node> ordered_nodes = g.computation_order();

    std::cout << "AlexNet computation order:\n";
    for ( auto const& nd : ordered_nodes )
        std::cout << nd << std::endl;

    REQUIRE(true);
}






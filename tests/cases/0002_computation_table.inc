#include "../../include/direct_space/graph.hpp"
#include "../../include/direct_space/node.hpp"
#include "../../include/direct_space/computation_table.hpp"

TEST_CASE( "unet", "[unet]" )
{
    using namespace nnl;

    graph<node> g;

    auto l0 = make_node( "Input", "Input-l0" );
    auto l1 = make_node( "l1" );
    auto l2 = make_node( "l2" );
    auto l3 = make_node( "l3" );
    auto l4 = make_node( "l4" );
    auto l5 = make_node( "l5" );

    g.connect( l0, l1 );
    g.connect( l1, l2 );
    g.connect( {l1, l2}, l3 );
    g.connect( {l2, l3}, l4 );
    g.connect( {l0, l4}, l5 );

    std::vector<node> ordered_nodes = g.computation_order();

    std::cout << "UNet computation order:\n";
    for ( auto const& nd : ordered_nodes )
        std::cout << nd << std::endl;

    auto const& computation_table = make_computation_table( ordered_nodes, g.edges() );
    for ( auto const& ct : computation_table )
        std::cout << ct << std::endl;

    REQUIRE(true);
}



TEST_CASE( "unet-2", "[unet-2]" )
{
    using namespace nnl;

    graph<node> g;

    auto l0 = make_node( "Input", "Input-l0" );
    auto l1 = make_node( "l1" );
    auto l2 = make_node( "l2" );
    auto l3 = make_node( "l3" );
    auto l4 = make_node( "l4" );
    auto l5 = make_node( "l5" );

    g.connect( l0, l1 );
    g.connect( l1, l2 );
    g.connect( {l0, l1, l2}, l3 );
    g.connect( {l2, l3}, l4 );
    g.connect( {l0, l3, l4}, l5 );

    std::vector<node> ordered_nodes = g.computation_order();

    std::cout << "UNet computation order:\n";
    for ( auto const& nd : ordered_nodes )
        std::cout << nd << std::endl;

    auto const& computation_table = make_computation_table( ordered_nodes, g.edges() );
    for ( auto const& ct : computation_table )
        std::cout << ct << std::endl;

    REQUIRE(true);
}

TEST_CASE( "unet-3", "[unet-3]" )
{
    using namespace nnl;

    graph<node> g;

    auto l0 = make_node( "Input", "Input-l0" );
    auto l1 = make_node( "l1" );
    auto l2 = make_node( "l2" );
    auto l3 = make_node( "l3" );
    auto l4 = make_node( "l4" );
    auto l5 = make_node( "l5" );
    auto lx0 = make_node( "Input", "lx0" );
    auto lx1 = make_node( "Input", "lx1" );
    auto lx2 = make_node( "Input", "lx2" );
    auto lx3 = make_node( "Input", "lx3" );
    auto lx4 = make_node( "Input", "lx4" );
    auto lx5 = make_node( "Input", "lx5" );

    g.connect( {l0, lx0}, l1 );
    g.connect( {lx2, l1}, l2 );
    g.connect( {lx3, l0, l1, l2}, l3 );
    g.connect( {lx4, l2, l3}, l4 );
    g.connect( {lx5, l0, l3, l4}, l5 );

    std::vector<node> ordered_nodes = g.computation_order();

    std::cout << "UNet computation order:\n";
    for ( auto const& nd : ordered_nodes )
        std::cout << nd << std::endl;

    auto const& computation_table = make_computation_table( ordered_nodes, g.edges() );
    for ( auto const& ct : computation_table )
        std::cout << ct << std::endl;

    REQUIRE(true);
}


TEST_CASE( "unet-4", "[unet-4]" )
{
    using namespace nnl;

    graph<node> g;

    auto l0 = make_node( "Input", "Input-l0" );
    auto l1 = make_node( "l1" );
    auto l2 = make_node( "l2" );
    auto l3 = make_node( "l3" );
    auto l4 = make_node( "l4" );
    auto l41 = make_node( "l4.1" );
    auto l42 = make_node( "l4.2" );
    auto l43 = make_node( "l4.3" );
    auto l44 = make_node( "l4.4" );
    auto l45 = make_node( "l4.5" );
    auto l46 = make_node( "l4.6" );
    auto l47 = make_node( "l4.7" );
    auto l48 = make_node( "l4.8" );
    auto l49 = make_node( "l4.9" );
    auto l5 = make_node( "l5" );
    auto l6 = make_node( "l6" );
    auto l7 = make_node( "l7" );
    auto l8 = make_node( "l8" );
    auto l9 = make_node( "l9" );

    l45.set_output_node();
    l7.set_output_node();
    l9.set_output_node();

    g.connect( l0, l1 );
    g.connect( l1, l2 );
    g.connect( {l0, l1, l2}, l3 );
    g.connect( {l2, l3}, l4 );
    g.connect( l4, l41 );
    g.connect( l41, l42 );
    g.connect( l42, l43 );
    g.connect( l43, l44 );
    g.connect( l44, l45 );
    g.connect( l45, l46 );
    g.connect( l46, l47 );
    g.connect( l47, l48 );
    g.connect( l48, l49 );
    g.connect( {l0, l3, l49}, l5 );
    g.connect( {l43, l45, l5}, l6 );
    g.connect( l6, l7 );
    g.connect( l7, l8 );
    g.connect( l8, l9 );

    std::vector<node> ordered_nodes = g.computation_order();

    std::cout << "UNet computation order:\n";
    for ( auto const& nd : ordered_nodes )
        std::cout << nd << std::endl;

    auto const& computation_table = make_computation_table( ordered_nodes, g.edges() );
    for ( auto const& ct : computation_table )
        std::cout << ct << std::endl;

    REQUIRE(true);
}






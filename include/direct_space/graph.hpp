#ifndef QGDLIJDFDHYJSQSCSHSWIEGCMSYUCBNKMOKXUHILCWFBFSJKBSEXVYQGSNAJOUWREBDQLARHF
#define QGDLIJDFDHYJSQSCSHSWIEGCMSYUCBNKMOKXUHILCWFBFSJKBSEXVYQGSNAJOUWREBDQLARHF

#include "../utility/utility.hpp"
//#include "./attribute.hpp"

namespace nnl
{

//---------------------------------------------------------------
// Begin of DAG
//---------------------------------------------------------------

template <typename Node>
struct graph
{
    void connect(Node a, Node b)
    {
        better_assert( a != b, "Cannot connect a node to itself, or there will be a cycle in the graph." );
        std::vector<Node>& input_nodes_of_node_b = edges_[b];
        if ( std::find( input_nodes_of_node_b.begin(), input_nodes_of_node_b.end(), a ) == input_nodes_of_node_b.end() )
            input_nodes_of_node_b.push_back( a );
    }

    void connect( std::initializer_list<Node> a, Node b )
    {
        for ( auto _a : a ) connect( _a, b );
    }

    void connect( std::vector<Node> const& a, Node b )
    {
        for ( auto _a : a ) connect( _a, b );
    }

    std::vector<Node> computation_order()
    {
        if ( computation_order_.empty() )
            topological_sort();
        return computation_order_;
    }

    std::map<Node, std::vector<Node>> edges() const
    {
        return edges_;
    }

    void inference_io_shapes(); //<-- implemented in "session.hpp"

private:
    void topological_sort()
    {
        std::unordered_set<Node> visited;
        for (auto const& v : edges_)
            if (visited.find(v.first) == visited.end())
                dfs(visited, v.first);
    }

    void dfs(std::unordered_set<Node>& visited, const Node& v)
    {
        better_assert( (visited.find(v) == visited.end()), format("Found a cycle in the graph with node {}.", v) );
        visited.insert(v);
        for (auto const& e : edges_[v])
            if (visited.find(e) == visited.end())
                dfs(visited, e);
        computation_order_.push_back(v);
    }

    std::map<Node, std::vector<Node>> edges_;
    std::vector<Node> computation_order_;
};//struct graph

//---------------------------------------------------------------
// End of DAG
//---------------------------------------------------------------

template <typename Node>
inline std::ostream& operator << (std::ostream& os, graph<Node> const& g )
{
    for ( auto const& [n, vn] : g.edges() )
    {
        os << n << ": [";
        for ( auto const& n_ : vn )
            os << n_ << ",";
        os << "]\n";
    }
    return os;
}

}//namespace nnl

#endif//QGDLIJDFDHYJSQSCSHSWIEGCMSYUCBNKMOKXUHILCWFBFSJKBSEXVYQGSNAJOUWREBDQLARHF


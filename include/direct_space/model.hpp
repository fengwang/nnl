#ifndef VOIEFGHWCTIFKWEDGGUTFHQWCLFCCTQCUDMBVIFNNHYCXJYRFLYKNTYTTEQEKYRNFBGHYTENA
#define VOIEFGHWCTIFKWEDGGUTFHQWCLFCCTQCUDMBVIFNNHYCXJYRFLYKNTYTTEQEKYRNFBGHYTENA

#include "../utility/utility.hpp"

#include "./layer.hpp"
#include "./graph.hpp"
#include "./session.hpp"
#include "./computation_table.hpp"

extern "C" void cuda_device_synchronize(); // <- implemented in 'src/cuda/device_management.cpp'

namespace nnl
{
    struct model
    {
        std::vector<layer>  input_layers_;
        std::vector<layer>  output_layers_;
        graph<node>         graph_;

        std::vector<layer>& input_layers() { return input_layers_; }
        std::vector<layer> const& input_layers() const { return input_layers_; }

        std::vector<layer>& output_layers() { return output_layers_; }
        std::vector<layer> const& output_layers() const { return output_layers_; }

        //graph<node>& graph() { return graph_; }
        //graph<node> const & graph() const { return graph_; }

        model( layer input, layer output ) : input_layers_{ {input, } }, output_layers_{ {output, } }
        {
            initialize_connections();
        }
        model( layer input, std::vector<layer> const& outputs ) : input_layers_{ {input, } }, output_layers_{ outputs }
        {
            initialize_connections();
        }
        model( std::vector<layer> const& inputs, layer output ) : input_layers_{ inputs }, output_layers_{ {output, } }
        {
            initialize_connections();
        }
        model( std::vector<layer> const& inputs, std::vector<layer> const& outputs ) : input_layers_{ inputs }, output_layers_{ outputs }
        {
            initialize_connections();
        }

        void initialize_connections()
        {
            if ( 0 == output_layers_.size() ) // <- numbers of input layers can be zero, but not the output layers.
            {
                spdlog::error( "Output layers are empty." );
                std::abort();
            }

            std::stack<layer> hks;
            {
                for ( auto l : output_layers_ )
                    hks.push( l );
            }

            std::set<layer> processed_layers;

            while ( !hks.empty() )
            {
                auto l = hks.top();
                hks.pop();

                auto const& ils = l.input_layers();
                if ( ils.size() <= 0 )
                {
                    spdlog::info( format( "Skipping node {} as it does not have any input layers.", l.underlying_node() ) );
                    continue;
                }

                for ( auto il : ils )
                {
                    if ( processed_layers.find( il ) != processed_layers.end() )
                    {
                        spdlog::info( format( "Skipping processed node: {} -> {}", il.underlying_node(), l.underlying_node() ) );
                        //better_assert( false, format( "Not expecting node {} to be referenced more than once in a acyclic graph. Debug please.", il.underlying_node() ) );
                        continue;
                    }
                    spdlog::info( format( "Connecting node {} -> {}", il.underlying_node(), l.underlying_node() ) );
                    graph_.connect( il.underlying_node(), l.underlying_node() );
                    hks.push( il );
                }
                processed_layers.insert( l ); // <- this layer is already processed.
            }

            // mark output nodes
            for ( auto l : output_layers_ )
            {
                auto n = l.underlying_node();
                spdlog::info( format("Mark node {} as output node.", n) );
                n.set_output_node();
            }

            spdlog::info( format("After initialization, the graph inside the model:\n{}\n", graph_) );
        }

        std::vector<tensor<default_engine_type>> predict( tensor<default_engine_type> input_tensor )
        {
            return predict( std::vector<tensor<default_engine_type>>{ {input_tensor,} } );
        }

        std::vector<tensor<default_engine_type>> predict( std::vector<tensor<default_engine_type>> const& input_tensors )
        {
            spdlog::info( "Executing prediction with {} input tensors", input_tensors.size() );

            auto& sess = get_default_session<default_engine_type>();

            if ( input_tensors.size() != input_layers_.size() )
            {
                spdlog::error( "Error model expects {} inputs, but received {}.", input_layers_.size(), input_tensors.size()  );
                std::abort();
            }

            // bind tensors to input node
            for ( auto idx : range( input_layers_.size() ) )
            {
                node n = input_layers_[idx].underlying_node();
                n.add_input( input_tensors[idx] );
                spdlog::info( nnl::format("Bind tensor {} to node {} ", input_tensors[idx], n) );
                {
                    auto tsor = input_tensors[idx];
                    sess.register_tensor( tsor.name(), tsor ); // <-- input tensor might having been already cleared in the previous runs.
                }
            }

            // topological sort of the computation order
            auto computation_table = make_computation_table( graph_ );
            spdlog::info( nnl::format("Calculated computation table:\n{}\n", computation_table) );

            // inference hidden tensor dimensions between layers
            graph_.inference_io_shapes();
            spdlog::info( nnl::format("After inference_io_shapes, the graph inside the model:\n{}\n", graph_) );

            if constexpr( debug_mode )
            {
                std::cout << "\nSession before prediction:\n" << sess << std::endl;
            }

            if constexpr( debug_mode )
            {
                std::cout << "Computation Table:\n";
                for ( auto & ct : computation_table )
                    std::cout << "\n" << ct << std::endl;
            }

            //iterate the computation table
            for ( auto & ct : computation_table )
            {
                auto& [ao, host2device_operations, device2host_operations, device2host_dismiss_operations, wp, wd, op] = ct;
                {
#if 0
                    if ( ao ) ao.arithmetic_operation();

                    for ( auto h2do : host2device_operations )
                        if (h2do) h2do.host2device_operation();

                    for ( auto d2ho : device2host_operations )
                        if (d2ho) d2ho.device2host_operation();

                    for ( auto d2hdo : device2host_dismiss_operations )
                        if (d2hdo) d2hdo.device2host_dismiss_operation();

                    if (wp) wp.weight_preloading();

                    if (wd) wd.weight_dismissing();

                    if (op) op.output_preallocating();
#else
                    //std::cout << "executing computation table:\n" << ct << std::endl;


                    for ( auto h2do : host2device_operations )
                        if (h2do) h2do.host2device_operation();

                    if (wp) wp.weight_preloading();
                    if ( ao ) ao.arithmetic_operation();

                    for ( auto d2ho : device2host_operations )
                        if (d2ho) d2ho.device2host_operation();

                    if (op) op.output_preallocating();

                    if (wd) wd.weight_dismissing();

                    for ( auto d2hdo : device2host_dismiss_operations )
                        if (d2hdo) d2hdo.device2host_dismiss_operation();

                    // Test: if this will make any difference
                    //std::this_thread::sleep_for(std::chrono::milliseconds(1));
#endif
                }

                //sess.synchronize();
                cuda_device_synchronize();
            }

            if constexpr( debug_mode )
            {
                std::cout << "\nSession After prediction:\n" << sess << std::endl;
            }

            // retrieve output tensors
            std::vector<tensor<default_engine_type>> output_tensors;
            {
                //output_tensors.resize( output_layers_.size() );

                for ( auto idx : range( output_layers_.size() ) )
                {
                    auto outputs = output_layers_[idx].underlying_node().outputs();
                    #if 0
                    if ( 1 != outputs.size() )
                    {
                        spdlog::error( "This layer has {} output, only one is expected.", outputs.size() );
                        std::abort();
                    }
                    #endif
                    better_assert( outputs.size() > 0 );
                    for ( auto jdx : range( outputs.size() ) )
                    {
                        output_tensors.push_back( sess.find_tensor_by_name( outputs[jdx] ) );
                        output_tensors.back().synchronize_to_host();
                    }
                }

                //sess.synchronize(); //waiting synchronize_to_host to finish
            }

            // output tensors are synchronized to host
            cuda_device_synchronize();

            // TODO: clean all device memory
            sess.device_clean();

            return output_tensors;
        }

    }; // struct model

    inline model make_model( layer input_layer, layer output_layer )
    {
        return model{ input_layer, output_layer };
    }

}

#endif//VOIEFGHWCTIFKWEDGGUTFHQWCLFCCTQCUDMBVIFNNHYCXJYRFLYKNTYTTEQEKYRNFBGHYTENA


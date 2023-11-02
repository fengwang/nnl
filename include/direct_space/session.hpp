#ifndef TLVVYPKNUGQHHYILQSAHOOQYRUBYHJUVYMCSGDJDHTYUHIYXTHOILSAWMJTGJHERXVHMRAMSY
#define TLVVYPKNUGQHHYILQSAHOOQYRUBYHJUVYMCSGDJDHTYUHIYXTHOILSAWMJTGJHERXVHMRAMSY

#include "../utility/utility.hpp"
#include "./session_context.hpp"
#include "./tensor.hpp"
#include "./node.hpp"
#include "./engine.hpp"
#include "./device.hpp"
#include "./graph.hpp"

namespace nnl
{
    template< Engine E >
    struct device_memory_manager;

    template< Engine E >
    struct session
    {
        device_memory_manager<E>                                    device_memory_manager_;

        // <tensor-id, tensor>, all tensors are registered here
        std::unordered_map<std::int64_t, tensor<E>>                 tensor_id_records_;

        // <tensor-name, tensor>, all named tensors are registered here
        std::unordered_map<std::string, tensor<E>>                  tensor_name_records_;

        // <device-memory, tensor-id>
        std::unordered_map<device_memory, std::int64_t>             device_memory_tensor_records_;
        // <tensor-id, device-memory>
        std::unordered_map<std::int64_t, device_memory>             tensor_device_memory_records_;

        device_memory_manager<E>& dm() { return device_memory_manager_; }

        tensor<E> find_tensor_by_name( std::string const& name )
        {
            auto itor = tensor_name_records_.find( name );
            if ( itor == tensor_name_records_.end() )
            {
                better_assert( false, nnl::format("Failed to find tensor {} from session.", name) );
            }
            //better_assert( itor != tensor_name_records_.end() );
            return itor->second;
        }

        // recall all device memory
        void device_clean()
        {
            device_memory_manager_.clear();
            device_memory_tensor_records_.clear();
            tensor_device_memory_records_.clear();
            clean(); // <-- also clean idle matrices
        }

        // use with care: if tensors are appended in a local scope, they might get removed.
        void clean()
        {
            std::vector<std::int64_t> ids;
            std::vector<std::string> names;

            for ( auto const& [id, tsor] : tensor_id_records_ )
            {
                if (tsor.use_count()<=2)
                {
                    ids.push_back( id );
                    names.push_back( tsor.name() );
                }
            }

            for ( auto id : ids ) // <-- clean tensor mapped device memories before cleaning tensors
            {
                if ( tensor_device_memory_records_.find(id) != tensor_device_memory_records_.end() )
                {
                    spdlog::info( "session::cleaning device memorytensor with id {}", id );
                    tensor_dismiss_device( id ); // <-- also clean device memory whe
                }
            }

            // <-- clean tensors
            for ( auto id : ids )
            {
                spdlog::info( "session::cleaning tensor with id {}", id );
                tensor_id_records_.erase( tensor_id_records_.find( id ) );
            }

            for ( auto const& name : names )
            {
                spdlog::info( "session::cleaning tensor with name {}", name );
                tensor_name_records_.erase( tensor_name_records_.find( name ) );
            }

        }

        // aggressive clean in the inference time, for dismissing operations
        void clean( tensor<E> tsor )
        {
            if ( tsor.use_count() < 2 )
            {
                spdlog::warn( "tensor {} is used outside the session, it will be kept alive.", tsor.name() );
                return;
            }

            auto const id = tsor.id();
            auto const& name = tsor.name();

            if ( tensor_device_memory_records_.find(id) != tensor_device_memory_records_.end() )
            {
                spdlog::info( "session::cleaning device memorytensor with id {}", id );
                tensor_dismiss_device( id ); // <-- also clean device memory whe
            }

            spdlog::info( "session::cleaning tensor with id {}", id );
            tensor_id_records_.erase( tensor_id_records_.find( id ) );

            spdlog::info( "session::cleaning tensor with name {}", name );
            tensor_name_records_.erase( tensor_name_records_.find( name ) );
        }

        void clean( std::string const& tensor_name )
        {
            auto itor = tensor_name_records_.find( tensor_name );
            if ( itor == tensor_name_records_.end() )
            {
                spdlog::warn( "Failed to clean tensor {}, as it is not found in the tensor_name_records.", tensor_name );
                return;
            }
            clean( itor -> second );
        }


        std::tuple<std::byte*, std::vector<std::int64_t>> find_device_memory_shape_by_name( std::string const& name )
        {
            if ( auto itor = tensor_name_records_.find( name ); itor != tensor_name_records_.end() )
            {
                tensor<E> const& tsor = itor->second;
                if ( auto jtor = tensor_device_memory_records_.find( tsor.id() ); jtor != tensor_device_memory_records_.end() )
                {
                    device_memory const& dm = jtor->second;
                    return std::make_tuple( dm.data(), tsor.shape() );
                }
            }
            spdlog::error( "session::find_device_memory_shape_by_name: Failed to find mapped device memory to tensor {}.", name );
            abort();
            return {nullptr, {}};
        }

        void synchronize()
        {
            device_memory_manager_.synchronize();
        }

        stream<E>& default_arithmetic_stream()
        {
            return device_memory_manager_.default_arithmetic_stream();
        }

        device_memory_manager<E>& get_device_memory_manager()
        {
            return device_memory_manager_;
        }

        // device is out of memory, defraction is on the go
        void update_device_memory( device_memory const& old_memory, device_memory const& new_memory )
        {
            if ( old_memory == new_memory ) return;
            // find the tensor which is assigned the old memory
            if ( auto itor = device_memory_tensor_records_.find( old_memory ); itor != device_memory_tensor_records_.end() )
            {
                std::int64_t tensor_id = itor->second;
                // updated device_memory_tensor_records_
                device_memory_tensor_records_.erase( itor );
                device_memory_tensor_records_[new_memory] = tensor_id;
                // updated tensor_device_memory_records_
                tensor_device_memory_records_[tensor_id] = new_memory;
                return;
            }
            spdlog::warn( "Failed updated device memory with old memory size {0:d} address {1:x}  and new memory size {2:d} address {3:x}. Are they mapped to tensors?", old_memory.size_in_bytes_, (std::int64_t)old_memory.address_, new_memory.size_in_bytes_, (std::int64_t)new_memory.address_ );
        }

        // host-->device
        // Note: stream should be synchronized after each call.
        void tensor_to_device( tensor<E>& tsor, device_memory& dm, stream<E>& sm, bool memory_copy_flag=true )
        {
            spdlog::info( "host-->device for tensor with <id>:{} and <name>:{}, copy floag is {}.", tsor.id(), tsor.name(), memory_copy_flag );
            assert( tsor.size() <= dm.size() );
            device_memory_tensor_records_[dm] = tsor.id();
            tensor_device_memory_records_[tsor.id()] = dm;
            if ( memory_copy_flag )
                host2device( tsor.data(), tsor.size(), dm.data(), sm );
        }


        device_memory tensor_to_device( tensor<E>& tsor, stream<E>& sm, bool memory_copy_flag=true )
        {
            // already in device?
            if ( auto itor = tensor_device_memory_records_.find( tsor.id() ); itor != tensor_device_memory_records_.end() )
            {
                if ( tsor.size() == itor->second.size() )
                {
                    spdlog::info( "tensor_to_device:: tensor {} is already in device", tsor.id() );
                    return itor->second;
                }
                //dismiss this device memory
                spdlog::warn( "session::tensor_to_device: the tensor size has been updated from {} to {}, dismiss and allocate new device memory.", itor->second.size(), tsor.size() );
                device_memory_manager_.dismiss_memory( itor->second );
                //
                // remove from device_memory_tensor_records_
                if ( auto jtor = device_memory_tensor_records_.find( itor->second ); jtor != device_memory_tensor_records_.end() )
                {
                    better_assert( itor->first == tsor.id(), "Expecting same tensor id for device memory" );
                    spdlog::info( format("remove device memory {} wit tensor id {} fro device_memory_tensor_records_", itor->second, itor->first) );
                    device_memory_tensor_records_.erase( jtor );
                }
            }
            std::int64_t const memory_size_in_bytes_to_acquire = tsor.size();
            device_memory ans = device_memory_manager_.acquire_memory( memory_size_in_bytes_to_acquire );
            if ( (0 == ans.size()) || (nullptr == ans.data()) )
            {
                spdlog::error( nnl::format( "session::tensor_to_device: Failed to allocate device memory for tensor {}.\nCurrent session: {}", tsor, (*this) ) );
                std::abort();
            }
            tensor_to_device( tsor, ans, sm, memory_copy_flag );
            return ans;
        }

        device_memory tensor_to_device( tensor<E>& tsor)
        {
            return tensor_to_device( tsor, get_device_memory_manager().default_host2device_stream() );
        }

        device_memory tensor_mapped_to_device( tensor<E>& tsor) // <-- without memcpy
        {
            return tensor_to_device( tsor, device_memory_manager_.default_host2device_stream(), false );
        }

        device_memory tensor_to_device( std::int64_t id, stream<E>& sm, bool memory_copy_flag=true )
        {
            auto itor = tensor_id_records_.find( id );
            assert( (itor != tensor_id_records_.end()) && "Failed to find tensor by id." );
            return tensor_to_device( itor->second, sm, memory_copy_flag );
        }

        device_memory tensor_to_device( std::int64_t id )
        {
            return tensor_to_device( id, device_memory_manager_.default_host2device_stream() );
        }

        device_memory tensor_mapped_to_device( std::int64_t id )
        {
            return tensor_to_device( id, device_memory_manager_.default_host2device_stream(), false );
        }

        device_memory tensor_to_device( std::string const& name, stream<E>& sm, bool memory_copy_flag=true )
        {
            spdlog::info( "tensor_to_device called with name={}, and memory_copy_flag={}", name, memory_copy_flag);
            auto itor = tensor_name_records_.find( name );
            better_assert( (itor != tensor_name_records_.end()), nnl::format("Failed to find tensor by name {}.", name) );
            return tensor_to_device( itor->second, sm, memory_copy_flag );
        }

        device_memory tensor_to_device( std::string const& name )
        {
            return tensor_to_device( name, device_memory_manager_.default_host2device_stream() );
        }

        device_memory tensor_mapped_to_device( std::string const& name )
        {
            return tensor_to_device( name, device_memory_manager_.default_host2device_stream(), false );
        }

        bool tensor_to_host( tensor<E>& tsor, stream<E>& sm )
        {
            spdlog::info( "device-->host for tensor with <id>:{} and <name>:{}", tsor.id(), tsor.name() );
            auto itor = tensor_device_memory_records_.find( tsor.id() );
            if (itor == tensor_device_memory_records_.end())
            {
                spdlog::error( "tensor_to_host: Failed to find tensor with id {} from session.", tsor.id() );
                return false;
            }
            auto dm = itor->second;
            device2host( dm.data(), dm.size(), tsor.data(), sm );
            return true;
        }

        bool tensor_to_host( tensor<E>& tsor )
        {
            return tensor_to_host( tsor, device_memory_manager_.default_device2host_stream() );
        }

        bool tensor_to_host( std::int64_t id, stream<E>& sm )
        {
            auto itor = tensor_id_records_.find( id );
            if (itor == tensor_id_records_.end())
            {
                spdlog::error( "tensor_to_host: Failed to find tensor with id {} from session.", id );
                return false;
            }
            return tensor_to_host( itor->second, sm );
        }

        bool tensor_to_host( std::int64_t id )
        {
            return tensor_to_host( id, device_memory_manager_.default_device2host_stream() );
        }

        bool tensor_to_host( std::string const& name, stream<E>& sm )
        {
            auto itor = tensor_name_records_.find( name );
            if (itor == tensor_name_records_.end())
            {
                spdlog::error( "tensor_to_host: Failed to find tensor with name {} from session.", name );
                return false;
            }
            return tensor_to_host(itor->second, sm);
        }

        bool tensor_to_host( std::string const& name )
        {
            return tensor_to_host( name, device_memory_manager_.default_device2host_stream() );
        }

        bool tensor_dismiss_device( tensor<E>& tsor )
        {
            /*
            if constexpr ( debug_mode ) // <- dump tensor to disk
            {
                tensor_to_host( tsor.name(), device_memory_manager_.default_io_stream() );
                tsor.save_txt( std::string{"./"}+tsor.name()+std::string{".txt"} );
            }
            */

            spdlog::info( "session::tensor_dismiss_device: removing tensor {}:{} from device.", tsor.name(), tsor.id() );
            auto itor = tensor_device_memory_records_.find( tsor.id() );
            if (itor == tensor_device_memory_records_.end())
            {
                spdlog::error( nnl::format( "tensor_dismiss_device: Failed to find tensor {}", tsor ) );
                return false;
            }
            spdlog::info( "session::tensor_dismiss_device: with tensor id {}.", tsor.id() );
            device_memory_manager_.dismiss_memory( itor->second );
            device_memory_tensor_records_.erase( itor->second );
            tensor_device_memory_records_.erase( itor );
            return true;
        }

        bool tensor_dismiss_device( std::int64_t id )
        {
            auto itor = tensor_id_records_.find( id );
            if (itor == tensor_id_records_.end())
            {
                spdlog::error( "tensor_dismiss_device: Failed to find tensor with id {} from session.", id );
                return false;
            }
            return tensor_dismiss_device( itor->second );
        }

        bool tensor_dismiss_device( std::string const& name )
        {
            auto itor = tensor_name_records_.find(name);
            if (itor == tensor_name_records_.end())
            {
                spdlog::error( "tensor_dismiss_device: Failed to find tensor with name {} from session.", name );
                return false;
            }
            return tensor_dismiss_device(itor->second);
        }

        // newly constructed tensors are supposed to be registered to the global session
        void register_tensor( tensor<E> const& t )
        {
            tensor_id_records_.insert( std::make_pair(t.id(), t ) );
            //spdlog::info( "Session:: Registered a tensor with id {}", t.id() );
            // TODO: register tensor with name
        }

        // newly constructed tensors are supposed to be registered to the global session
        void register_tensor( std::string name, tensor<E>& t )
        {
            if ( tensor_name_records_.find( name ) != tensor_name_records_.end() )
            {
                if ( tensor_id_records_.find( t.id() ) != tensor_id_records_.end() )
                {
                    spdlog::warn( nnl::format("Tensor {} has been registered. Skipping.", t) );
                    return;
                }

                spdlog::warn( "This tensor <name {}, id {}> has been registered", t.name(), t.id() );
                name = name + "-->" + std::to_string(t.id());
                t.set_name( name );
                spdlog::warn( "changing to a new name {}", name );
                return register_tensor( name, t );
            }

            tensor_name_records_.insert( std::make_pair( name, t ) );
            register_tensor( t );
            spdlog::info( "Session:: Registered a tensor with name {}", name );
        }
    }; //struct session

    template< Engine E >
    std::ostream& operator<<( std::ostream& os, session<E> const& s )
    {
        os << "Session:\n";
        os << s.device_memory_manager_ << "\n";
        os << "\tTensor ID Records:\n";
        for ( auto const&[id, ts] : s.tensor_id_records_ )
            os << "\t\t" << id << ": <tensor> " << ts << "\n";
        os << "\tTensor Name Records:\n";
        for ( auto const&[nm, ts] : s.tensor_name_records_ )
            os << "\t\t" << nm << ": <tensor> " << ts << "\n";
        os << "\tDevice memory tensor Records:\n";
        for ( auto const&[dm, id] : s.device_memory_tensor_records_ )
            os << "\t\t" << dm << ": <id> " << id << "\n";
        os << "\tDevice memory tensor Records:\n";
        for ( auto const&[id, dm] : s.tensor_device_memory_records_ )
            os << "\t\t" <<" <tensor> " << s.tensor_id_records_.at(id) << " " << dm << "\n";
            //os << "\t\t" <<" <id> " << id << " " << dm << "\n";
        return os;
    }

    template< Engine E >
    inline session<E>& get_default_session()
    {
        return singleton<session<E>>::instance();
    }

    //
    // This method iteratively go through the computation graph,
    // inference input and output shapes for each node,
    // and allocate tensors for the input and output tensors
    //
    template<typename Node >
    void graph<Node>::inference_io_shapes()
    {
        auto edges = (*this).edges();
        auto nodes = (*this).computation_order();

        for ( auto& node : nodes )
        {
            spdlog::info( nnl::format( "inferencing io shapes for node {}.", node.name() ) );
            if ( node.is_memcpy_only() ) // skip input tensors
            {
                spdlog::info( nnl::format( "skip {} as it is memcpy only.", node.name() ) );
                continue;
            }

            {
                auto& input_nodes = edges[node];
                {
                    for ( auto in : input_nodes )
                    {
                        spdlog::info(  "Receive input node {} for node {}" , in.name(), node.name());
                    }
                }


                std::vector<std::string> input_tensor_names;
                for ( auto& input_node : input_nodes )
                {
                    std::vector<std::string> output_names = input_node.outputs();
                    {
                        // remove duplicated output names
                        std::sort( output_names.begin(), output_names.end() );
                        output_names.erase( std::unique( output_names.begin(), output_names.end() ), output_names.end() );
                    }
                    spdlog::info( nnl::format("input node {} has {} outputs: {}.", input_node.name(), output_names.size(), output_names ) );
                    /*
                    better_assert( output_names.size() == 1, nnl::format( "Expect a single output name, but received {}", output_names.size() ) );
                    input_tensor_names.push_back( output_names[0] );
                    spdlog::info( "appending {} to input_tensor_names for input node {}", output_names[0], input_node.name() );
                    */
                    for ( auto idx : range( output_names.size() ) )
                    {
                        input_tensor_names.push_back( output_names[idx] );
                        spdlog::info( "appending {} to input_tensor_names for input node {}", output_names[idx], input_node.name() );
                    }
                }
                node.set_inputs( input_tensor_names ); // <-- refresh input tensors, which should be the output of input nodes
                spdlog::info( nnl::format( "Graph::inference_io_shapes: set inputs {} to node {} id {}", input_tensor_names, node.name(), node.id() ) );

                std::vector<std::vector<std::int64_t>> output_shapes = node.output_shapes(); // <-- calculate output shape with the new input tensors
                better_assert( 1 <= output_shapes.size() ); // <-- will we have node producing more than 1 outputs?
                //spdlog::info( nnl::format( "Graph::inference_io_shapes: found output shape {}", output_shapes[0] ) );
                spdlog::info( nnl::format( "Graph::inference_io_shapes: found output shape {}", output_shapes ) );

                // create a new tensor for the output
                std::vector<tensor<default_engine_type>> output_tensors;
                for ( auto idx : range( output_shapes.size() ) )
                {
                    output_tensors.emplace_back( make_tensor<default_engine_type>( output_shapes[idx], "float32", std::string{"output_tensor_for_node_"}+node.name()+std::string{"_node_id_"}+std::to_string(node.id())+std::string{"_output_index_"}+std::to_string(idx) ) ); //TODO: if not fp32
                    spdlog::info( nnl::format( "Graph::inference_io_shapes: set output tensor {} to node {}", output_tensors[idx], node.name() ) );
                }
                node.set_outputs( output_tensors );
            }
        }
    }

}//namespace nnl

#endif//TLVVYPKNUGQHHYILQSAHOOQYRUBYHJUVYMCSGDJDHTYUHIYXTHOILSAWMJTGJHERXVHMRAMSY


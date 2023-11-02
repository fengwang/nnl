#ifndef FVGYPNDHHNEYLIATJROIVBVHMVPWHPHPLPSFOGCLYNYRHUHAKRTPAAIPKPFKXVEVOQASAFMIH
#define FVGYPNDHHNEYLIATJROIVBVHMVPWHPHPLPSFOGCLYNYRHUHAKRTPAAIPKPFKXVEVOQASAFMIH

#include "../utility/utility.hpp"
#include "./attribute.hpp"
#include "./engine.hpp"
#include "./session.hpp"
#include "./stream.hpp"
#include "./device.hpp"

namespace nnl
{

template< Engine E >
struct session;

template< Engine E >
session<E>& get_default_session();

namespace node_
{
    template< typename Concrete >
    struct crtp_attributes
    {
        std::vector<attribute> attributes() const
        {
            return attributes_;
        }

        void set_attributes( std::vector<attribute> const& attributes )
        {
            attributes_ = attributes;
        }

        void add_attribute( attribute const& att )
        {
            attributes_.push_back( att );
        }

        std::vector<attribute> attributes_;

    };//struct crtp_attributes

    template< typename Concrete >
    struct crtp_name
    {
        std::string name() const
        {
            return name_;
        }

        void set_name( std::string const& name )
        {
            if ( name.size() )
                name_ = name;
            else
            {
                auto& zen = static_cast<Concrete&>(*this);
                name_ = std::string{"Node-"} + std::to_string( zen.id() );
            }
        }

        std::string name_;
    };//struct crtp_name

    template< typename Concrete >
    struct crtp_arithmetic_operation
    {
        std::vector<std::byte*> device_side_weights;
        std::vector<std::byte*> device_side_inputs;
        std::vector<std::byte*> device_side_outputs;

        std::vector<std::vector<std::int64_t>> weights_shape;
        std::vector<std::vector<std::int64_t>> inputs_shape;
        std::vector<std::vector<std::int64_t>> outputs_shape;

        std::byte* device_side_buffer;
        std::int64_t buffer_size_in_bytes;

        auto arithmetic_operation()
        {
            auto& zen = static_cast<Concrete&>( *this );
            auto& sess = get_default_session<default_engine_type>();

            // regenerate shapes and mapped device memory address
            {
                spdlog::info( "\tregenerate device side weights for node {}:{}", zen.id(), zen.name() );
                device_side_weights.clear();
                weights_shape.clear();
                std::vector<std::string> const& weights_name = zen.weights();
                for ( auto const& name : weights_name )
                {
                    auto const& [mem, shape] = sess.find_device_memory_shape_by_name( name ); // <-- ensure weights are loaded
                    device_side_weights.push_back(mem);
                    weights_shape.push_back( shape );
                }
            }

            {
                spdlog::info( "\tregenerate device side inputs for node {}:{}", zen.id(), zen.name() );
                device_side_inputs.clear();
                inputs_shape.clear();
                std::vector<std::string> const& inputs_name = zen.inputs();
                for ( auto const& name : inputs_name )
                {
                    auto const& [mem, shape] = sess.find_device_memory_shape_by_name( name );
                    device_side_inputs.push_back(mem);
                    inputs_shape.push_back( shape );
                    spdlog::info( nnl::format( "determines input for node id {}: {}", zen.id(), name ) );
                }
            }

            {
                spdlog::info( "\tremap device side outputs for node {}:{}", zen.id(), zen.name() );
                device_side_outputs.clear();
                outputs_shape.clear();
                std::vector<std::string> const& outputs_name = zen.outputs();
                for ( auto const& name : outputs_name )
                {
                    auto const& [mem, shape] = sess.find_device_memory_shape_by_name( name );
                    device_side_outputs.push_back(mem);
                    outputs_shape.push_back( shape );

                    /*
                    if constexpr ( debug_mode )
                    {
                        spdlog::info( nnl::format("\t\t node get output tensor of shape {} at {}.", shape, mem) );
                    }
                    */
                }
            }

            // buffer
            device_memory_buffer<default_engine_type> buff; // RAII
            {
                std::string const& bf = zen.buffer();
                if ( bf.size()  ) // <-- if manually assigned buffer, use it
                {
                    auto const& [address, size_in_bytes] = sess.tensor_mapped_to_device( bf );
                    device_side_buffer = address;
                    buffer_size_in_bytes = size_in_bytes;
                    spdlog::warn( "Warn: layer buffer requirements could change with different input layer shapes, manually assign buffer is not recommanded." );
                }
                else // <-- automaticlly apply buffer
                {
                    //calculate buffer size with respect to inputs and outputs
                    std::int64_t buffer_size_in_bytes = zen.calculate_buffer_size_in_bytes( zen.weights_shape, zen.inputs_shape );
                    if ( buffer_size_in_bytes > 0 )
                    {
                        buff.reserve( buffer_size_in_bytes );
                        zen.device_side_buffer = buff.data();
                        zen.buffer_size_in_bytes = buffer_size_in_bytes;
                    }

                    if constexpr (debug_mode)
                    {
                        spdlog::info( "Node {} requested {} bytes buffer memory starting at {}.", zen.id(), buffer_size_in_bytes, (std::int64_t)buff.data() );
                    }
                }
            }

#if 0
            zen.implement_arithmetic_operation( zen.device_side_weights, zen.weights_shape,
                                                 zen.device_side_inputs, zen.inputs_shape,
                                                 zen.device_side_outputs, zen.outputs_shape,
                                                 zen.device_side_buffer, zen.buffer_size_in_bytes,
                                                 sess.default_arithmetic_stream() );
#else
            zen.implement_attributed_arithmetic_operation
            (
                zen.device_side_weights, zen.weights_shape,
                zen.device_side_inputs, zen.inputs_shape,
                zen.device_side_outputs, zen.outputs_shape,
                zen.device_side_buffer, zen.buffer_size_in_bytes,
                sess.default_arithmetic_stream(),
                zen.attributes()
            );
#endif
            //
            // TODO: implement_arithmetic_operation is default for fp32
            //       dispatch other methods to address fp16, (u)int8, and (u)int4 cases
            //
        }


        virtual void implement_attributed_arithmetic_operation
        (
            std::vector<std::byte*> const& d_w, std::vector<std::vector<std::int64_t>> const& w_s,
            std::vector<std::byte*> const& d_i, std::vector<std::vector<std::int64_t>> const& i_s,
            std::vector<std::byte*> const& d_o, std::vector<std::vector<std::int64_t>> const& o_s,
            std::byte* d_b, std::int64_t b_s,
            stream<default_engine_type>& sm,
            [[maybe_unused]] std::vector<attribute> const& attributes
        )
        {
            auto& zen = static_cast<Concrete&>( *this );
            zen.implement_arithmetic_operation( d_w, w_s, d_i, i_s, d_o, o_s, d_b, b_s, sm );
        }

        virtual void implement_arithmetic_operation
        (
            [[maybe_unused]] std::vector<std::byte*> const& d_w, [[maybe_unused]] std::vector<std::vector<std::int64_t>> const& w_s,
            [[maybe_unused]] std::vector<std::byte*> const& d_i, [[maybe_unused]] std::vector<std::vector<std::int64_t>> const& i_s,
            [[maybe_unused]] std::vector<std::byte*> const& d_o, [[maybe_unused]] std::vector<std::vector<std::int64_t>> const& o_s,
            [[maybe_unused]] std::byte* d_b, [[maybe_unused]] std::int64_t b_s,
            [[maybe_unused]] stream<default_engine_type>& sm
        )
        {
            auto& zen = static_cast<Concrete&>( *this );
            spdlog::error( "Node <id {}, name {}> does not have arithmetic operation implemented.", zen.id(), zen.name() );
        }

    }; // crtp_arithmetic_operation

#if 0
    Adding a weight to a node:
        auto t = make_tensor<default_engine_type>( {32, 3, 3, 64}, "float32", "layer_1_weight_1" );
        std::copy( src, src+n, t.data() );
        node.add_weight( t.name() );

    Load weights to device: <<-- returns a vector of functions
        node.load_weights_to_device();
    Unload weights
        node.unload_weights_from_device();

#endif

    template< typename Concrete >
    struct crtp_weights
    {
        void load_weights_to_device()
        {
            spdlog::info( "There are {} weights in this node.", weights_.size() );
            auto& sess = get_default_session<default_engine_type>();
            for ( auto tensor_name : weights_ )
            {
                spdlog::info( "Loading weights to device for tensor {}", tensor_name );
                //sess.tensor_to_device( tensor_name );
                sess.tensor_to_device( tensor_name, sess.dm().default_weight_preloading_stream() );
            }
        }

        void dismiss_weights_from_device()
        {
            auto& sess = get_default_session<default_engine_type>();
            for ( auto tensor_name : weights_ )
            {
                if ( ! sess.tensor_dismiss_device( tensor_name ) )
                {
                    auto const& self = static_cast<Concrete const&>(*this);
                    spdlog::error( nnl::format( "node {} ::dismiss_weights_from_defice: Failed to dismiss weights from device for tensor {}", self.name(), tensor_name) );
                }
                // weights cannot be dismissed
            }
        }

        std::vector<std::string> weights() const
        {
            return weights_;
        }

        std::vector<std::vector<std::int64_t>> weight_shapes()
        {
            // weights shape can be cached as they do not change over inference routine
            if ( weights_shape_.size() > 0 )
                return weights_shape_;

            std::vector<std::string> const& weights_name = weights();
            auto& sess = get_default_session<default_engine_type>();
            for ( std::string const& name : weights_name )
            {
                //auto const& [mem, shape] = sess.find_device_memory_shape_by_name( name );
                //weights_shape_.push_back( shape );
                weights_shape_.push_back( sess.find_tensor_by_name( name ).shape() );
            }

            return weights_shape_;
        }

        void add_weight( std::string w )
        {
            weights_.push_back( w );
        }

        void set_weights( std::vector<std::string> const& weights )
        {
            weights_.clear();
            weights_ = weights;
        }

        // This is the tensor id for the weights
        std::vector<std::string> weights_;
        std::vector<std::vector<std::int64_t>> weights_shape_;
    };//struct crtp_weights


    //
    // TODO: buffer needs not to be mapped to host memory. Remove host-side tensor
    //
    template< typename Concrete >
    struct crtp_buffer
    {
        virtual std::int64_t calculate_buffer_size_in_bytes
        (
            [[maybe_unused]] std::vector<std::vector<std::int64_t>> const& weights_shape,
            [[maybe_unused]] std::vector<std::vector<std::int64_t>> const& inputs_shape
        )
        {
            return 0;
        }

        void buffer_preallocating()
        {
            if ( buffer_.size() )
            {
                auto& sess = get_default_session<default_engine_type>();
                sess.tensor_mapped_to_device( buffer_ ); // just map device memory, no memcpy required
            }
            else
            {
                auto const& self = static_cast<Concrete const&>(*this);
                spdlog::info( "buffer_preallocating:: node {} with id {} does not have buffer.", self.name(), self.id() );
            }
        }

        void dismiss_buffer_from_device()
        {
            if ( buffer_.size() )
            {
                auto const& self = static_cast<Concrete const&>(*this);
                spdlog::info( nnl::format("dismiss_buffer_from_device:: node {} with id {} dismiss buffer {}.", self.name(), self.id(), buffer_) );
                auto& sess = get_default_session<default_engine_type>();
                if ( !sess.tensor_dismiss_device( buffer_ ) )
                {
                    auto const& self = static_cast<Concrete const&>(*this);
                    spdlog::error( nnl::format( "node {} ::dismiss_buffer_from_device: Failed to dismiss buffer from device for buffer {}", self.name(), buffer_ ) );
                }
            }
            else
            {
                auto const& self = static_cast<Concrete const&>(*this);
                spdlog::info( "dismiss_buffer_from_device:: node {} with id {} does not need buffer.", self.name(), self.id() );
            }


            // clean memory footprint
            auto& sess = get_default_session<default_engine_type>();
            sess.clean( buffer_ );
        }

        std::string buffer() const
        {
            return buffer_;
        }

        void add_buffer( std::string const& buffer_name )
        {
            if ( buffer_name.size() < 1 )
            {
                spdlog::error( "add_buffer error:: buffer name cannot be empty." );
                std::abort();
            }

            buffer_ = buffer_name;
        }

        void set_buffer( std::string const& buffer )
        {
            buffer_ = buffer;
        }

        void set_buffer( std::initializer_list<std::int64_t> shape, std::string const& dtype, std::string name=std::string{} )
        {
            if ( name.size() < 1 )
            {
                auto const& self = static_cast<Concrete const&>( *this );
                name =std::string{"buffer_"} + std::to_string( self.id() );
            }
            auto tsor = make_tensor<default_engine_type>( shape, dtype, name );
            set_buffer( tsor.name() );
        }


        std::string buffer_;
    };//struct crtp_buffer


    template< typename Concrete >
    struct crtp_inputs
    {
        void load_inputs_to_device()
        {
            auto& sess = get_default_session<default_engine_type>();
            for ( auto tensor_name : inputs_ )
                sess.tensor_to_device( tensor_name );
        }

        void unload_inputs_from_device()
        {
            auto& sess = get_default_session<default_engine_type>();
            for ( auto tensor_name : inputs_ )
            {
                sess.tensor_to_host( tensor_name );
            }
        }

        void dismiss_inputs_from_device()
        {
            auto& zen = static_cast<Concrete&>( *this );
            auto& sess = get_default_session<default_engine_type>();
            for ( auto tensor_name : inputs_ )
            {
                spdlog::warn( "Dismiss input tensor {} from device by node {}::dismiss_inputs_from_device.", tensor_name, zen.name() );
                //sess.tensor_dismiss_device( tensor_name );
                sess.clean( tensor_name ); // <-- aggressive clean
            }
        }

        std::vector<std::string> inputs() const
        {
            return inputs_;
        }

        std::vector<std::vector<std::int64_t>> input_shapes() const
        {
            // inputs shape cannot be cached as i/o tensor shapes might change
            std::vector<std::vector<std::int64_t>> ans;
            {
                auto& sess = get_default_session<default_engine_type>();
                std::vector<std::string> const& inputs_name = inputs();
                better_assert( inputs_name.size() );
                for ( auto const& name : inputs_name )
                {
                    //auto const& [mem, shape] = sess.find_device_memory_shape_by_name( name );
                    //ans.push_back( shape );
                    ans.push_back( sess.find_tensor_by_name( name ).shape() );
                }
            }
            return ans;
        }

        virtual void add_input( std::string const& w ) // <-- input layer will take the chance to set output
        {
            better_assert( (w.size() > 0), "add_input cannot accept empty weight name." );
            inputs_.push_back( w );
        }

        void set_inputs( std::vector<std::string> const& inputs )
        {
            inputs_ = inputs;
        }

        // This is the tensor id for the inputs
        std::vector<std::string> inputs_;
    };//struct crtp_inputs

    template< typename Concrete >
    struct crtp_outputs
    {
        void load_outputs_to_device()
        {
            auto& sess = get_default_session<default_engine_type>();
            for ( auto tensor_name : outputs_ )
                sess.tensor_to_device( tensor_name );
        }

        void unload_outputs_from_device()
        {
            auto& sess = get_default_session<default_engine_type>();
            for ( auto tensor_name : outputs_ )
            {
                sess.tensor_to_host( tensor_name );
            }
        }

        // dismiss but not removing the record
        void dismiss_outputs_from_device()
        {
            auto& zen = static_cast<Concrete&>( *this );
            if ( zen.is_memcpy_only() ) // <-- skip for input layers
            {
                return;
            }

            auto& sess = get_default_session<default_engine_type>();
            for ( auto tensor_name : outputs_ )
            {
                spdlog::info( "Dismiss output tensor {} from device by node {}::dismiss_outputs_from_device.", tensor_name, zen.name() );
                spdlog::info( nnl::format( "Tensor {}: {}", tensor_name, sess.find_tensor_by_name(tensor_name) ) );
                if constexpr( debug_mode ) // sync to host if in debug mode
                {
                    auto tsor = sess.find_tensor_by_name( tensor_name );
                    tsor.synchronize_to_host();
                }

                if ( ! sess.tensor_dismiss_device( tensor_name ) )
                {
                    auto const& self = static_cast<Concrete const&>(*this);
                    spdlog::error( nnl::format( "node {} ::dismiss_outputs_from_device: Failed to dismiss output tensor {} from device.", self.name(), tensor_name ) );
                }
            }

            if constexpr( debug_mode ) // sync to harddisk if in debug mode
            {
                sess.synchronize();

                for ( auto tensor_name : outputs_ )
                {
                    auto tsor = sess.find_tensor_by_name( tensor_name );
                    tsor.save_txt(  tensor_name + std::string{".txt"} );
                }
            }
        }

        // dismiss and then clear the record
        void clean_outputs_from_device()
        {
            auto& zen = static_cast<Concrete&>( *this );
            zen.dismiss_outputs_from_device();

            auto& sess = get_default_session<default_engine_type>();
            for ( auto tensor_name : outputs_ )
            {
                sess.clean( tensor_name );
            }
        }

        void output_preallocating()
        {
            auto& sess = get_default_session<default_engine_type>();
            for ( auto tensor_name : outputs_ )
            {
                spdlog::info( "Preallocating device memory for tensor {}.", tensor_name );
                sess.tensor_mapped_to_device( tensor_name ); // <-- mapped without copying memory
            }
        }

        std::vector<std::string> outputs() const
        {
            return outputs_;
        }

        void add_output( std::string w )
        {
            outputs_.push_back( w );
        }

        void set_outputs( std::vector<std::string> const& outputs )
        {
            outputs_ = outputs;
        }

        std::vector<std::vector<std::int64_t>> output_shapes()
        {
            auto& zen = static_cast<Concrete&>( *this );
            return zen.inference_outputs_shape( zen.weight_shapes(), zen.input_shapes() );
        }

        virtual std::vector<std::vector<std::int64_t>> inference_outputs_shape
        (
            [[maybe_unused]] std::vector<std::vector<std::int64_t>> const& weights_shape,
            [[maybe_unused]] std::vector<std::vector<std::int64_t>> const& inputs_shape
        )
        {
            auto& zen = static_cast<Concrete&>( *this );
            better_assert( false, nnl::format("inference_outputs_shape method for node {} has not yet implemented."), zen.name() );
            return inputs_shape;
        }

        // This is the tensor id for the outputs
        std::vector<std::string> outputs_;
    };//struct crtp_outputs






    // if true, this node only counts on memcpy from host to device, i.e., an input/const node
    template< typename Concrete >
    struct crtp_is_memcpy_only
    {
        crtp_is_memcpy_only(): _is_memcpy_only{ false } {}

        bool is_memcpy_only() const
        {
            return _is_memcpy_only;
        }

        void set_memcpy_only()
        {
            _is_memcpy_only = true;
        }

        bool _is_memcpy_only;
    };//struct crtp_is_memcpy_only

    template< typename Concrete >
    struct crtp_is_output_node
    {
        crtp_is_output_node(): _is_output_node{ false } {}

        bool is_output_node() const
        {
            return _is_output_node;
        }

        void set_output_node()
        {
            _is_output_node = true;
        }

        bool _is_output_node;
    };//struct crtp_is_output_node


struct node :
    enable_id<node>,
    crtp_arithmetic_operation<node>,
    crtp_attributes<node>,
    crtp_weights<node>,
    crtp_buffer<node>,
    crtp_inputs<node>,
    crtp_outputs<node>,
    crtp_is_memcpy_only<node>,
    crtp_is_output_node<node>,
    crtp_name<node>
{
}; // struct nnl::node_::node

}//namespace node_

// TODO: enable crtp
struct node
{
    std::shared_ptr<node_::node> node_;

    node(std::shared_ptr<node_::node> nd=std::make_shared<node_::node>()) : node_{ nd } {}

    node( std::shared_ptr<node_::node> nd, std::string const& name ) : node_{ nd }
    {
        set_name( name );
    }

    // add an attribute to the node, example:
    //
    // auot xxx_node = make_node( "xxx");
    // xxx_node.add_attribute( make_attribute_heads( 5 ) );
    // xxx_node.add_attribute( make_attribute_input_shape( std::vector<std::int64_t>{ 1, 2, 3, 4,}  ) );
    //
    void add_attribute( attribute const& attr )
    {
        node_->add_attribute( attr );
    }

    std::vector<attribute> attributes() const
    {
        return (*node_).attributes();
    }

    int id() const
    {
        return node_->id();
    }

    operator bool() const noexcept
    {
        return node_ ? true : false;
    }

    void set_name( std::string const& name )
    {
        assert( node_ );
        node_->set_name( name );
    }

    std::string name() const
    {
        assert( node_ );
        return node_->name();
    }

    std::vector<std::string> weights() const
    {
        return node_->weights();
    }

    void add_weight( std::string const& w )
    {
        node_->add_weight( w );
    }

    void add_weight( tensor<default_engine_type> const& t )
    {
        add_weight( t.name() );
    }

    void set_weights( std::vector<std::string> const& ws )
    {
        node_->set_weights( ws );
    }

    //TODO: here default_engine_type can be replaced with <Engine E>
    void set_weights( std::vector<tensor<default_engine_type>> const& ts )
    {
        std::vector<std::string> ws;
        for ( auto const& t : ts )
            ws.push_back( t.name() );
        node_->set_weights( ws );
    }

    std::vector<std::string> inputs() const
    {
        return node_->inputs();
    }

    void add_input( std::string const& w )
    {
        node_->add_input( w );
    }

    void add_input( tensor<default_engine_type>const& t )
    {
        add_input( t.name() );
    }

    void set_inputs( std::vector<std::string> const& ws )
    {
        node_->set_inputs( ws );
    }

    void set_inputs( std::vector<tensor<default_engine_type>> const& ts )
    {
        std::vector<std::string> ws;
        for ( auto const& t : ts )
            ws.push_back( t.name() );
        node_->set_inputs( ws );
    }

    std::vector<std::string> outputs() const
    {
        return node_->outputs();
    }

    void add_output( std::string const& w )
    {
        node_->add_output( w );
    }

    void add_output( tensor<default_engine_type> const& t )
    {
        add_output( t.name() );
    }

    void set_outputs( std::vector<std::string> const& ws )
    {
        node_->set_outputs( ws );
    }

    void set_outputs( std::vector<tensor<default_engine_type>> const& ts )
    {
        std::vector<std::string> ws;
        for ( auto const& t : ts )
            ws.push_back( t.name() );
        node_->set_outputs( ws );
    }

    std::vector<std::vector<std::int64_t>> output_shapes()
    {
        return node_->output_shapes();
    }

    std::vector<std::vector<std::int64_t>> input_shapes()
    {
        return node_->input_shapes();
    }

    std::vector<std::vector<std::int64_t>> weight_shapes()
    {
        return node_->weight_shapes();
    }

    void set_buffer( std::initializer_list<std::int64_t> shape, std::string const& dtype, std::string const& name=std::string{} )
    {
        node_->set_buffer( shape, dtype, name );
    }

    void add_buffer( std::string const& name )
    {
        node_->add_buffer( name );
    }

    void add_buffer( tensor<default_engine_type> const& t )
    {
        add_buffer( t.name() );
    }

    std::string buffer() const
    {
        return node_->buffer();
    }

    void set_memcpy_only()
    {
        node_->set_memcpy_only();
    }

    bool is_memcpy_only() const
    {
        return node_->is_memcpy_only();
    }

    void set_output_node()
    {
        node_->set_output_node();
    }

    bool is_output_node() const
    {
        return node_->is_output_node();
    }


    //
    // interfaces for computation table
    //
     void arithmetic_operation()
    {
        if ( !node_ ) return;
        spdlog::info( "arithmetic operation for node {} with id {}", (*this).name(), (*this).id() );
        node_->arithmetic_operation();
    }

    void host2device_operation() // for case of host2device_operation
    {
        if ( !node_ ) return;
        spdlog::info( "host2device operation for node {} with id {}", (*this).name(), (*this).id() );
        node_->load_outputs_to_device();
    }

    void device2host_operation()
    {
        if ( !node_ ) return;
        spdlog::info( "device2host operation for node {} with id {}", (*this).name(), (*this).id() );
        node_->unload_outputs_from_device();
        node_->dismiss_outputs_from_device();
    }

    void device2host_dismiss_operation()
    {
        if ( !node_ ) return;
        spdlog::info( "device2host dismiss operation for node {} with id {}", (*this).name(), (*this).id() );
        // Not here....
        //node_->unload_outputs_from_device();
        //
        //node_->dismiss_outputs_from_device();
        node_->clean_outputs_from_device(); // <-- remove the output tensor from host as well
    }

    // buffer is coupled with weights
    void weight_preloading()
    {
        if ( !node_ ) return;
        spdlog::info( "weight preloading operation for node {} with id {}", (*this).name(), (*this).id() );
        node_->load_weights_to_device();
        spdlog::info( "Preloading weights by preallocating device space" );
        node_->buffer_preallocating();
    }

    // buffer is coupled with weights
    void weight_dismissing()
    {
        if ( !node_ ) return;
        spdlog::info( "weight dismissing operation for node {} with id {}", (*this).name(), (*this).id() );
        node_->dismiss_weights_from_device();
        node_->dismiss_buffer_from_device();
    }

    void output_preallocating()
    {
        if ( !node_ ) return;
        spdlog::info( "output preallocating operation for node {} with id {}", (*this).name(), (*this).id() );
        node_->output_preallocating();
    }

};//struct node

inline bool operator == ( node const& lhs, node const& rhs )
{
    return lhs.node_ == rhs.node_;
}

inline bool operator < ( node const& lhs, node const& rhs )
{
    return lhs.node_ < rhs.node_;
}

inline std::ostream& operator<< ( std::ostream& os, node const& rhs )
{
    if ( not rhs )
        return os << "<Node[dummy]>";
    return os << "<Node[" << rhs.id() << "] : " << rhs.name() << ">";
}

inline node make_node( std::string const& name )
{
    node ans;
    ans.set_name( name );
    return ans;
}

namespace node_
{
    struct input : node
    {
        template< typename ... Args >
        input( Args&&... args ) : node{ std::forward<Args>(args)... }
        {
            (*this).set_memcpy_only();
        }

        // directly bind to output
        void add_input( std::string const& w ) override
        {
            if ( w.size() < 1 )
            {
                spdlog::error( "add_input:: weight name cannot be empty." );
                std::abort();
            }
            outputs_.push_back( w );
        }

        std::vector<std::vector<std::int64_t>> inference_outputs_shape ( std::vector<std::vector<std::int64_t>> const& weights_shape, std::vector<std::vector<std::int64_t>> const& inputs_shape ) override
        {
            better_assert( 0 == weights_shape.size() );
            better_assert( 1 == inputs_shape.size() );
            spdlog::info( nnl::format( "inference_outputs_shape with node {} id {}: {}", (*this).name(), (*this).id(), inputs_shape ) );
            return inputs_shape;
        }
    }; //node_::input

    struct gelu : node
    {
        void implement_arithmetic_operation( std::vector<std::byte*>const&, std::vector<std::vector<std::int64_t>>const&, std::vector<std::byte*>const&, std::vector<std::vector<std::int64_t>>const&, std::vector<std::byte*>const&, std::vector<std::vector<std::int64_t>>const&, std::byte*, std::int64_t, stream<default_engine_type>& ) override;

        std::vector<std::vector<std::int64_t>> inference_outputs_shape ( std::vector<std::vector<std::int64_t>> const& weights_shape, std::vector<std::vector<std::int64_t>> const& inputs_shape ) override
        {
            better_assert( weights_shape.size() == 0 );
            better_assert( inputs_shape.size() == 1 );
            return inputs_shape;
        }
    }; //node_::gelu

    struct add : node
    {
        void implement_arithmetic_operation( std::vector<std::byte*>const&, std::vector<std::vector<std::int64_t>>const&, std::vector<std::byte*>const&, std::vector<std::vector<std::int64_t>>const&, std::vector<std::byte*>const&, std::vector<std::vector<std::int64_t>>const&, std::byte*, std::int64_t, stream<default_engine_type>& ) override;

        // shape broadcasting
        std::vector<std::vector<std::int64_t>> inference_outputs_shape ( [[maybe_unused]] std::vector<std::vector<std::int64_t>> const& weights_shape, std::vector<std::vector<std::int64_t>> const& inputs_shape ) override
        {
            better_assert( weights_shape.size() == 0 );
            better_assert( inputs_shape.size() == 2 );

            std::vector<std::int64_t> lhs_shape = inputs_shape[0];
            std::int64_t const lhs_size = std::accumulate( lhs_shape.begin(), lhs_shape.end(), std::int64_t{1}, [](std::int64_t a, std::int64_t b){ return a*b; } );
            std::int64_t const lhs_col = *(lhs_shape.rbegin());
            std::int64_t const lhs_row = lhs_size / lhs_col;

            std::vector<std::int64_t> rhs_shape = inputs_shape[1];
            std::int64_t const rhs_size = std::accumulate( rhs_shape.begin(), rhs_shape.end(), std::int64_t{1}, [](std::int64_t a, std::int64_t b){ return a*b; } );
            std::int64_t const rhs_col = *(rhs_shape.rbegin());
            std::int64_t const rhs_row = rhs_size / rhs_col;

            std::vector<std::vector<std::int64_t>> ans;
            ans.emplace_back( std::vector<std::int64_t>{ { std::max(lhs_row, rhs_row), std::max(lhs_col, rhs_col), } } );
            spdlog::info( nnl::format("add::inference_outputs_shape: <{}> + <{}> ==> <{}>", lhs_shape, rhs_shape, ans[0]) );
            return ans;
        }
    }; //node_::add


    struct softmax : node
    {
        void implement_arithmetic_operation( std::vector<std::byte*>const&, std::vector<std::vector<std::int64_t>>const&, std::vector<std::byte*>const&, std::vector<std::vector<std::int64_t>>const&, std::vector<std::byte*>const&, std::vector<std::vector<std::int64_t>>const&, std::byte*, std::int64_t, stream<default_engine_type>& ) override;

        std::vector<std::vector<std::int64_t>> inference_outputs_shape ( std::vector<std::vector<std::int64_t>> const& weights_shape, std::vector<std::vector<std::int64_t>> const& inputs_shape ) override
        {
            better_assert( weights_shape.size() == 0 );
            better_assert( inputs_shape.size() == 1 );
            return inputs_shape;
        }
    }; //node_::softmax

    //
    // layer_norm can optionally take two weights:
    // 1. the first one is the scale tensor, or g vector in gpt2 context
    // 2. the last one is the offset tensor, or b vector in gpt2 context
    // Those two tensors are one dimentional, identical to the laste dimension of the input tensor (which is a 2D tensor)
    //
    struct layer_norm : node
    {
        void implement_arithmetic_operation( std::vector<std::byte*>const&, std::vector<std::vector<std::int64_t>>const&, std::vector<std::byte*>const&, std::vector<std::vector<std::int64_t>>const&, std::vector<std::byte*>const&, std::vector<std::vector<std::int64_t>>const&, std::byte*, std::int64_t, stream<default_engine_type>& ) override;

        std::vector<std::vector<std::int64_t>> inference_outputs_shape ( std::vector<std::vector<std::int64_t>> const& weights_shape, std::vector<std::vector<std::int64_t>> const& inputs_shape ) override
        {
            better_assert( weights_shape.size() == 0 || weights_shape.size() == 2 );
            better_assert( inputs_shape.size() == 1 );
            return inputs_shape;
        }

        std::int64_t calculate_buffer_size_in_bytes
        (
            [[maybe_unused]] std::vector<std::vector<std::int64_t>> const& w_s,
            [[maybe_unused]] std::vector<std::vector<std::int64_t>> const& i_s
        ) override
        {
            better_assert( i_s.size() == 1 );
            better_assert( i_s[0][0] >= 1 );
            return ( ( (i_s[0][0] * sizeof(std::float32_t))  >> 3 ) + 1 ) << 4;
        }
    }; //node_::layer_norm

    struct gemm : node
    {
        void implement_arithmetic_operation( std::vector<std::byte*>const&, std::vector<std::vector<std::int64_t>>const&, std::vector<std::byte*>const&, std::vector<std::vector<std::int64_t>>const&, std::vector<std::byte*>const&, std::vector<std::vector<std::int64_t>>const&, std::byte*, std::int64_t, stream<default_engine_type>& ) override;

        std::vector<std::vector<std::int64_t>> inference_outputs_shape ( std::vector<std::vector<std::int64_t>> const& weights_shape, std::vector<std::vector<std::int64_t>> const& inputs_shape ) override
        {
            better_assert( weights_shape.size() == 1 );
            better_assert( 2 == weights_shape[0].size() );
            better_assert( inputs_shape.size() == 1 );
            better_assert( 2 == inputs_shape[0].size() );
            std::vector<std::vector<std::int64_t>> ans;
            {
                std::vector<std::int64_t> output_shape{ { inputs_shape[0][0], weights_shape[0][1], } };
                ans.push_back( output_shape );
            }
            return ans;
        }
    }; //node_::gemm

    struct linear : node
    {
        void implement_arithmetic_operation( std::vector<std::byte*>const&, std::vector<std::vector<std::int64_t>>const&, std::vector<std::byte*>const&, std::vector<std::vector<std::int64_t>>const&, std::vector<std::byte*>const&, std::vector<std::vector<std::int64_t>>const&, std::byte*, std::int64_t, stream<default_engine_type>& ) override;

        std::vector<std::vector<std::int64_t>> inference_outputs_shape ( std::vector<std::vector<std::int64_t>> const& weights_shape, std::vector<std::vector<std::int64_t>> const& inputs_shape ) override
        {
            better_assert( weights_shape.size() == 2 ); // y = W x + b
            better_assert( 2 == weights_shape[0].size() );
            better_assert( inputs_shape.size() == 1 );
            better_assert( 2 == inputs_shape[0].size() );
            std::vector<std::vector<std::int64_t>> ans;
            {
                std::vector<std::int64_t> output_shape{ { inputs_shape[0][0], weights_shape[0][1], } };
                ans.push_back( output_shape );
            }
            spdlog::info( nnl::format( "inference_outputs_shape with node {} id {}: {}", (*this).name(), (*this).id(), ans) );
            return ans;
        }
    }; //node_::linear


    struct scaled_offset : node
    {
        void implement_arithmetic_operation( std::vector<std::byte*>const&, std::vector<std::vector<std::int64_t>>const&, std::vector<std::byte*>const&, std::vector<std::vector<std::int64_t>>const&, std::vector<std::byte*>const&, std::vector<std::vector<std::int64_t>>const&, std::byte*, std::int64_t, stream<default_engine_type>& ) override;

        std::vector<std::vector<std::int64_t>> inference_outputs_shape ( std::vector<std::vector<std::int64_t>> const& weights_shape, std::vector<std::vector<std::int64_t>> const& inputs_shape ) override
        {
            better_assert( (3==inputs_shape.size()) || ((1==inputs_shape.size()) && (2==weights_shape.size())) );

            std::vector<std::vector<std::int64_t>> ans;
            ans.push_back( inputs_shape[0] );
            return ans;
        }
    }; //node_::scaled_offset

    struct attention : node
    {
        void implement_arithmetic_operation( std::vector<std::byte*>const&, std::vector<std::vector<std::int64_t>>const&, std::vector<std::byte*>const&, std::vector<std::vector<std::int64_t>>const&, std::vector<std::byte*>const&, std::vector<std::vector<std::int64_t>>const&, std::byte*, std::int64_t, stream<default_engine_type>& ) override;

        std::int64_t calculate_buffer_size_in_bytes
        (
            [[maybe_unused]] std::vector<std::vector<std::int64_t>> const& w_s,
            [[maybe_unused]] std::vector<std::vector<std::int64_t>> const& i_s
        ) override
        {
            /*
             * Attention node has 3 inputs, q, k and v. Dimensions are [n_q, d_k], [n_k, d_k] and [n_k, d_v]
             * The buffer size is supposed to be  n_q x n_k
             */
            better_assert( 0 == w_s.size() );
            better_assert( 3 == i_s.size() );
            better_assert( 2 == i_s[0].size() );
            better_assert( 2 == i_s[1].size() );

            return i_s[0][0] * i_s[1][0] * sizeof(std::float32_t);
        }

        std::vector<std::vector<std::int64_t>> inference_outputs_shape ( [[maybe_unused]]std::vector<std::vector<std::int64_t>> const& weights_shape, std::vector<std::vector<std::int64_t>> const& inputs_shape ) override
        {
            better_assert( 3 == inputs_shape.size() );
            better_assert( 2 == inputs_shape[0].size() );
            better_assert( 2 == inputs_shape[1].size() );
            better_assert( 2 == inputs_shape[2].size() );
            better_assert( 0 == weights_shape.size() );

            std::vector<std::vector<std::int64_t>> ans;
            ans.push_back( inputs_shape[0] );
            ans[0][1] = inputs_shape[2][1];
            return ans;
        }

    }; //node_::attention

    // we use this to calculate the dot-product of Q K^T for query product attention
    struct query_product : node
    {
        void implement_arithmetic_operation( std::vector<std::byte*>const&, std::vector<std::vector<std::int64_t>>const&, std::vector<std::byte*>const&, std::vector<std::vector<std::int64_t>>const&, std::vector<std::byte*>const&, std::vector<std::vector<std::int64_t>>const&, std::byte*, std::int64_t, stream<default_engine_type>& ) override;


        std::vector<std::vector<std::int64_t>> inference_outputs_shape ( [[maybe_unused]]std::vector<std::vector<std::int64_t>> const& weights_shape, std::vector<std::vector<std::int64_t>> const& inputs_shape ) override
        {
            better_assert( 2 == inputs_shape.size() );
            better_assert( 2 == inputs_shape[0].size() );
            better_assert( 2 == inputs_shape[1].size() );
            better_assert( 0 == weights_shape.size() );

            better_assert( inputs_shape[0][1] == inputs_shape[1][1] );

            std::vector<std::vector<std::int64_t>> ans;;
            ans.push_back( inputs_shape[0] );
            ans[0][1] = inputs_shape[1][0];
            return ans;
        }
    }; //node_::query_product

    // we use this node to calculate the scaled mask of the query product
    struct scaled_mask : node
    {
        void implement_arithmetic_operation( std::vector<std::byte*>const&, std::vector<std::vector<std::int64_t>>const&, std::vector<std::byte*>const&, std::vector<std::vector<std::int64_t>>const&, std::vector<std::byte*>const&, std::vector<std::vector<std::int64_t>>const&, std::byte*, std::int64_t, stream<default_engine_type>& ) override;

        std::vector<std::vector<std::int64_t>> inference_outputs_shape ( std::vector<std::vector<std::int64_t>> const& weights_shape, std::vector<std::vector<std::int64_t>> const& inputs_shape ) override
        {
            better_assert( weights_shape.size() == 0 ); // no weights
            better_assert( inputs_shape.size() == 1 ); // only one input
            better_assert( inputs_shape[0].size() == 2 ); // only one matrix for input
            better_assert( inputs_shape[0][0] == inputs_shape[0][1] ); // only one squared-matrix for input
            return inputs_shape;
        }
    }; //node_::scaled_mask

    // we use this to calculate the dot-product of A \dot  B for two matrices
    struct multiply : node
    {
        void implement_arithmetic_operation( std::vector<std::byte*>const&, std::vector<std::vector<std::int64_t>>const&, std::vector<std::byte*>const&, std::vector<std::vector<std::int64_t>>const&, std::vector<std::byte*>const&, std::vector<std::vector<std::int64_t>>const&, std::byte*, std::int64_t, stream<default_engine_type>& ) override;

        std::vector<std::vector<std::int64_t>> inference_outputs_shape ( [[maybe_unused]]std::vector<std::vector<std::int64_t>> const& weights_shape, std::vector<std::vector<std::int64_t>> const& inputs_shape ) override
        {
            better_assert( 2 == inputs_shape.size() );
            better_assert( 2 == inputs_shape[0].size() );
            better_assert( 2 == inputs_shape[1].size() );
            better_assert( 0 == weights_shape.size() );

            better_assert( inputs_shape[0][1] == inputs_shape[1][0] );

            std::vector<std::vector<std::int64_t>> ans;;
            ans.push_back( inputs_shape[0] );
            ans[0][1] = inputs_shape[1][1];
            return ans;
        }
    }; //node_::multiply

    struct multi_head_attention : node
    {
        void implement_attributed_arithmetic_operation( std::vector<std::byte*>const&, std::vector<std::vector<std::int64_t>>const&, std::vector<std::byte*>const&, std::vector<std::vector<std::int64_t>>const&, std::vector<std::byte*>const&, std::vector<std::vector<std::int64_t>>const&, std::byte*, std::int64_t, stream<default_engine_type>&, std::vector<attribute> const& ) override;

        std::int64_t calculate_buffer_size_in_bytes
        (
            [[maybe_unused]] std::vector<std::vector<std::int64_t>> const& weights_shape,
            [[maybe_unused]] std::vector<std::vector<std::int64_t>> const& inputs_shape
        ) override
        {
            std::int64_t heads = -1;
            for ( auto const& att : (*this).attributes() )
                if ( att.name() == "heads" )
                    heads = att.heads().value();
            better_assert( 0 < heads );

            if constexpr( debug_mode )
            {
                spdlog::info( "Found {} heads in the multi_head_attention node." );
            }

            std::int64_t const n_seq = inputs_shape[0][0];
            std::int64_t const n_embd = inputs_shape[0][1];
            return ((( sizeof(std::float32_t) * (n_seq * 4 * n_embd + n_seq * n_seq * heads) ) >> 4) + 1 ) << 4;
        }

        std::vector<std::vector<std::int64_t>> inference_outputs_shape ( [[maybe_unused]]std::vector<std::vector<std::int64_t>> const& weights_shape, std::vector<std::vector<std::int64_t>> const& inputs_shape ) override
        {
            better_assert( inputs_shape.size() == 1 ); // only 1 input
            better_assert( 2 == inputs_shape[0].size() ); // input is a matrix
            better_assert( weights_shape.size() == 4 ); // w_att, b_att, w_pro, b_pro
            better_assert( weights_shape[0].size() == 2);
            better_assert( weights_shape[1].size() == 1);
            better_assert( weights_shape[2].size() == 2);
            better_assert( weights_shape[3].size() == 1);

            std::int64_t const n_embd = inputs_shape[0][1];

            // check w_att
            better_assert( weights_shape[0][0] == n_embd );
            better_assert( weights_shape[0][1] == n_embd*3 );

            // check b_att
            better_assert( weights_shape[1][0] == n_embd*3 );

            // check w_proj
            better_assert( weights_shape[2][0] == n_embd );
            better_assert( weights_shape[2][1] == n_embd, nnl::format("Expect the 2nd dimension of W_proj to be {}, but got {}", n_embd, weights_shape[2][1]) );

            // check b_proj
            better_assert( weights_shape[3][0] == n_embd );

            return inputs_shape;
        }

    }; //node_::multi_head_attention

    // projection to vocabulary: layer_norm( input, alpha ) @ wte'
    struct vocabulary_projection : node
    {
        //void implement_attributed_arithmetic_operation( std::vector<std::byte*>const&, std::vector<std::vector<std::int64_t>>const&, std::vector<std::byte*>const&, std::vector<std::vector<std::int64_t>>const&, std::vector<std::byte*>const&, std::vector<std::vector<std::int64_t>>const&, std::byte*, std::int64_t, stream<default_engine_type>&, std::vector<std::shared_ptr<attribute>> const& ) override;
        void implement_arithmetic_operation( std::vector<std::byte*>const&, std::vector<std::vector<std::int64_t>>const&, std::vector<std::byte*>const&, std::vector<std::vector<std::int64_t>>const&, std::vector<std::byte*>const&, std::vector<std::vector<std::int64_t>>const&, std::byte*, std::int64_t, stream<default_engine_type>& ) override;

        std::vector<std::vector<std::int64_t>> inference_outputs_shape ( [[maybe_unused]]std::vector<std::vector<std::int64_t>> const& weights_shape, std::vector<std::vector<std::int64_t>> const& inputs_shape ) override
        {
            #if 0
            There are 3 weights:
                - gamma (scale) parameters for layer normalization, of shape (n_embd,)
                - beta (offset) parameters for layer normalization, of shape (n_embd,)
                - wte matrix, of shape (n_vocab, n_embd)
            There are 1 input:
                - input matrix, of shape (n_seq, n_embd)
            Output is of shape (n_embd, ), as only the last row is expected
            #endif

            better_assert( 1 == inputs_shape.size() ); // only 1 input
            better_assert( 3 == weights_shape.size() ); // 3 weights

            better_assert( 2 == inputs_shape[0].size() ); // 2D input shape
            better_assert( 1 == weights_shape[0].size() ); // scale param
            better_assert( 1 == weights_shape[1].size() ); // offset param
            better_assert( 2 == weights_shape[2].size() ); // wte

            std::int64_t const n_embd = inputs_shape[0][1];
            better_assert( n_embd == weights_shape[0][0] );
            better_assert( n_embd == weights_shape[1][0] );
            better_assert( n_embd == weights_shape[2][1] );

            //return make_vector( make_vector( n_embd ) );
            return make_vector( make_vector( n_embd ), make_vector(std::int64_t{/*2*/8}) ); // -> two outputs: wte[argmax(xxx)], and [argmax(xxx),max(xxx)]. The second output only need 2 positions, we assign 8 to align the memory
        }

        std::int64_t calculate_buffer_size_in_bytes
        (
            [[maybe_unused]] std::vector<std::vector<std::int64_t>> const& w_s,
            [[maybe_unused]] std::vector<std::vector<std::int64_t>> const& i_s
        ) override
        {
            better_assert( 3 == w_s.size() );
            better_assert( 1 == i_s.size() );
            better_assert( 2 == i_s[0].size() );
            std::int64_t const n_embd = i_s[0][1];
            std::int64_t const n_voc = w_s[2][0];
            std::int64_t const max_buffer_size = std::max( n_embd, n_voc ) + 2;
            // as only the last row will be reduced using layer normalization operator, O(1) extra space is needed
            return sizeof(float) * (((max_buffer_size >> 4) + 1) << 5);
        }

    }; //node_::vocabulary_projection


}//namespace node_

static const std::unordered_map<std::string, std::function<node()>> node_builder
{
    {
        "Input",
        []()
        {
            return node{ std::make_shared<node_::input>() };
        }
    },
    {
        "Gelu",
        []()
        {
            return node{ std::make_shared<node_::gelu>() };
        }
    },
    {
        "Add",
        []()
        {
            return node{ std::make_shared<node_::add>() };
        }
    },
    {
        "Softmax",
        []()
        {
            return node{ std::make_shared<node_::softmax>() };
        }
    },
    {
        "LayerNorm",
        []()
        {
            return node{ std::make_shared<node_::layer_norm>() };
        }
    },
    {
        "Gemm",
        []()
        {
            return node{ std::make_shared<node_::gemm>() };
        }
    },
    {
        "Linear",
        []()
        {
            return node{ std::make_shared<node_::linear>() };
        }
    },
    {
        "Dense", // <-- Dense is an alias name for Linear
        []()
        {
            return node{ std::make_shared<node_::linear>() };
        }
    },
    {
        "ScaledOffset",
        []()
        {
            return node{ std::make_shared<node_::scaled_offset>() };
        }
    },
    {
        #if 0
            Attention layer takes 3 inputs, [q, k, v], and produces one output o,  requires a buffer b

            tensors          q           k           v           o          b
            dimensions  [n_q, d_k]  [n_k, d_k]   [n_k, d_v]  [n_q, d_v]  [n_q, n_k]
        #endif
        "Attention",
        []()
        {
            return node{ std::make_shared<node_::attention>() };
        }
    },
    {
        #if 0
            QueryProductAttention layer takes 2 inputs, [q, k], and produces one output o
            basically it calculates $q \dot k^T$
            tensors          q           k           o
            dimensions  [n_q, d_k]  [n_k, d_k]   [n_q, n_k]
        #endif
        "QueryProduct",
        []()
        {
            return node{ std::make_shared<node_::query_product>() };
        }
    },
    {
        "ScaledMask",
        []()
        {
            return node{ std::make_shared<node_::scaled_mask>() };
        }
    },
    {
        "Multiply", // multiplies two matrices
        []()
        {
            return node{ std::make_shared<node_::multiply>() };
        }
    },
    {
        "MultiHeadAttention",
        []()
        {
            return node{ std::make_shared<node_::multi_head_attention>() };
        }
    },
    {
        "VocabularyProjection",
        []()
        {
            return node{ std::make_shared<node_::vocabulary_projection>() };
        }
    }
};


inline node make_node( std::string const& type, std::string const& name )
{
#if 0
    List of not to implement, https://github.com/jaymody/picoGPT/blob/29e78cc52b58ed2c1c483ffea2eb46ff6bdec785/gpt2_pico.py#L3-L58
        1. gelu
        2. softmax
        3. layer_norm
        4. linear/dense
        5. ffn
        6. attention
        7. split <- np.split
        8. tri  <- np.tri
        9. hstack <- np.hstack
        10. + //<- add
        11. argmax <- np.argmax
        12. append <- np.append
#endif
    if ( auto itor = node_builder.find(type); itor != node_builder.end() )
    {
        node ans = (itor->second)();
        ans.set_name( name );
        return ans;
    }

    spdlog::error( "Failed to make_node with type {}.", type );
    return {nullptr};
}


}//namespace nnl

// for unordered_map/set
namespace std
{

    template<>
    struct hash<nnl::node>
    {
        std::size_t operator()( nnl::node const& n ) const noexcept
        {
            if ( !n ) return 0;
            return std::hash<std::int64_t>{}( n.id() );
        }
    }; // hash node
}//namespace std







#endif//FVGYPNDHHNEYLIATJROIVBVHMVPWHPHPLPSFOGCLYNYRHUHAKRTPAAIPKPFKXVEVOQASAFMIH


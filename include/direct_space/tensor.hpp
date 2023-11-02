#ifndef MBOIIYRBNSLGVSYYOBORDXWUMFEWDMCVDCEOLAIDNCRMXHOSSVJUGDOPXLETNRSKEIOJTRKUI
#define MBOIIYRBNSLGVSYYOBORDXWUMFEWDMCVDCEOLAIDNCRMXHOSSVJUGDOPXLETNRSKEIOJTRKUI

#include "../utility/utility.hpp"
#include "./allocator.hpp"
#include "./engine.hpp"

namespace nnl
{

namespace tensor_
{

    template < typename Tensor, Dtype T >
    struct enable_dtype
    {
        std::string dtype() const
        {
            typedef T value_type;
            if constexpr ( std::is_same_v< std::int8_t, value_type > )
            {
                return std::string { "int8" };
            }
            else if constexpr ( std::is_same_v< std::uint8_t, value_type > )
            {
                return std::string { "uint8" };
            }
            else if constexpr ( std::is_same_v< std::int16_t, value_type > )
            {
                return std::string { "int16" };
            }
            else if constexpr ( std::is_same_v< std::uint16_t, value_type > )
            {
                return std::string { "uint16" };
            }
            else if constexpr ( std::is_same_v< std::int32_t, value_type > )
            {
                return std::string { "int32" };
            }
            else if constexpr ( std::is_same_v< std::uint32_t, value_type > )
            {
                return std::string { "uint32" };
            }
            else if constexpr ( std::is_same_v< std::int64_t, value_type > )
            {
                return std::string { "int64" };
            }
            else if constexpr ( std::is_same_v< std::uint64_t, value_type > )
            {
                return std::string { "uint64" };
            }
            else if constexpr ( std::is_same_v< std::float16_t, value_type > )
            {
                return std::string { "float16" };
            }
            else if constexpr ( std::is_same_v< std::float32_t, value_type > )
            {
                return std::string { "float32" };
            }
            else if constexpr ( std::is_same_v< std::float64_t, value_type > )
            {
                return std::string { "float64" };
            }
            else
            {
                return {};
            }
        }
    }; // struct enable_dtype

    template < Dtype T, class Host_Allocator >
    struct tensor : enable_id< tensor< T, Host_Allocator > >,
                    enable_dtype< tensor< T, Host_Allocator >, T >
    {
        typedef T value_type;

        std::shared_ptr< std::vector< std::byte, Host_Allocator > > data_; // <-- save host blob here
        std::vector< std::int64_t >                                 shape_;

        std::int64_t use_count() const noexcept
        {
            if ( !data_ ) return 0;
            return data_.use_count();
        }

        tensor( std::vector< std::int64_t > const& shape ) : shape_ { shape }
        {
            std::int64_t const size = ( 0 == shape_.size() ) ? 0 : std::accumulate( shape_.begin(), shape_.end(), 1LL, []( auto x, auto y ) { return x * y; } );
            assert( size >= 0 );
            data_ = std::make_shared< std::vector< std::byte, Host_Allocator > >();
            data_->reserve( (((size*sizeof(value_type))>>4)+1)<<4 );
            data_->resize( size * sizeof( value_type ) );
        }

        void reshape( std::vector< std::int64_t > const& new_shape )
        {
            shape_                  = new_shape;
            std::int64_t const size = ( 0 == shape_.size() ) ? 0 : std::accumulate( shape_.begin(), shape_.end(), 1LL, []( auto x, auto y ) { return x * y; } );
            std::int64_t const max_bytes_to_reserve =  std::max( ((((size*sizeof(value_type))>>4)+1) << 4), data_->capacity() );
            data_->reserve( max_bytes_to_reserve );
            data_->resize( size * sizeof( value_type ) );
        }

        void reserve( std::vector< std::int64_t > const& new_shape )
        {
            std::int64_t const size = ( 0 == new_shape.size() ) ? 0 : std::accumulate( new_shape.begin(), new_shape.end(), 1LL, []( auto x, auto y ) { return x * y; } );
            std::int64_t const max_bytes_to_reserve =  std::max( ((((size*sizeof(value_type))>>4)+1) << 4), data_->capacity() );
            data_->reserve( max_bytes_to_reserve );
        }


        std::vector< std::int64_t > shape() const
        {
            return shape_;
        }

        std::byte* data()
        {
            return data_->data();
        }

        std::byte const* data() const
        {
            return data_->data();
        }

        std::int64_t size() const
        {
            return data_->size();
        }

        // reallocation aware clear
        void clear()
        {
            data_->clear();
            data_->shrink_to_fit();
            shape_.clear();
        }

        ~tensor()
        {
            data_ = nullptr;
            shape_.clear();
        }

    }; // struct nnl::tensor_::tensor

} // namespace tensor_

// forward declaration of session
template < Engine E >
struct session;

template < Engine E >
session< E >& get_default_session();

template < Engine E >
struct tensor
{
    std::variant<
        tensor_::tensor< std::int8_t, host_allocator< E, std::byte > >,
        tensor_::tensor< std::uint8_t, host_allocator< E, std::byte > >,
        tensor_::tensor< std::int16_t, host_allocator< E, std::byte > >,
        tensor_::tensor< std::uint16_t, host_allocator< E, std::byte > >,
        tensor_::tensor< std::int32_t, host_allocator< E, std::byte > >,
        tensor_::tensor< std::uint32_t, host_allocator< E, std::byte > >,
        tensor_::tensor< std::int64_t, host_allocator< E, std::byte > >,
        tensor_::tensor< std::uint64_t, host_allocator< E, std::byte > >,
        tensor_::tensor< std::float16_t, host_allocator< E, std::byte > >,
        tensor_::tensor< std::float32_t, host_allocator< E, std::byte > >,
        tensor_::tensor< std::float64_t, host_allocator< E, std::byte > > >
        data_;

    std::string name_;

    template < typename T >
    void import_from( T* src )
    {
        std::copy_n( reinterpret_cast< std::byte* >( src ), ( *this ).size(), ( *this ).data() );
    }

    template < typename T >
    void import_from( T const* src )
    {
        std::copy_n( reinterpret_cast< std::byte* >( const_cast<T*>(src) ), ( *this ).size(), ( *this ).data() );
    }

    // designeded to load quick storage of data from numpy.ndarray.tofile
    bool load_memory( std::string const& memory_file_path )
    {
        if constexpr (debug_mode)
        {
            spdlog::info( nnl::format( "Tensor {} is trying to load memory from file {} with tensor shape {}", (*this).name(), memory_file_path, (*this).shape() ) );
        }

        std::ifstream ifs{ memory_file_path, std::ios::binary };
        if ( !ifs )
        {
            spdlog::error( "tensor::load_memory: failed to open file {}", memory_file_path );
            return false;
        }

        std::vector< char > buffer{ ( std::istreambuf_iterator< char >( ifs ) ), ( std::istreambuf_iterator< char >() ) };
        if constexpr (debug_mode)
        {
            spdlog::info( nnl::format( "loaded buffer size {}", buffer.size() ) );
        }

        {
            if ( std::distance( buffer.begin(), buffer.end() ) != size() )
            {
                spdlog::error( "tensor::load_memory: memory size mismatch. Expecting {} bytes, but got {} bytes from file {}",
                               size(), std::distance(buffer.begin(), buffer.end()), memory_file_path );
                return false;
            }
        }

        if constexpr (debug_mode)
        {
            spdlog::info( nnl::format( "Tensor {} loads memory from file {} with tensor shape {} and size {}", (*this).name(), memory_file_path, (*this).shape(), (*this).size() ) );
        }
        //std::copy( buffer.begin(), buffer.end(), reinterpret_cast<char*>(data()) );
        std::copy_n( buffer.begin(), (*this).size(), reinterpret_cast<char*>(data()) );

        if  constexpr (debug_mode)
        {
            if ( memory_file_path == "./examples/gpt2-1558M/assets/0.ln_1.b.bin" )
            {
                (*this).save_txt( "./0.ln_1.b.txt" );
            }
        }


        return true;
    }

    bool save_memory( std::string const& memory_file_path )
    {
        std::ofstream ofs{ memory_file_path, std::ios::binary };
        if ( !ofs )
        {
            spdlog::error( "tensor::save_memory: failed to open file {}", memory_file_path );
            return false;
        }

        spdlog::info( "memory loaded from {}", memory_file_path );
        std::copy_n( reinterpret_cast<char*>(data()), size(), std::ostream_iterator<char>{ ofs, "" } );
        return true;
    }

    //TODO
    // purpose: swap the data from pinned memory to mmap memory
    bool swap_out( std::string const& swap_file_path );
    //TODO
    // purpose: swap the data to pinned memory from mmap memory
    bool swap_in( std::string const& swap_file_path );


    // TODO
    bool load_txt( std::string const& txt_file_path, std::string const& delimiter = std::string { "\t" } );

    bool save_txt( std::string const& txt_file_path, std::string const& delimiter = std::string { "\t" } )
    {

        auto const& _save_txt = [this]< typename T >( std::string const& txt_file_path, std::string const& delimiter ) -> bool {
            std::int64_t lines             = 1;
            std::int64_t elements_per_line = ( *this ).size()/sizeof(T);
            if ( ( *this ).shape().size() == 2 )
            {
                lines             = *( ( *this ).shape().begin() );
                elements_per_line = *( ( *this ).shape().rbegin() );
            }
            std::ofstream ofs { txt_file_path };
            if ( !ofs )
            {
                spdlog::error( "tensor::save_txt: failed to open file {}", txt_file_path );
                return false;
            }

            if constexpr ( std::is_same_v< T, std::float32_t > )
            {
                ofs.precision( 7 ); // <- config in config.hpp?
            }
            if constexpr ( std::is_same_v< T, std::float64_t > )
            {
                ofs.precision( 16 ); // <- config in config.hpp?
            }

            // TODO: how to export float16?
            if constexpr ( std::is_same_v< T, std::float16_t > )
            {
                std::ostream_iterator< std::uint16_t > output_iterator { ofs, delimiter.c_str() };
                for ( auto idx : range( lines ) )
                {
                    std::int64_t const start = idx * elements_per_line;
                    std::int64_t const end   = start + elements_per_line;
                    std::copy( reinterpret_cast< std::uint16_t* >( ( *this ).data() ) + start, reinterpret_cast< std::uint16_t* >( ( *this ).data() ) + end, output_iterator );
                    ofs << "\n";
                }
            }
            else
            {
                std::ostream_iterator< T > output_iterator { ofs, delimiter.c_str() };
                for ( auto idx : range( lines ) )
                {
                    std::int64_t const start = idx * elements_per_line;
                    std::int64_t const end   = start + elements_per_line;
                    std::copy( reinterpret_cast< T* >( ( *this ).data() ) + start, reinterpret_cast< T* >( ( *this ).data() ) + end, output_iterator );
                    ofs << "\n";
                }
            }
            ofs.close();

            return true;
        };

        std::string const& dt = ( ( *this ).dtype() );
        {
            if ( dt == std::string { "int8" } )
            {
                return _save_txt.template operator()< std::int8_t >( txt_file_path, delimiter );
            }
            if ( dt == std::string { "uint8" } )
            {
                return _save_txt.template operator()< std::uint8_t >( txt_file_path, delimiter );
            }
            if ( dt == std::string { "int16" } )
            {
                return _save_txt.template operator()< std::int16_t >( txt_file_path, delimiter );
            }
            if ( dt == std::string { "uint16" } )
            {
                return _save_txt.template operator()< std::uint16_t >( txt_file_path, delimiter );
            }
            if ( dt == std::string { "int32" } )
            {
                return _save_txt.template operator()< std::int32_t >( txt_file_path, delimiter );
            }
            if ( dt == std::string { "uint32" } )
            {
                return _save_txt.template operator()< std::uint32_t >( txt_file_path, delimiter );
            }
            if ( dt == std::string { "int64" } )
            {
                return _save_txt.template operator()< std::int64_t >( txt_file_path, delimiter );
            }
            if ( dt == std::string { "uint64" } )
            {
                return _save_txt.template operator()< std::uint64_t >( txt_file_path, delimiter );
            }
            if ( dt == std::string { "float16" } )
            {
                return _save_txt.template operator()< std::float16_t >( txt_file_path, delimiter );
            }
            if ( dt == std::string { "float32" } )
            {
                return _save_txt.template operator()< std::float32_t >( txt_file_path, delimiter );
            }
            if ( dt == std::string { "float64" } )
            {
                return _save_txt.template operator()< std::float64_t >( txt_file_path, delimiter );
            }

            spdlog::error( "tensor::save_txt: unknown dtype {}.", ( *this ).dtype() );
        }

        return false;
    }

    std::string name() const
    {
        return name_;
    }

    void set_name( std::string const& name )
    {
        better_assert( name.size(), "tensor::set_name: name cannot be empty." );
        name_ = name;
    }

    std::int64_t use_count() const noexcept
    {
        return std::visit( []( auto const& ts ) { return ts.use_count(); }, data_ );
    }

    void reshape( std::vector< std::int64_t > const& new_shape )
    {
        std::visit( [&new_shape]( auto& ts ) { return ts.reshape( new_shape ); }, data_ );
    }

    // alias of reshape
    void resize( std::vector< std::int64_t > const& new_shape )
    {
        reshape( new_shape );
    }

    // reserve memory for an enven large shape
    void reserve( std::vector< std::int64_t > const& new_shape )
    {
        std::visit( [&new_shape]( auto& ts ) { return ts.reserve( new_shape ); }, data_ );
    }

    std::byte const* data() const
    {
        return std::visit( []( auto const& ts ) { return ts.data(); }, data_ );
    }

    std::byte* data()
    {
        return std::visit( []( auto& ts ) { return ts.data(); }, data_ );
    }

    std::int64_t id() const
    {
        return std::visit( []( auto const& ts ) { return ts.id(); }, data_ );
    }

    std::int64_t size() const
    {
        return std::visit( []( auto const& ts ) { return ts.size(); }, data_ );
    }

    std::vector< std::int64_t > shape() const
    {
        return std::visit( []( auto const& ts ) { return ts.shape(); }, data_ );
    }

    std::string dtype() const
    {
        return std::visit( []( auto const& ts ) { return ts.dtype(); }, data_ );
    }

    //
    // Note: intentionally trigger deallocation if not empty
    //
    void clear() const
    {
        return std::visit( []( auto const& ts ) { return ts.clear(); }, data_ );
    }

    template < Dtype T >
    tensor( tensor_::tensor< T, host_allocator< E, std::byte > > const& data,
        std::string const&                                              name = std::string {} )
        : data_ { data }
    {
        on_constructed( name );
    }

    template < Dtype T >
    tensor( tensor_::tensor< T, host_allocator< E, std::byte > >&& data,
        std::string const&                                         name = std::string {} )
        : data_ { std::move( data ) }
    {
        on_constructed( name );
    }

    tensor()                           = delete;
    tensor( tensor const& )            = default;
    tensor( tensor&& )                 = default;
    tensor& operator=( tensor const& ) = default;
    tensor& operator=( tensor&& )      = default;

    // copy device memory that mapped to this tensor back to host memory
    bool synchronize_to_host()
    {
        auto& sess = get_default_session< E >();
        return sess.tensor_to_host( ( *this ).id() );
    }

    //
    // | version int64 | dtype: string, 8 bytes | len: int64, 8 bytes | shape: int64, 8 * len bytes |  data: binary |
    //
    bool to_file( std::string const& file_name )
    {
        auto const& write_value = []< typename T >( std::ofstream& ofs, T value ) {
            ofs.write( reinterpret_cast< char const* >( std::addressof( value ) ), sizeof( T ) );
        };
        auto const& write_data = []< typename T >( std::ofstream& ofs, T* data, std::int64_t size_in_bytes ) {
            ofs.write( reinterpret_cast< char const* >( data ), size_in_bytes );
        };

        union u_tag
        {
            std::int64_t i_tag_;
            char         c_tag_[8];
        };

        std::ofstream ofs( file_name, std::ios::out | std::ios::binary );

        if ( !ofs )
        {
            spdlog::error( "tensor::to_file: cannot open file: {}", file_name );
            return false;
        }

        {
            write_value( ofs, version ); // write library version
            {
                u_tag              ut;
                std::string const& dt = dtype();
                std::fill( ut.c_tag_, ut.c_tag_ + 8, char {} );
                std::copy( dt.begin(), dt.end(), ut.c_tag_ );
                write_value( ofs, ut.i_tag_ ); // write dtype
            }
            std::vector< std::int64_t > const& sp = ( *this ).shape();
            write_value( ofs, static_cast< std::int64_t >( sp.size() ) ); // write length of shape
            {
                for ( auto idx : range( sp.size() ) )
                {
                    write_value( ofs, sp[idx] ); // write each element of shape
                }
            }
            write_data( ofs, data(), ( *this ).size() ); // write binary data
        }

        if ( !ofs.good() )
        {
            spdlog::error( "tensor::to_file: error happens when writing to file: {}", file_name );
            return false;
        }

        ofs.close();
        return true;
    }

private:
    void on_constructed( std::string const& name ) // const
    {
        auto& sess      = get_default_session< E >();
        ( *this ).name_ = ( name.size() == 0 ) ? ( std::string { "tensor-" } + std::to_string( ( *this ).id() ) ) : name;

        sess.register_tensor( ( *this ).name_, *this );
    }

}; // struct tensor

template < Engine E >
std::ostream& operator<<( std::ostream& os, tensor< E > const& tsor )
{
    os << "Tensor: <id> " << tsor.id() << ", <name> " << tsor.name() << ", <dtype> " << tsor.dtype() << ", <size> " << tsor.size() << ", <shape> ( ";
    for ( auto const& s : tsor.shape() )
    {
        os << s << ", ";
    }
    os << ") <use_count> " << tsor.use_count();
    os << " <memory>: " << std::hex << std::showbase << tsor.data();
    return os;
}

template < Engine E >
inline tensor< E > make_tensor( std::vector< std::int64_t > const& shape,
    std::string const& dtype, std::string const& name = std::string {} )
{
    if ( dtype == std::string { "int8" } )
    {
        return tensor< E > { tensor_::tensor< std::int8_t, host_allocator< E, std::byte > > { shape }, name };
    }

    if ( dtype == std::string { "uint8" } )
    {
        return tensor< E > { tensor_::tensor< std::uint8_t, host_allocator< E, std::byte > > { shape }, name };
    }

    if ( dtype == std::string { "int16" } )
    {
        return tensor< E > { tensor_::tensor< std::int16_t, host_allocator< E, std::byte > > { shape }, name };
    }

    if ( dtype == std::string { "uint16" } )
    {
        return tensor< E > { tensor_::tensor< std::uint16_t, host_allocator< E, std::byte > > { shape }, name };
    }

    if ( dtype == std::string { "int32" } )
    {
        return tensor< E > { tensor_::tensor< std::int32_t, host_allocator< E, std::byte > > { shape }, name };
    }

    if ( dtype == std::string { "uint32" } )
    {
        return tensor< E > { tensor_::tensor< std::uint32_t, host_allocator< E, std::byte > > { shape }, name };
    }

    if ( dtype == std::string { "int64" } )
    {
        return tensor< E > { tensor_::tensor< std::int64_t, host_allocator< E, std::byte > > { shape }, name };
    }

    if ( dtype == std::string { "uint64" } )
    {
        return tensor< E > { tensor_::tensor< std::uint64_t, host_allocator< E, std::byte > > { shape }, name };
    }

    if ( dtype == std::string { "float16" } )
    {
        return tensor< E > { tensor_::tensor< std::float16_t, host_allocator< E, std::byte > > { shape }, name };
    }

    if ( dtype == std::string { "float32" } )
    {
        return tensor< E > { tensor_::tensor< std::float32_t, host_allocator< E, std::byte > > { shape }, name };
    }

    if ( dtype == std::string { "float64" } )
    {
        return tensor< E > { tensor_::tensor< std::float64_t, host_allocator< E, std::byte > > { shape }, name };
    }

    spdlog::error( "Encountered unknown dtype {}. Only (u)int 8, 16, 32, 64 and float 16, 32 and 64 are supported.", dtype );
    std::abort();
}

template < Engine E >
inline tensor< E > make_tensor( std::initializer_list< std::int64_t > shape, std::string const& dtype, std::string const& name = std::string {} )
{
    //return make_tensor< E >( std::vector< std::int64_t > { shape }, dtype, name );
    return make_tensor< E >( make_vector( shape ), dtype, name );
}

template < Engine E >
inline tensor< E > make_tensor( std::string const& dtype, std::string const& name = std::string {} )
{
    return make_tensor< E >( std::vector< std::int64_t > {}, dtype, name );
}

//
// initializer tensor from memory/file
//
template < Engine E >
inline tensor< E > make_tensor_from_memory( std::initializer_list< std::int64_t > shape, std::string const& dtype, std::string const& memory_file, std::string const& name = std::string {} )
{
    //tensor<E> ans = make_tensor< E >( std::vector< std::int64_t > { shape }, dtype, name );
    tensor<E> ans = make_tensor< E >( make_vector( shape ), dtype, name );
    ans.load_memory( memory_file );
    return ans;
}

template < Engine E >
inline tensor< E > make_tensor_from_memory( std::initializer_list< std::int64_t > shape, std::string const& dtype, char const* const memory_file, std::string const& name = std::string {} )
{
    return make_tensor_from_memory<E>( shape, dtype, std::string { memory_file }, name );
}

template < Engine E, typename T >
inline tensor< E > make_tensor_from_memory( std::initializer_list< std::int64_t > shape, std::string const& dtype, T* memory, std::string const& name = std::string {} )
{
    //tensor<E> ans = make_tensor< E >( std::vector< std::int64_t > { shape }, dtype, name );
    tensor<E> ans = make_tensor< E >( make_vector( shape ), dtype, name );
    ans.import_from( memory );
    return ans;
}




// alias of `make_tensor_from_memory`
template < Engine E >
inline tensor< E > from_memory( std::initializer_list< std::int64_t > shape, std::string const& dtype, std::string const& memory_file, std::string const& name = std::string {} )
{
    return make_tensor_from_memory<E>( shape, dtype, memory_file, name );
}

// alias of `make_tensor_from_memory`
template < Engine E, typename T >
inline tensor< E > from_memory( std::initializer_list< std::int64_t > shape, std::string const& dtype, T* memory, std::string const& name = std::string {} )
{
    return make_tensor_from_memory<E>( shape, dtype, memory, name );
}

// only for test purpose
template < Engine E, typename T >
inline tensor< E > random( std::initializer_list< std::int64_t > shape, T min, T max, std::string const& dtype, std::string const& name = std::string {} )
{
    auto ans = make_tensor< E >( shape, dtype, name );
    {
        std::random_device               rd;
        std::mt19937                     gen( rd() );
        std::uniform_real_distribution<> dis( min, max );
        T*                               dat   = reinterpret_cast< T* >( ans.data() );
        std::int64_t const               total = std::accumulate( shape.begin(), shape.end(), std::int64_t { 1 }, []( std::int64_t a, std::int64_t b ) { return a * b; } );
        for ( auto idx : range( total ) )
        {
            dat[idx] = dis( gen );
        }
    }
    return ans;
}

template < Engine E, typename T >
inline tensor< E > random( std::initializer_list< std::int64_t > shape, std::string const& dtype, T min, T max, std::string const& name = std::string {} )
{
    return random< E >( shape, min, max, dtype, name );
}

//
// load serialized tensor
//
template < Engine E >
inline tensor< E > from_file( std::string const& file_name )
{
    std::ifstream ifs( file_name, std::ios::in | std::ios::binary );

    if ( !ifs )
    {
        spdlog::error( "nnl::from_file: cannot open file {} for input.", file_name );
        std::exit( -1 );
    }

    std::vector< char > buffer { ( std::istreambuf_iterator< char >( ifs ) ), ( std::istreambuf_iterator< char >() ) };

    std::string const dtype { buffer.data() + 8 }; // dtype in [8, 16)

    std::int64_t len;
    std::copy( buffer.data() + 16, buffer.data() + 24, reinterpret_cast< char* >( std::addressof( len ) ) ); // len in [16, 24)

    std::vector< std::int64_t > shape( len );
    std::copy( buffer.data() + 24, buffer.data() + 24 + 8 * len, reinterpret_cast< char* >( shape.data() ) ); // shape in [24, 24+8*len]

    tensor< E > ans = make_tensor< E >( shape, dtype );
    std::copy( buffer.data() + 24 + 8 * len, buffer.data() + buffer.size(), reinterpret_cast< char* >( ans.data() ) ); // data in [24+8*len]

    ifs.close();
    return ans;
}

// alias of `from_file`
template < Engine E >
inline tensor< E > make_tensor_from_file( std::string const& file_name )
{
    return from_file<E>( file_name );
}

} // namespace nnl

#endif // MBOIIYRBNSLGVSYYOBORDXWUMFEWDMCVDCEOLAIDNCRMXHOSSVJUGDOPXLETNRSKEIOJTRKUI

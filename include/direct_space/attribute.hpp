#ifndef OUGBSOFVIZJGUKRQYGVKCBXFFVVBQAIWGGPVBWXYRECVWQNXNRGOGOBYBTTONTLCUJVIXQJQEVFGDLOG
#define OUGBSOFVIZJGUKRQYGVKCBXFFVVBQAIWGGPVBWXYRECVWQNXNRGOGOBYBTTONTLCUJVIXQJQEVFGDLOG

#include "../utility/utility.hpp"

namespace cereal
{
    template<typename T> struct NameValuePair;
    template<typename T> NameValuePair<T> make_nvp( std::string const&, T&& );
    template<typename T> NameValuePair<T> make_nvp( char const*, T&& );
}//namespace cereal

namespace nnl
{


template< typename C >
struct attribute_method_heads
{
    typedef std::remove_cvref_t<std::int64_t> value_type;

    std::optional<value_type> heads() const
    {
        auto& self = static_cast<C const&>(*this);
        if ( self.name() != "heads" )
            return std::nullopt;
        if ( value_type const* p = std::get_if<value_type>( &self.value() ) )
            return std::optional<value_type>( *p );
        return std::nullopt;
    }

    value_type& heads()
    {
        auto& self = static_cast<C&>(*this);
        if ( "heads" != self.name() )
        {
            self.name() = "heads";
            self.value() = value_type{};
        }
        return std::get<value_type>( self.value() );
    }
}; //struct attribute_method_heads

template< typename C >
struct attribute_method_dilations
{
    typedef std::remove_cvref_t<std::int64_t> value_type;

    std::optional<value_type> dilations() const
    {
        auto& self = static_cast<C const&>(*this);
        if ( self.name() != "dilations" )
            return std::nullopt;
        if ( value_type const* p = std::get_if<value_type>( &self.value() ) )
            return std::optional<value_type>( *p );
        return std::nullopt;
    }

    value_type& dilations()
    {
        auto& self = static_cast<C&>(*this);
        if ( "dilations" != self.name() )
        {
            self.name() = "dilations";
            self.value() = value_type{};
        }
        return std::get<value_type>( self.value() );
    }
}; //struct attribute_method_dilations

template< typename C >
struct attribute_method_group
{
    typedef std::remove_cvref_t<std::int64_t> value_type;

    std::optional<value_type> group() const
    {
        auto& self = static_cast<C const&>(*this);
        if ( self.name() != "group" )
            return std::nullopt;
        if ( value_type const* p = std::get_if<value_type>( &self.value() ) )
            return std::optional<value_type>( *p );
        return std::nullopt;
    }

    value_type& group()
    {
        auto& self = static_cast<C&>(*this);
        if ( "group" != self.name() )
        {
            self.name() = "group";
            self.value() = value_type{};
        }
        return std::get<value_type>( self.value() );
    }
}; //struct attribute_method_group

template< typename C >
struct attribute_method_input_shape
{
    typedef std::remove_cvref_t<std::vector<std::int64_t>> value_type;

    std::optional<value_type> input_shape() const
    {
        auto& self = static_cast<C const&>(*this);
        if ( self.name() != "input_shape" )
            return std::nullopt;
        if ( value_type const* p = std::get_if<value_type>( &self.value() ) )
            return std::optional<value_type>( *p );
        return std::nullopt;
    }

    value_type& input_shape()
    {
        auto& self = static_cast<C&>(*this);
        if ( "input_shape" != self.name() )
        {
            self.name() = "input_shape";
            self.value() = value_type{};
        }
        return std::get<value_type>( self.value() );
    }
}; //struct attribute_method_input_shape

template< typename C >
struct attribute_method_kernel_shape
{
    typedef std::remove_cvref_t<std::vector<std::int64_t>> value_type;

    std::optional<value_type> kernel_shape() const
    {
        auto& self = static_cast<C const&>(*this);
        if ( self.name() != "kernel_shape" )
            return std::nullopt;
        if ( value_type const* p = std::get_if<value_type>( &self.value() ) )
            return std::optional<value_type>( *p );
        return std::nullopt;
    }

    value_type& kernel_shape()
    {
        auto& self = static_cast<C&>(*this);
        if ( "kernel_shape" != self.name() )
        {
            self.name() = "kernel_shape";
            self.value() = value_type{};
        }
        return std::get<value_type>( self.value() );
    }
}; //struct attribute_method_kernel_shape

template< typename C >
struct attribute_method_pads
{
    typedef std::remove_cvref_t<std::vector<std::int64_t>> value_type;

    std::optional<value_type> pads() const
    {
        auto& self = static_cast<C const&>(*this);
        if ( self.name() != "pads" )
            return std::nullopt;
        if ( value_type const* p = std::get_if<value_type>( &self.value() ) )
            return std::optional<value_type>( *p );
        return std::nullopt;
    }

    value_type& pads()
    {
        auto& self = static_cast<C&>(*this);
        if ( "pads" != self.name() )
        {
            self.name() = "pads";
            self.value() = value_type{};
        }
        return std::get<value_type>( self.value() );
    }
}; //struct attribute_method_pads

template< typename C >
struct attribute_method_strides
{
    typedef std::remove_cvref_t<std::vector<std::int64_t>> value_type;

    std::optional<value_type> strides() const
    {
        auto& self = static_cast<C const&>(*this);
        if ( self.name() != "strides" )
            return std::nullopt;
        if ( value_type const* p = std::get_if<value_type>( &self.value() ) )
            return std::optional<value_type>( *p );
        return std::nullopt;
    }

    value_type& strides()
    {
        auto& self = static_cast<C&>(*this);
        if ( "strides" != self.name() )
        {
            self.name() = "strides";
            self.value() = value_type{};
        }
        return std::get<value_type>( self.value() );
    }
}; //struct attribute_method_strides

template< typename C >
struct attribute_method_perm
{
    typedef std::remove_cvref_t<std::vector<std::int64_t>> value_type;

    std::optional<value_type> perm() const
    {
        auto& self = static_cast<C const&>(*this);
        if ( self.name() != "perm" )
            return std::nullopt;
        if ( value_type const* p = std::get_if<value_type>( &self.value() ) )
            return std::optional<value_type>( *p );
        return std::nullopt;
    }

    value_type& perm()
    {
        auto& self = static_cast<C&>(*this);
        if ( "perm" != self.name() )
        {
            self.name() = "perm";
            self.value() = value_type{};
        }
        return std::get<value_type>( self.value() );
    }
}; //struct attribute_method_perm

template< typename C >
struct attribute_method_axes
{
    typedef std::remove_cvref_t<std::vector<std::int64_t>> value_type;

    std::optional<value_type> axes() const
    {
        auto& self = static_cast<C const&>(*this);
        if ( self.name() != "axes" )
            return std::nullopt;
        if ( value_type const* p = std::get_if<value_type>( &self.value() ) )
            return std::optional<value_type>( *p );
        return std::nullopt;
    }

    value_type& axes()
    {
        auto& self = static_cast<C&>(*this);
        if ( "axes" != self.name() )
        {
            self.name() = "axes";
            self.value() = value_type{};
        }
        return std::get<value_type>( self.value() );
    }
}; //struct attribute_method_axes

template< typename C >
struct attribute_method_indices
{
    typedef std::remove_cvref_t<std::vector<std::int64_t>> value_type;

    std::optional<value_type> indices() const
    {
        auto& self = static_cast<C const&>(*this);
        if ( self.name() != "indices" )
            return std::nullopt;
        if ( value_type const* p = std::get_if<value_type>( &self.value() ) )
            return std::optional<value_type>( *p );
        return std::nullopt;
    }

    value_type& indices()
    {
        auto& self = static_cast<C&>(*this);
        if ( "indices" != self.name() )
        {
            self.name() = "indices";
            self.value() = value_type{};
        }
        return std::get<value_type>( self.value() );
    }
}; //struct attribute_method_indices

template< typename C >
struct attribute_method_axe
{
    typedef std::remove_cvref_t<std::int64_t> value_type;

    std::optional<value_type> axe() const
    {
        auto& self = static_cast<C const&>(*this);
        if ( self.name() != "axe" )
            return std::nullopt;
        if ( value_type const* p = std::get_if<value_type>( &self.value() ) )
            return std::optional<value_type>( *p );
        return std::nullopt;
    }

    value_type& axe()
    {
        auto& self = static_cast<C&>(*this);
        if ( "axe" != self.name() )
        {
            self.name() = "axe";
            self.value() = value_type{};
        }
        return std::get<value_type>( self.value() );
    }
}; //struct attribute_method_axe

template< typename C >
struct attribute_method_max
{
    typedef std::remove_cvref_t<std::float32_t> value_type;

    std::optional<value_type> max() const
    {
        auto& self = static_cast<C const&>(*this);
        if ( self.name() != "max" )
            return std::nullopt;
        if ( value_type const* p = std::get_if<value_type>( &self.value() ) )
            return std::optional<value_type>( *p );
        return std::nullopt;
    }

    value_type& max()
    {
        auto& self = static_cast<C&>(*this);
        if ( "max" != self.name() )
        {
            self.name() = "max";
            self.value() = value_type{};
        }
        return std::get<value_type>( self.value() );
    }
}; //struct attribute_method_max

template< typename C >
struct attribute_method_min
{
    typedef std::remove_cvref_t<std::float32_t> value_type;

    std::optional<value_type> min() const
    {
        auto& self = static_cast<C const&>(*this);
        if ( self.name() != "min" )
            return std::nullopt;
        if ( value_type const* p = std::get_if<value_type>( &self.value() ) )
            return std::optional<value_type>( *p );
        return std::nullopt;
    }

    value_type& min()
    {
        auto& self = static_cast<C&>(*this);
        if ( "min" != self.name() )
        {
            self.name() = "min";
            self.value() = value_type{};
        }
        return std::get<value_type>( self.value() );
    }
}; //struct attribute_method_min

template< typename C >
struct attribute_method_alpha
{
    typedef std::remove_cvref_t<std::float32_t> value_type;

    std::optional<value_type> alpha() const
    {
        auto& self = static_cast<C const&>(*this);
        if ( self.name() != "alpha" )
            return std::nullopt;
        if ( value_type const* p = std::get_if<value_type>( &self.value() ) )
            return std::optional<value_type>( *p );
        return std::nullopt;
    }

    value_type& alpha()
    {
        auto& self = static_cast<C&>(*this);
        if ( "alpha" != self.name() )
        {
            self.name() = "alpha";
            self.value() = value_type{};
        }
        return std::get<value_type>( self.value() );
    }
}; //struct attribute_method_alpha

template< typename C >
struct attribute_method_beta
{
    typedef std::remove_cvref_t<std::float32_t> value_type;

    std::optional<value_type> beta() const
    {
        auto& self = static_cast<C const&>(*this);
        if ( self.name() != "beta" )
            return std::nullopt;
        if ( value_type const* p = std::get_if<value_type>( &self.value() ) )
            return std::optional<value_type>( *p );
        return std::nullopt;
    }

    value_type& beta()
    {
        auto& self = static_cast<C&>(*this);
        if ( "beta" != self.name() )
        {
            self.name() = "beta";
            self.value() = value_type{};
        }
        return std::get<value_type>( self.value() );
    }
}; //struct attribute_method_beta

template< typename C >
struct attribute_method_to
{
    typedef std::remove_cvref_t<std::string> value_type;

    std::optional<value_type> to() const
    {
        auto& self = static_cast<C const&>(*this);
        if ( self.name() != "to" )
            return std::nullopt;
        if ( value_type const* p = std::get_if<value_type>( &self.value() ) )
            return std::optional<value_type>( *p );
        return std::nullopt;
    }

    value_type& to()
    {
        auto& self = static_cast<C&>(*this);
        if ( "to" != self.name() )
        {
            self.name() = "to";
            self.value() = value_type{};
        }
        return std::get<value_type>( self.value() );
    }
}; //struct attribute_method_to

template< typename C >
struct attribute_method_dtype
{
    typedef std::remove_cvref_t<std::string> value_type;

    std::optional<value_type> dtype() const
    {
        auto& self = static_cast<C const&>(*this);
        if ( self.name() != "dtype" )
            return std::nullopt;
        if ( value_type const* p = std::get_if<value_type>( &self.value() ) )
            return std::optional<value_type>( *p );
        return std::nullopt;
    }

    value_type& dtype()
    {
        auto& self = static_cast<C&>(*this);
        if ( "dtype" != self.name() )
        {
            self.name() = "dtype";
            self.value() = value_type{};
        }
        return std::get<value_type>( self.value() );
    }
}; //struct attribute_method_dtype

template< typename C >
struct attribute_field_value
{
    std::variant<std::int64_t, std::vector<std::int64_t>, std::float32_t, std::string> value_;
    constexpr std::variant<std::int64_t, std::vector<std::int64_t>, std::float32_t, std::string>& value() noexcept { return value_; }
    constexpr std::variant<std::int64_t, std::vector<std::int64_t>, std::float32_t, std::string> const& value() const noexcept { return value_; }
}; //struct attribute_field_value

template< typename C >
struct attribute_field_name
{
    std::string name_;
    constexpr std::string& name() noexcept { return name_; }
    constexpr std::string const& name() const noexcept { return name_; }
}; //struct attribute_field_name




struct attribute :
       attribute_method_heads<attribute>,
       attribute_method_dilations<attribute>,
       attribute_method_group<attribute>,
       attribute_method_input_shape<attribute>,
       attribute_method_kernel_shape<attribute>,
       attribute_method_pads<attribute>,
       attribute_method_strides<attribute>,
       attribute_method_perm<attribute>,
       attribute_method_axes<attribute>,
       attribute_method_indices<attribute>,
       attribute_method_axe<attribute>,
       attribute_method_max<attribute>,
       attribute_method_min<attribute>,
       attribute_method_alpha<attribute>,
       attribute_method_beta<attribute>,
       attribute_method_to<attribute>,
       attribute_method_dtype<attribute>,
       attribute_field_name<attribute>,
       attribute_field_value<attribute>
{
    //YOUR FANCY CODE
};//struct attribute

constexpr inline attribute make_attribute_heads( std::int64_t const& heads )
{
    attribute attribute_;
    attribute_.name() = "heads";
    attribute_.value() = heads;
    return attribute_;
}

constexpr inline attribute make_attribute_dilations( std::int64_t const& dilations )
{
    attribute attribute_;
    attribute_.name() = "dilations";
    attribute_.value() = dilations;
    return attribute_;
}

constexpr inline attribute make_attribute_group( std::int64_t const& group )
{
    attribute attribute_;
    attribute_.name() = "group";
    attribute_.value() = group;
    return attribute_;
}

constexpr inline attribute make_attribute_input_shape( std::vector<std::int64_t> const& input_shape )
{
    attribute attribute_;
    attribute_.name() = "input_shape";
    attribute_.value() = input_shape;
    return attribute_;
}

constexpr inline attribute make_attribute_kernel_shape( std::vector<std::int64_t> const& kernel_shape )
{
    attribute attribute_;
    attribute_.name() = "kernel_shape";
    attribute_.value() = kernel_shape;
    return attribute_;
}

constexpr inline attribute make_attribute_pads( std::vector<std::int64_t> const& pads )
{
    attribute attribute_;
    attribute_.name() = "pads";
    attribute_.value() = pads;
    return attribute_;
}

constexpr inline attribute make_attribute_strides( std::vector<std::int64_t> const& strides )
{
    attribute attribute_;
    attribute_.name() = "strides";
    attribute_.value() = strides;
    return attribute_;
}

constexpr inline attribute make_attribute_perm( std::vector<std::int64_t> const& perm )
{
    attribute attribute_;
    attribute_.name() = "perm";
    attribute_.value() = perm;
    return attribute_;
}

constexpr inline attribute make_attribute_axes( std::vector<std::int64_t> const& axes )
{
    attribute attribute_;
    attribute_.name() = "axes";
    attribute_.value() = axes;
    return attribute_;
}

constexpr inline attribute make_attribute_indices( std::vector<std::int64_t> const& indices )
{
    attribute attribute_;
    attribute_.name() = "indices";
    attribute_.value() = indices;
    return attribute_;
}

constexpr inline attribute make_attribute_axe( std::int64_t const& axe )
{
    attribute attribute_;
    attribute_.name() = "axe";
    attribute_.value() = axe;
    return attribute_;
}

constexpr inline attribute make_attribute_max( std::float32_t const& max )
{
    attribute attribute_;
    attribute_.name() = "max";
    attribute_.value() = max;
    return attribute_;
}

constexpr inline attribute make_attribute_min( std::float32_t const& min )
{
    attribute attribute_;
    attribute_.name() = "min";
    attribute_.value() = min;
    return attribute_;
}

constexpr inline attribute make_attribute_alpha( std::float32_t const& alpha )
{
    attribute attribute_;
    attribute_.name() = "alpha";
    attribute_.value() = alpha;
    return attribute_;
}

constexpr inline attribute make_attribute_beta( std::float32_t const& beta )
{
    attribute attribute_;
    attribute_.name() = "beta";
    attribute_.value() = beta;
    return attribute_;
}

constexpr inline attribute make_attribute_to( std::string const& to )
{
    attribute attribute_;
    attribute_.name() = "to";
    attribute_.value() = to;
    return attribute_;
}

constexpr inline attribute make_attribute_dtype( std::string const& dtype )
{
    attribute attribute_;
    attribute_.name() = "dtype";
    attribute_.value() = dtype;
    return attribute_;
}

// Example Usage:
//
// #include <YOU KNOW HOW TO INCLUDE CEREAL LIBS>
//
// auto m = make_attribute( ... );
// {
//     std::ofstream ofs( "attribute.json" );
//     cereal::JSONOutputArchive oarchive( ofs );
//     oarchive( cereal::make_nvp("attribute", m) );
// }
// {
//     std::ifstream ifs( "attribute.json" );
//     cereal::JSONInputArchive iarchive( ifs );
//     iarchive( m );
// }
//
template< typename Archive >
void serialize( Archive& ar, attribute& m )
{
    ar( cereal::make_nvp( "name", m.name() ) );
    ar( cereal::make_nvp( "value", m.value() ) );
}

} // namespace nnl

#endif//OUGBSOFVIZJGUKRQYGVKCBXFFVVBQAIWGGPVBWXYRECVWQNXNRGOGOBYBTTONTLCUJVIXQJQEVFGDLOG


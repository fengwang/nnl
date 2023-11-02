#ifndef NDLDUTTRTIPEFRYWESBIVBMGPXJGJQKRMKHUKUSCMJNODBHEIUJHLGOUQAVYJYXQUGPPNXOAJ
#define NDLDUTTRTIPEFRYWESBIVBMGPXJGJQKRMKHUKUSCMJNODBHEIUJHLGOUQAVYJYXQUGPPNXOAJ

#include "../include.hpp"

namespace nnl
{

    namespace
    {
        template< typename T, typename ... Args >
        constexpr std::vector<typename std::remove_cvref_t<T>> _make_vector( std::vector<typename std::remove_cvref_t<T>>& ans, T const& val, Args const& ... args )
        {
            ans.push_back( val );
            if constexpr ( sizeof...(args) > 0 )
                return _make_vector( ans, args... );
            else
                return ans;
        }
    }

    template< typename T, typename ... Args >
    constexpr std::vector<typename std::remove_cvref_t<T>> make_vector( T const& value, Args const& ... args )
    {
        std::vector<typename std::remove_cvref_t<T>> ans;
        ans.reserve( sizeof...(args) + 1 );
        return _make_vector( ans, value, args... );
    }

    template< typename T >
    constexpr std::vector<T> make_vector( std::initializer_list<T> lst )
    {
        std::vector<T> ans;
        ans.resize( lst.size() );
        std::copy( std::begin(lst), std::end(lst), std::begin(ans) );
        return ans;
    }


}//namespace nnl

#endif//NDLDUTTRTIPEFRYWESBIVBMGPXJGJQKRMKHUKUSCMJNODBHEIUJHLGOUQAVYJYXQUGPPNXOAJ

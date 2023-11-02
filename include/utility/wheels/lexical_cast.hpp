#ifndef TWKJVHPVYJLTDSREHUKPFEUMOISXDIECEEHXNIWVCPENRBMGSEQFOWIQRGCXSHXCHVOOAQRAC
#define TWKJVHPVYJLTDSREHUKPFEUMOISXDIECEEHXNIWVCPENRBMGSEQFOWIQRGCXSHXCHVOOAQRAC

#include "../include.hpp"

namespace nnl
{

    template <typename T, typename U>
    T const lexical_cast( U const& from )
    {
        T var;

        std::stringstream ss;
        ss << from;
        ss >> var;

        return var;
    }

}//namespace nnl

#endif//TWKJVHPVYJLTDSREHUKPFEUMOISXDIECEEHXNIWVCPENRBMGSEQFOWIQRGCXSHXCHVOOAQRAC

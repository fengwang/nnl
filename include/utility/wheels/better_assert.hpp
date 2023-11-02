#ifndef JWSTAWESJPDEXKGCYCJESXWYQEHAOVFVGMYWKYBLTCGWJLBSQIJCHASMPGUCMEQWUAWBBUBFK
#define JWSTAWESJPDEXKGCYCJESXWYQEHAOVFVGMYWKYBLTCGWJLBSQIJCHASMPGUCMEQWUAWBBUBFK

#include "../include.hpp"
#include "./color.hpp"

namespace private_namespace
{   // for macro `better_assert`
    template< typename... Args >
    void print_assertion([[maybe_unused]] std::ostream& out, [[maybe_unused]] Args&&... args)
    {
        if constexpr( nnl::ndebug_mode )
        {
        }
        else
        {
            out.precision( 20 );
            (out << ... << args) << std::endl;
            abort();
        }
    }
}

#ifdef better_assert
#undef better_assert
#endif

//
// enhancing 'assert' macro, usage:
//
// int a;
// ...
// better_assert( a > 0 ); //same as 'assert'
// better_assert( a > 0, "a is expected to be larger than 0, but actually a = " a ); //with more info dumped to std::cerr
//
#define better_assert(EXPRESSION, ... ) ((EXPRESSION) ? (void)0 : private_namespace::print_assertion(std::cerr, color::rize("Assertion Failure", "Red", "Default", "Bold"), ": '", color::rize(#EXPRESSION, "Green"), "' in File: ", color::rize(__FILE__, "Yellow"), " in Line: ",  color::rize(__LINE__, "Blue"), " "  __VA_OPT__(,) __VA_ARGS__))

#endif//JWSTAWESJPDEXKGCYCJESXWYQEHAOVFVGMYWKYBLTCGWJLBSQIJCHASMPGUCMEQWUAWBBUBFK


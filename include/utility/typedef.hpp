#ifndef UYEBHIBSBRAYIPXEFJOMFUTICMBPSAUQSCIFGCFLUEAFACUHWVKOQPXDGCRCBYMPVUXDOUJWM
#define UYEBHIBSBRAYIPXEFJOMFUTICMBPSAUQSCIFGCFLUEAFACUHWVKOQPXDGCRCBYMPVUXDOUJWM


#include "./wheels/float16_t.hpp"




// floats
namespace std
{
#ifndef __STDCPP_FLOAT16_T__
    #ifdef __GNUC__ // <-- for both gcc ang clang ...
        typedef _Float16 float16_t;
    #else
        typedef numeric::float16_t float16_t;
    #endif
#endif

#ifndef __STDCPP_FLOAT32_T__
    typedef float float32_t;
    //typedef float _Float32;
    //typedef double _Float32x;
#endif

#ifndef __STDCPP_FLOAT64_T__
    typedef double float64_t;
    //typedef double _Float64x;
    //typedef double _Float64;
#endif

#if 0
#ifndef __STDCPP_FLOAT128_T__
    typedef long double float128_t;
#endif
#endif

}//namespace std

#endif//UYEBHIBSBRAYIPXEFJOMFUTICMBPSAUQSCIFGCFLUEAFACUHWVKOQPXDGCRCBYMPVUXDOUJWM

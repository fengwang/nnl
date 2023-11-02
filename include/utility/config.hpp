#ifndef NUIVQSWDXLNHNIYYBJKSSCMFHEOUVAJPOOAFKXEEDIWVULWRXWRQCEOWGSULHYXSIFJRNNBJK
#define NUIVQSWDXLNHNIYYBJKSSCMFHEOUVAJPOOAFKXEEDIWVULWRXWRQCEOWGSULHYXSIFJRNNBJK

#include "./include.hpp"

namespace nnl
{

    #ifdef DEBUG
    inline constexpr std::int64_t debug_mode = 1;
    #else
    inline constexpr std::int64_t debug_mode = 0;
    #endif

    #ifdef NDEBUG
    inline constexpr std::int64_t ndebug_mode = 1;
    #else
    inline constexpr std::int64_t ndebug_mode = 0;
    #endif

    inline constexpr std::int64_t version = 20230505;

    #if defined(CUDA)
    inline constexpr std::int64_t cuda_mode = 1;
    inline constexpr std::int64_t rocm_mode = 0;
    inline constexpr std::int64_t opencl_mode = 0;
    #elif defined(ROCM)
    inline constexpr std::int64_t cuda_mode = 0;
    inline constexpr std::int64_t rocm_mode = 1;
    inline constexpr std::int64_t opencl_mode = 0;
    #elif defined(OPENCL)
    inline constexpr std::int64_t cuda_mode = 0;
    inline constexpr std::int64_t rocm_mode = 0;
    inline constexpr std::int64_t opencl_mode = 1;
    #else
    inline constexpr std::int64_t cuda_mode = 0;
    inline constexpr std::int64_t rocm_mode = 0;
    inline constexpr std::int64_t opencl_mode = 0;
    #endif

    #ifdef _MSC_VER
    inline constexpr std::int64_t is_windows_platform = 1;
    #else
    inline constexpr std::int64_t is_windows_platform = 0;
    #endif

}//namespace nnl

#endif//NUIVQSWDXLNHNIYYBJKSSCMFHEOUVAJPOOAFKXEEDIWVULWRXWRQCEOWGSULHYXSIFJRNNBJK


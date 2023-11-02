#ifndef BEKWLOVBSGMPWIYEPAAHUKFHRBKAHHBRCDGPORAHDBCPHIMXTKHQNEQJBHQVYEBLCKNHBQQRG
#define BEKWLOVBSGMPWIYEPAAHUKFHRBKAHHBRCDGPORAHDBCPHIMXTKHQNEQJBHQVYEBLCKNHBQQRG

#include "../utility/utility.hpp"

namespace nnl
{
    #if 0
        Dispatch algorithms for different engines.
        Working like tags.
    #endif

    struct dummy_engine{}; // will never be used
    struct cuda_engine{}; //WIP
    struct rocm_engine{}; //TODO
    struct opencl_engine{}; //TODO

    template< typename T >
    inline constexpr bool is_engine_v = std::is_same_v<T, cuda_engine> ||
                                        std::is_same_v<T, rocm_engine> ||
                                        std::is_same_v<T, opencl_engine>;

    template< typename T >
    concept Engine = is_engine_v<T>;

    inline auto constexpr get_default_engine() noexcept
    {
        if constexpr (cuda_mode)
        {
            return cuda_engine{};
        }
        else if constexpr (rocm_mode)
        {
            return rocm_engine{};
        }
        else if constexpr (opencl_mode)
        {
            return opencl_engine{};
        }
        else
        {
            return dummy_engine{};
        }
    }

    using default_engine_type = std::remove_cv_t<decltype( get_default_engine() )>;

}//namespace nnl

#endif//BEKWLOVBSGMPWIYEPAAHUKFHRBKAHHBRCDGPORAHDBCPHIMXTKHQNEQJBHQVYEBLCKNHBQQRG


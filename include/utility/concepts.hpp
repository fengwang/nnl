#ifndef QADXOBONMEVKFJNNPSFPIESJYGULMAQYXJJNJSHQVVAVUJHVWQYUVBFRBSOARUGVIYKINDGTV
#define QADXOBONMEVKFJNNPSFPIESJYGULMAQYXJJNJSHQVVAVUJHVWQYUVBFRBSOARUGVIYKINDGTV

#include "./typedef.hpp"

namespace nnl
{

    // types that are allowed to be instanced with the tensor
    template< typename T >
    inline constexpr bool is_tensor_dtype_v =
    std::is_same_v<T, std::int8_t> ||
    std::is_same_v<T, std::uint8_t> ||
    std::is_same_v<T, std::int16_t> ||
    std::is_same_v<T, std::uint16_t> ||
    std::is_same_v<T, std::int32_t> ||
    std::is_same_v<T, std::uint32_t> ||
    std::is_same_v<T, std::int64_t> ||
    std::is_same_v<T, std::uint64_t> ||
    std::is_same_v<T, std::float16_t> ||
    std::is_same_v<T, std::float32_t> ||
    std::is_same_v<T, std::float64_t>;

    template< typename T >
    concept Dtype = is_tensor_dtype_v<T>;


}//namespace nnl

#endif//QADXOBONMEVKFJNNPSFPIESJYGULMAQYXJJNJSHQVVAVUJHVWQYUVBFRBSOARUGVIYKINDGTV


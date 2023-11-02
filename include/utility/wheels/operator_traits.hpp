#ifndef _OPERATOR_TRAITS_HPP_INCLUDED_OJISDFLJASOIJ498UAFDKLJ489UALFKJALDFKJF
#define _OPERATOR_TRAITS_HPP_INCLUDED_OJISDFLJASOIJ498UAFDKLJ489UALFKJALDFKJF

#include "../include.hpp"

namespace nnl
{

template< typename, typename = void, typename = void >
struct has_assignment : std::false_type {};

template< typename T, typename U >
struct has_assignment< T, U, std::void_t<decltype( std::declval<T&>() = std::declval<U const&>() )>> : std::true_type {};

template< typename T >
struct has_assignment< T, void, std::void_t<decltype( std::declval<T&>() = std::declval<T const&>() )>> : std::true_type {};

template< class T >
inline constexpr bool has_assignment_v = has_assignment<T>::value;

template< class T, class U >
inline constexpr bool has_assignment_v2 = has_assignment<T, U>::value;

template< typename, typename = void, typename = void >
struct has_move_assignment : std::false_type {};

template< typename T, typename U >
struct has_move_assignment< T, U, std::void_t<decltype( std::declval<T&>() = std::declval<U&&>() )>> : std::true_type {};

template< typename T >
struct has_move_assignment< T, void, std::void_t<decltype( std::declval<T&>() = std::declval<T&&>() )>> : std::true_type {};

template< class T >
inline constexpr bool has_move_assignment_v = has_move_assignment<T>::value;

template< class T, class U >
inline constexpr bool has_move_assignment_v2 = has_move_assignment<T, U>::value;

template< typename, typename = void >
struct has_addition : std::false_type {};

template< typename T >
struct has_addition < T, std::void_t < decltype( std::declval<T const&>() + std::declval<T const&>() ) > > : std::true_type {};

template< class T >
inline constexpr bool has_addition_v = has_addition<T>::value;

template< typename, typename = void >
struct has_subtraction : std::false_type {};

template< typename T >
struct has_subtraction < T, std::void_t < decltype( std::declval<T const&>() - std::declval<T const&>() ) > > : std::true_type {};

template< class T >
inline constexpr bool has_subtraction_v = has_subtraction<T>::value;

template< typename, typename = void >
struct has_unary_plus : std::false_type {};

template< typename T >
struct has_unary_plus < T, std::void_t < decltype( + std::declval<T const&>() ) > > : std::true_type {};

template< class T >
inline constexpr bool has_unary_plus_v = has_unary_plus<T>::value;

template< typename, typename = void >
struct has_unary_minus : std::false_type {};

template< typename T >
struct has_unary_minus < T, std::void_t < decltype( - std::declval<T const&>() ) > > : std::true_type {};

template< class T >
inline constexpr bool has_unary_minus_v = has_unary_minus<T>::value;

template< typename, typename = void >
struct has_multiplication : std::false_type {};

template< typename T >
struct has_multiplication< T, std::void_t<decltype( std::declval<T const&>() * std::declval<T const&>() )>> : std::true_type {};

template< class T >
inline constexpr bool has_multiplication_v = has_multiplication<T>::value;

template< typename, typename = void >
struct has_division : std::false_type {};

template< typename T >
struct has_division < T, std::void_t < decltype( std::declval<T const&>() / std::declval<T const&>() ) > > : std::true_type {};

template< class T >
inline constexpr bool has_division_v = has_division<T>::value;

template< typename, typename = void >
struct has_modulo : std::false_type {};

template< typename T >
struct has_modulo < T, std::void_t < decltype( std::declval<T const&>() % std::declval<T const&>() ) > > : std::true_type {};

template< class T >
inline constexpr bool has_modulo_v = has_modulo<T>::value;

template< typename, typename = void >
struct has_prefix_increment : std::false_type {};

template< typename T >
struct has_prefix_increment < T, std::void_t < decltype( ++std::declval<T&>() ) > > : std::true_type {};

template< class T >
inline constexpr bool has_prefix_increment_v = has_prefix_increment<T>::value;

template< typename, typename = void >
struct has_postfix_increment : std::false_type {};

template< typename T >
struct has_postfix_increment < T, std::void_t < decltype( std::declval<T&>()++ ) > > : std::true_type {};

template< class T >
inline constexpr bool has_postfix_increment_v = has_postfix_increment<T>::value;

template< typename, typename = void >
struct has_prefix_decrement : std::false_type {};

template< typename T >
struct has_prefix_decrement < T, std::void_t < decltype( --std::declval<T&>() ) > > : std::true_type {};

template< class T >
inline constexpr bool has_prefix_decrement_v = has_prefix_decrement<T>::value;

template< typename, typename = void >
struct has_postfix_decrement : std::false_type {};

template< typename T >
struct has_postfix_decrement < T, std::void_t < decltype( std::declval<T&>()-- ) > > : std::true_type {};

template< class T >
inline constexpr bool has_postfix_decrement_v = has_postfix_decrement<T>::value;

template< typename, typename = void >
struct has_equal_to : std::false_type {};

template< typename T >
struct has_equal_to< T, std::void_t<decltype( std::declval<T const&>() == std::declval<T const&>() )>> : std::true_type {};

template< class T >
inline constexpr bool has_equal_to_v = has_equal_to<T>::value;

template< typename, typename = void >
struct has_not_equal_to : std::false_type {};

template< typename T >
struct has_not_equal_to < T, std::void_t < decltype( std::declval<T const&>() != std::declval<T const&>() ) > > : std::true_type {};

template< class T >
inline constexpr bool has_not_equal_to_v = has_not_equal_to<T>::value;

template< typename, typename = void >
struct has_greater_than : std::false_type {};

template< typename T >
struct has_greater_than< T, std::void_t<decltype( std::declval<T const&>() > std::declval<T const&>() )> > : std::true_type {};

template< class T >
inline constexpr bool has_greater_than_v = has_greater_than<T>::value;

template< typename, typename = void >
struct has_less_than : std::false_type {};

template< typename T >
struct has_less_than < T, std::void_t<decltype( std::declval<T const&>() < std::declval<T const&>() )>> : std::true_type {};

template< class T >
inline constexpr bool has_less_than_v = has_less_than<T>::value;

template< typename, typename = void >
struct has_greater_than_or_equal_to : std::false_type {};

template< typename T >
struct has_greater_than_or_equal_to< T, std::void_t<decltype( std::declval<T const&>() >= std::declval<T const&>() )> > : std::true_type {};

template< class T >
inline constexpr bool has_greater_than_or_equal_to_v = has_greater_than_or_equal_to<T>::value;

template< typename, typename = void >
struct has_less_than_or_equal_to : std::false_type {};

template< typename T >
struct has_less_than_or_equal_to < T, std::void_t<decltype( std::declval<T const&>() <= std::declval<T const&>() )>> : std::true_type {};

template< class T >
inline constexpr bool has_less_than_or_equal_to_v = has_less_than_or_equal_to<T>::value;

template< typename, typename = void >
struct has_logical_not : std::false_type {};

template< typename T >
struct has_logical_not < T, std::void_t < decltype( !std::declval<T const&>() ) > > : std::true_type {};

template< class T >
inline constexpr bool has_logical_not_v = has_logical_not<T>::value;

template< typename, typename = void >
struct has_logical_and : std::false_type {};

template< typename T >
struct has_logical_and < T, std::void_t < decltype(  std::declval<T const&>()&& std::declval<T const&>() ) > > : std::true_type {};

template< class T >
inline constexpr bool has_logical_and_v = has_logical_and<T>::value;

template< typename, typename = void >
struct has_logical_or : std::false_type {};

template< typename T >
struct has_logical_or < T, std::void_t < decltype(  std::declval<T const&>() || std::declval<T const&>() ) > > : std::true_type {};

template< class T >
inline constexpr bool has_logical_or_v = has_logical_or<T>::value;

template< typename, typename = void >
struct has_bitwise_not : std::false_type {};

template< typename T >
struct has_bitwise_not < T, std::void_t < decltype(  ~ std::declval<T const&>() ) > > : std::true_type {};

template< class T >
inline constexpr bool has_bitwise_not_v = has_bitwise_not<T>::value;

template< typename, typename = void >
struct has_bitwise_and : std::false_type {};

template< typename T >
struct has_bitwise_and< T, std::void_t<decltype(  std::declval<T const&>() & std::declval<T const&>() )>> : std::true_type {};

template< class T >
inline constexpr bool has_bitwise_and_v = has_bitwise_and<T>::value;

template< typename, typename = void >
struct has_bitwise_or : std::false_type {};

template< typename T >
struct has_bitwise_or < T, std::void_t < decltype(  std::declval<T const&>() | std::declval<T const&>() ) > > : std::true_type {};

template< class T >
inline constexpr bool has_bitwise_or_v = has_bitwise_or<T>::value;

template< typename, typename = void >
struct has_bitwise_xor : std::false_type {};

template< typename T >
struct has_bitwise_xor< T, std::void_t<decltype(  std::declval<T const&>() ^ std::declval<T const&>() )>> : std::true_type {};

template< class T >
inline constexpr bool has_bitwise_xor_v = has_bitwise_xor<T>::value;

template< typename, typename = void >
struct has_bitwise_left_shift : std::false_type {};

template< typename T >
struct has_bitwise_left_shift < T, std::void_t < decltype(  std::declval<T const&>()  << 1 ) > > : std::true_type {};

template< class T >
inline constexpr bool has_bitwise_left_shift_v = has_bitwise_left_shift<T>::value;

template< typename, typename = void >
struct has_bitwise_right_shift : std::false_type {};

template< typename T >
struct has_bitwise_right_shift < T, std::void_t < decltype(  std::declval<T const&>()  >> 1 ) > > : std::true_type {};

template< class T >
inline constexpr bool has_bitwise_right_shift_v = has_bitwise_right_shift<T>::value;

template< typename, typename = void >
struct has_addition_assignment : std::false_type {};

template< typename T >
struct has_addition_assignment < T, std::void_t < decltype( std::declval<T&>() += std::declval<T const&>() ) > > : std::true_type {};

template< class T >
inline constexpr bool has_addition_assignment_v = has_addition_assignment<T>::value;

template< typename, typename = void >
struct has_subtraction_assignment : std::false_type {};

template< typename T >
struct has_subtraction_assignment < T, std::void_t < decltype( std::declval<T&>() -= std::declval<T const&>() ) > > : std::true_type {};

template< class T >
inline constexpr bool has_subtraction_assignment_v = has_subtraction_assignment<T>::value;

template< typename, typename = void >
struct has_multiplication_assignment : std::false_type {};

template< typename T >
struct has_multiplication_assignment< T, std::void_t<decltype( std::declval<T&>() *= std::declval<T const&>() )>> : std::true_type {};

template< class T >
inline constexpr bool has_multiplication_assignment_v = has_multiplication_assignment<T>::value;

template< typename, typename = void >
struct has_division_assignment : std::false_type {};

template< typename T >
struct has_division_assignment < T, std::void_t < decltype( std::declval<T&>() /= std::declval<T const&>() ) > > : std::true_type {};

template< class T >
inline constexpr bool has_division_assignment_v = has_division_assignment<T>::value;

template< typename, typename = void >
struct has_modulo_assignment : std::false_type {};

template< typename T >
struct has_modulo_assignment < T, std::void_t < decltype( std::declval<T&>() %= std::declval<T const&>() ) > > : std::true_type {};

template< class T >
inline constexpr bool has_modulo_assignment_v = has_modulo_assignment<T>::value;

template< typename, typename = void >
struct has_bitwise_and_assignment : std::false_type {};

template< typename T >
struct has_bitwise_and_assignment< T, std::void_t<decltype(  std::declval<T&>() &= std::declval<T const&>() )>> : std::true_type {};

template< class T >
inline constexpr bool has_bitwise_and_assignment_v = has_bitwise_and_assignment<T>::value;

template< typename, typename = void >
struct has_bitwise_or_assignment : std::false_type {};

template< typename T >
struct has_bitwise_or_assignment < T, std::void_t < decltype(  std::declval<T&>() |= std::declval<T const&>() ) > > : std::true_type {};

template< class T >
inline constexpr bool has_bitwise_or_assignment_v = has_bitwise_or_assignment<T>::value;

template< typename, typename = void >
struct has_bitwise_left_shift_assignment : std::false_type {};

template< typename T >
struct has_bitwise_left_shift_assignment < T, std::void_t < decltype(  std::declval<T&>()  <<= 1 ) > > : std::true_type {};

template< class T >
inline constexpr bool has_bitwise_left_shift_assignment_v = has_bitwise_left_shift_assignment<T>::value;

template< typename, typename = void >
struct has_bitwise_right_shift_assignment : std::false_type {};

template< typename T >
struct has_bitwise_right_shift_assignment < T, std::void_t < decltype(  std::declval<T&>()  >>= 1 ) > > : std::true_type {};

template< class T >
inline constexpr bool has_bitwise_right_shift_assignment_v = has_bitwise_right_shift_assignment<T>::value;

template< typename, typename = void >
struct has_ostream : std::false_type {};

template< typename T >
struct has_ostream < T, std::void_t < decltype(  std::declval<std::ostream&>()  << std::declval<T const&>() ) > > : std::true_type {};

template< class T >
inline constexpr bool has_ostream_v = has_ostream<T>::value;

template< typename, typename = void >
struct has_istream : std::false_type {};

template< typename T >
struct has_istream < T, std::void_t < decltype(  std::declval<std::istream&>()  >> std::declval<T&>() ) > > : std::true_type {};

template< class T >
inline constexpr bool has_istream_v = has_istream<T>::value;

template< typename, typename = void >
struct has_bracket : std::false_type {};

template< typename T >
struct has_bracket< T, std::void_t<decltype(  ( std::declval<T&>() )[0] )>> : std::true_type {};

template< class T >
inline constexpr bool has_bracket_v = has_bracket<T>::value;

template< typename, typename = void >
struct has_const_bracket : std::false_type {};

template< typename T >
struct has_const_bracket< T, std::void_t<decltype(  ( std::declval<T const&>() )[0] )>> : std::true_type {};

template< class T >
inline constexpr bool has_const_bracket_v = has_const_bracket<T>::value;

}//namespace nnl

#endif//_OPERATOR_TRAITS_HPP_INCLUDED_OJISDFLJASOIJ498UAFDKLJ489UALFKJALDFKJF


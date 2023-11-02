#include "../../include/direct_space/graph.hpp"
#include "../../include/direct_space/node.hpp"
#include "../../include/direct_space/computation_table.hpp"
#include "../../include/direct_space/allocator.hpp"
#include "../../include/direct_space/engine.hpp"
#include "../../include/direct_space/stream.hpp"
#include "../../include/utility/wheels/cached_allocator.hpp"
#include "../../include/direct_space/session.hpp"
#include "../../include/direct_space/device.hpp"
#include "../../include/direct_space/tensor.hpp"

TEST_CASE( "tensor-3", "[tensor-3]" )
{
    std::cout << "Test case 25 of tensor-3" << std::endl;

    using namespace nnl;

    std::vector<std::string> dtypes{ {"int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64", "float32", "float64", } };
    std::int64_t max_dim = 8;

    for ( auto const& dtype : dtypes )
    {
        for ( std::int64_t r : range( max_dim ) )
        {
            for ( std::int64_t c : range( max_dim ) )
            {
                auto t = random<cuda_engine>( {r+1, c+1,}, 0.0, 5.0, dtype );
                t.to_file( "./tensor_3_output.tmp.bin" );
                auto s = from_file<cuda_engine>( "./tensor_3_output.tmp.bin" );
                REQUIRE( t.shape() == s.shape() );
                REQUIRE( t.dtype() == s.dtype() );
                std::int8_t* t_ptr = reinterpret_cast<std::int8_t*>( t.data() );
                std::int8_t* s_ptr = reinterpret_cast<std::int8_t*>( s.data() );
                for ( auto idx : range( t.size() ) )
                    CHECK( t_ptr[idx] == s_ptr[idx] );
            }
        }
    }

}


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

TEST_CASE( "tensor", "[tensor]" )
{
    std::cout << "Test case 6 of tensor started." << std::endl;

    using namespace nnl;

    {
        auto t = make_tensor<cuda_engine>( {1, 2, 3, 4}, "int8", "int8_tensor" );
        std::cout << "Created tensor size in bytes: " << t.size() << std::endl;
        REQUIRE( t.size() == 1*2*3*4 );
    }
    {
        auto t = make_tensor<cuda_engine>( {1, 2}, "uint8" );
        std::cout << "Created tensor size in bytes: " << t.size() << std::endl;
        REQUIRE( t.size() == 1*2 );
    }
    {
        auto t = make_tensor<cuda_engine>( {1, 2, 3}, "float32" );
        std::cout << "Created tensor size in bytes: " << t.size() << std::endl;
        REQUIRE( t.size() == 1*2*3*sizeof(float) );
    }
    {
        auto t = make_tensor<cuda_engine>( {1, 2, 3}, "uint64" );
        std::cout << "Created tensor size in bytes: " << t.size() << std::endl;
        REQUIRE( t.size() == 1*2*3*sizeof(std::uint64_t) );
    }

    // shallow copy test
    {
        auto t = make_tensor<cuda_engine>( {1, 2, 3, 4, 5, 6, 7, 8}, "float32" );
        std::cout << "Created tensor size in bytes: " << t.size() << std::endl;
        float* data = reinterpret_cast<float*>(t.data());
        unsigned long const n = t.size() / sizeof(float);
        for ( unsigned long i = 0; i < n; ++i )
            data[i] = 1.0f * i;
        auto s = t;
        float* sata = reinterpret_cast<float*>(t.data());

        for ( unsigned long i = 0; i < n; ++i )
            REQUIRE( sata[i] == data[i] );
    }

    {
        auto& sess = get_default_session<cuda_engine>();

        // create tensor
        auto t = make_tensor<cuda_engine>( {1, 2, 3, 4, 5, 6, 7, 8,}, "float32" );
        //auto t = make_tensor<cuda_engine>( {1, 2, 3,}, "float32" );
        {
            float* data = reinterpret_cast<float*>(t.data());
            unsigned long const n = t.size() / sizeof(float);
            for ( unsigned long i = 0; i < n; ++i )
                data[i] = 1.0f * i;
        }
        // copy to device
        [[maybe_unused]] auto const& dm = sess.tensor_to_device( t );
        //sess.device_memory_manager_.default_stream().synchronize();
        sess.synchronize();
        // update host
        {
            float* data = reinterpret_cast<float*>(t.data());
            unsigned long const n = t.size() / sizeof(float);
            for ( unsigned long i = 0; i < n; ++i )
                data[i] = 1.0f * i + 1.0f;
        }

        // copy device back to host
        sess.tensor_to_host( t );
        //sess.device_memory_manager_.default_stream().synchronize();
        sess.synchronize();

        // data should remain as before
        {
            float* data = reinterpret_cast<float*>(t.data());
            unsigned long const n = t.size() / sizeof(float);
            for ( unsigned long i = 0; i < n; ++i )
            {
                if ( !(std::abs(data[i]  - 1.0f * i) < 1.0e-3) )
                {
                    spdlog::error( "data[{}]={}, but {} is expected.", i, data[i], i );
                }
                REQUIRE( (std::abs(data[i]  - 1.0f * i) < 1.0e-3) );
            }
        }

        std::cout << "The current session is:\n" << sess << std::endl;

        sess.tensor_dismiss_device( t );

        std::cout << "After dismissing the tensor, the session is:\n" << sess << std::endl;

        REQUIRE( true );
    }

    {
        auto t = make_tensor<cuda_engine>( {1, 2, 3, 4}, "int8", "int8_tensor_se" );
        std::cout << "Created tensor size in bytes: " << t.size() << std::endl;
        auto& sess = get_default_session<cuda_engine>();
        std::cout << "Before reshape, the session is\n" << sess << "\n";
        t.reshape( {2, 3, 4, 5} );
        REQUIRE( t.size() == 2*3*4*5 );
        std::cout << "After reshape, the session is\n" << sess << "\n";
    }

    {
        auto t1 = make_tensor<cuda_engine>( {1, 2}, "int16", "tsor16-1" );
        std::cout << "tensor 16/1 is " << t1 << "\n";
        auto t2 = make_tensor<cuda_engine>( "int16", "tsor16-2" );
        std::cout << "tensor 16/2 is " << t2 << "\n";
        REQUIRE( true );
    }



    {
        auto t1 = random<cuda_engine>( {8, 8}, -1.0f, 1.0f, "float32", "float32-random" );
        std::cout << "random tensor is\n" << t1 << "\n";
        float* dat = reinterpret_cast<float*>(t1.data());
        for ( auto r : range(8) )
        {
            for ( auto c : range(8) )
                std::cout << dat[r*8+c] << " ";
            std::cout << "\n";
        }
        REQUIRE( true );
    }


}


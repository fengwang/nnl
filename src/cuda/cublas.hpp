#ifndef USVNXVMUHPIJYXEXXVNRVEJNNSQWWNVELGFISEKABBNHTVUMUOSHTLDQGPIPCIKGOGUQEHDLB
#define USVNXVMUHPIJYXEXXVNRVEJNNSQWWNVELGFISEKABBNHTVUMUOSHTLDQGPIPCIKGOGUQEHDLB

#include "./cublas_assert.hpp"
#include "../../include/utility/utility.hpp"
#include "../../include/direct_space/stream.hpp"

namespace nnl
{

    struct cublas_handle
    {
        cublasHandle_t handle_;

        cublas_handle()
        {
            cublas_assert( cublasCreate(&handle_) );
        }

        ~cublas_handle()
        {
            cublas_assert( cublasDestroy(handle_) );
        }

        cublasHandle_t handle()
        {
            return handle_;
        }

        void set_stream( stream<cuda_engine>& sm ); // ./implemented in memory_management.cpp

    };//struct cublas_handle

    inline cublas_handle& get_default_cublas_handle()
    {
        return singleton<cublas_handle>::instance();
    }

    // C <= A * B
    // where A or A' is [m x n], B or B' is [n x k] and C is [m x k]
    template< typename T > requires std::floating_point<T>
    void cuda_gemm( T* A, bool a_transposed, T* B, bool b_transposed,
                    std::size_t m, std::size_t n, std::size_t k, T* C, T alpha, T beta, stream<cuda_engine>& sm, int batch_count=1 )
    {
        if ( batch_count < 1 )
            better_assert( false, format("gemm only supports positive batch_count, but got {}", batch_count) );

        cublas_handle& _hd = get_default_cublas_handle();
        _hd.set_stream( sm );
        cublasHandle_t hd = _hd.handle();

        T* a = A;
        T* b = B;
        T* c = C;

        T* result_ptr = c;
        T* first_ptr = b;
        T* second_ptr = a;
        int row_of_c = m;
        int col_of_c = k;
        int common_dimension = n;

        int ld_of_first_ptr = b_transposed ? n : k;
        int ld_of_second_ptr = a_transposed ? m : n;
        int ld_of_result_ptr = k;

        cublasOperation_t first_transposed = b_transposed ? CUBLAS_OP_T : CUBLAS_OP_N;
        cublasOperation_t second_transposed = a_transposed ? CUBLAS_OP_T : CUBLAS_OP_N;

        if constexpr( std::is_same_v<T, float> )
        {
            if ( 1 == batch_count )
                cublas_assert( cublasSgemm_v2( hd, first_transposed, second_transposed, col_of_c, row_of_c, common_dimension, &alpha, first_ptr, ld_of_first_ptr, second_ptr, ld_of_second_ptr, &beta, result_ptr, ld_of_result_ptr ) );
            else
                cublas_assert( cublasSgemmBatched( hd, first_transposed, second_transposed, col_of_c, row_of_c, common_dimension, &alpha, &first_ptr, ld_of_first_ptr, &second_ptr, ld_of_second_ptr, &beta, &result_ptr, ld_of_result_ptr, batch_count ) );
        }
        else if constexpr( std::is_same_v<T, double> )
        {
            if ( 1 == batch_count )
                cublas_assert( cublasDgemm_v2( hd, first_transposed, second_transposed, col_of_c, row_of_c, common_dimension, &alpha, first_ptr, ld_of_first_ptr, second_ptr, ld_of_second_ptr, &beta, result_ptr, ld_of_result_ptr ) );
            else
                cublas_assert( cublasDgemmBatched( hd, first_transposed, second_transposed, col_of_c, row_of_c, common_dimension, &alpha, &first_ptr, ld_of_first_ptr, &second_ptr, ld_of_second_ptr, &beta, &result_ptr, ld_of_result_ptr, batch_count ) );
        }
        else
        {
            better_assert( false, "gemm only supports float and double!" );
        }

        // ... explicitly sync
        //sm.synchronize();
    }


}//namespace nnl

#endif//USVNXVMUHPIJYXEXXVNRVEJNNSQWWNVELGFISEKABBNHTVUMUOSHTLDQGPIPCIKGOGUQEHDLB

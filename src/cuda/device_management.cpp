#include "./cuda_assert.hpp"

extern "C"
void cuda_device_synchronize()
{
    cuda_assert( cudaDeviceSynchronize() );
}



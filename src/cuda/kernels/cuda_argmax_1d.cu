#include <cub/cub.cuh>
#include <cub/device/device_reduce.cuh>

extern "C"
void cuda_argmax_1d( float* input, int n_input, float* cache, int n_cache, int* result, cudaStream_t sm )
{
    std::size_t cache_size = static_cast<std::size_t>(n_cache * sizeof(float));
    cub::KeyValuePair<int, float> *_result = reinterpret_cast<cub::KeyValuePair<int, float>*>(result);
    //cub::DeviceReduce::ArgMax( cache, cache_size, input,  _result, n_input, sm, false );
    cub::DeviceReduce::ArgMax( cache, cache_size, input,  _result, n_input, sm );
}


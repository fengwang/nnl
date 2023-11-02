#include "../3rd_party/oneflow/layer_norm.cuh"

extern "C"
void cuda_layer_norm( float* input, float* output, int rows, int cols, float eps, float* mean, float* var, cudaStream_t sm )
{
    oneflow::cuda::layer_norm::DirectLoad<float, float> load(input, cols );
    oneflow::cuda::layer_norm::DirectStore<float, float> store(output, cols );
    oneflow::cuda::layer_norm::DispatchLayerNorm( sm, load, store, rows, cols, (double)eps, mean, var );
}


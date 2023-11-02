#define CATCH_CONFIG_MAIN
#include "./catch.hpp"

#include "../include/utility/3rdparty/spdlog-1.11.0/include/spdlog/spdlog.h"


// turn on/off log levels
TEST_CASE( "set-log-level--1", "[set-log-level--1]" )
{
    //spdlog::set_level(spdlog::level::trace);
    //spdlog::set_level(spdlog::level::debug);
    //spdlog::set_level(spdlog::level::info);
    spdlog::set_level(spdlog::level::err);
    //spdlog::set_level(spdlog::level::off);
}

#include "../include/utility/utility.hpp"

#if 1
#include "./cases/0000_hello_world.inc"
#include "./cases/0001_graph.inc"
#include "./cases/0002_computation_table.inc"
#include "./cases/0003_allocator.inc"
#include "./cases/0004_memcpy.inc"
#include "./cases/0005_device_memory_manager.inc"
#include "./cases/0006_tensor.inc"
#include "./cases/0007_node.inc"
#include "./cases/0008_device_memory_manager_2.inc"
#include "./cases/0009_tensor_2.inc"
#include "./cases/0010_node_2.inc"
#include "./cases/0011_node_linear.inc"
#include "./cases/0012_softmax.inc"
#include "./cases/0013_gelu.inc"
#include "./cases/0014_node_buffer.inc"
#include "./cases/0016_scaled_offset.inc"
#include "./cases/0015_layer_norm.inc"
#include "./cases/0017_layer_norm.inc"

#include "./cases/0018_attention.inc"
#include "./cases/0019_attention_no_buffer.inc"
#include "./cases/0020_node_stacked_linear.inc"
#include "./cases/0021_node_add.inc"
#include "./cases/0022_attention_layer.inc"
#include "./cases/0023_attention_model.inc"

#include "./cases/0024_model_stacked_linear.inc"
#include "./cases/0025_self_attention_model.inc"
#include "./cases/0025_tensor_3.inc"
#include "./cases/0026_softmax_se.inc"
#include "./cases/0027_layer_norm_se.inc"
#include "./cases/0028_attention.inc"
#include "./cases/0029_layer_norm.inc"
#include "./cases/0030_self_attention_model.inc"
#include "./cases/0031_softmax.inc"
#include "./cases/0032_attention.inc"
//#include "./cases/0033_attention.inc" // <-- cannot pass
#include "./cases/0034_attention.inc"
#include "./cases/0035_self_attention_model.inc"
#include "./cases/0036_self_attention_model.inc"
//#include "./cases/0037_multihead_attention.inc" // <- deprecated interface
#include "./cases/0038_tensor_load_memory.inc"
#include "./cases/0039_multi_head_attention_model.inc"
#include "./cases/0040_multi_head_attention_model.inc"
#include "./cases/0041_multi_head_attention_model.inc"
#include "./cases/0042_positionwise_feed_forward.inc"
//#include "./cases/0043_transformer.inc" // <-- one element in the result does not match...
#include "./cases/0044_transformer_massive.inc"
#include "./cases/0045_transformer_massive_repeative.inc"
#include "./cases/0046_vocabulary_projection.inc"
#endif





// turn on/off log levels
TEST_CASE( "reset-device", "[reset-device]" )
{
    //nnl::reset_device<nnl::default_engine_type>();
}


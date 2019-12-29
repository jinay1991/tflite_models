///
/// @file
/// @copyright Copyright (c) 2020. All Rights Reserved.
///
#ifndef PERCEPTION_ARGUMENT_PARSER_CLI_OPTIONS_H_
#define PERCEPTION_ARGUMENT_PARSER_CLI_OPTIONS_H_

#include <cstdint>
#include <string>

namespace perception
{
struct CLIOptions
{
    bool input_floating = false;
    bool profiling = false;
    bool verbose = false;
    float input_mean = 127.5f;
    float input_std = 127.5f;
    std::int32_t loop_count = 1;
    std::int32_t max_profiling_buffer_entries = 1024;
    std::int32_t number_of_results = 5;
    std::int32_t number_of_threads = 4;
    std::string input_layer_type = "uint8_t";
    std::string input_name = "data/grace_hopper.jpg";
    std::string labels_name = "data/labels.txt";
    std::string model_name = "external/mobilenet_v2_1.0_224_quant/mobilenet_v2_1.0_224_quant.tflite";
    std::string result_directory = "intermediate_tensors";
};

}  // namespace perception
#endif  /// PERCEPTION_ARGUMENT_PARSER_CLI_OPTIONS_H_

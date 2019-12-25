///
/// @file
/// @copyright Copyright (c) 2019. All Rights Reserved.
///
#ifndef PERCEPTION_CLI_H_
#define PERCEPTION_CLI_H_

#include <cstdint>
#include <string>

#include "tensorflow/lite/model.h"

namespace perception
{
struct CLIOptions
{
    bool verbose = false;
    bool input_floating = false;
    bool profiling = false;
    float input_mean = 127.5f;
    float input_std = 127.5f;
    std::int32_t loop_count = 1;
    std::int32_t max_profiling_buffer_entries = 1024;
    std::int32_t number_of_threads = 4;
    std::int32_t number_of_results = 5;
    std::string input_bmp_name = "data/grace_hopper.jpg";
    std::string input_layer_type = "uint8_t";
    std::string labels_file_name = "data/labels.txt";
    std::string model_name = "external/mobilenet_v2_1.0_224_quant/mobilenet_v2_1.0_224_quant.tflite";
    std::string save_results_directory = "intermediate_layers";
    tflite::FlatBufferModel* model;
};

CLIOptions ParseCommandLineOptions(int argc, char** argv);

}  // namespace perception

#endif  /// PERCEPTION_CLI_H_
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
    bool accel = false;
    bool old_accel = false;
    bool input_floating = false;
    bool profiling = false;
    bool allow_fp16 = false;
    bool gl_backend = false;
    std::int32_t loop_count = 1;
    float input_mean = 127.5f;
    float input_std = 127.5f;
    std::string model_name = "external/mobilenet_v2_1.0_224_quant/mobilenet_v2_1.0_224_quant.tflite";
    tflite::FlatBufferModel* model;
    std::string input_bmp_name = "./data/grace_hopper.bmp";
    std::string labels_file_name = "./data/labels.txt";
    std::string input_layer_type = "uint8_t";
    std::int32_t number_of_threads = 4;
    std::int32_t number_of_results = 5;
    std::int32_t max_profiling_buffer_entries = 1024;
    std::int32_t number_of_warmup_runs = 2;
};

CLIOptions ParseCommandLineOptions(int argc, char** argv);

}  // namespace perception

#endif  /// PERCEPTION_CLI_H_
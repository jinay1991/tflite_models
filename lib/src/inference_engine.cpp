///
/// @file
/// @copyright Copyright (c) 2019. All Rights Reserved.
///
#include <sys/time.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

#include <experimental/filesystem>

#include "absl/memory/memory.h"
#include "tensorflow/lite/core/api/profiler.h"
#include "tensorflow/lite/delegates/nnapi/nnapi_delegate.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/optional_debug_tools.h"
#include "tensorflow/lite/profiling/profiler.h"
#include "tensorflow/lite/string_util.h"
#include "tensorflow/lite/tools/evaluation/utils.h"

#include "perception/inference_engine.h"
#include "perception/utils/bitmap_helper.h"
#include "perception/utils/resize.h"

#define LOG(x) std::cerr

namespace perception
{
namespace
{
constexpr double get_us(struct timeval t) { return (t.tv_sec * 1000000 + t.tv_usec); }

void PrintProfilingInfo(const tflite::profiling::ProfileEvent* e, std::uint32_t op_index,
                        TfLiteRegistration registration)
{
    // output something like
    // time (ms) , Node xxx, OpCode xxx, symblic name
    //      5.352, Node   5, OpCode   4, DEPTHWISE_CONV_2D

    LOG(INFO) << std::fixed << std::setw(10) << std::setprecision(3)
              << (e->end_timestamp_us - e->begin_timestamp_us) / 1000.0 << ", Node " << std::setw(3)
              << std::setprecision(3) << op_index << ", OpCode " << std::setw(3) << std::setprecision(3)
              << registration.builtin_code << ", "
              << tflite::EnumNameBuiltinOperator(static_cast<tflite::BuiltinOperator>(registration.builtin_code))
              << "\n";
}
inline std::ostream& operator<<(std::ostream& os, const TfLiteIntArray* v)
{
    if (!v)
    {
        os << " (null)";
        return os;
    }
    for (int k = 0; k < v->size; k++)
    {
        os << " " << std::dec << v->data[k];
    }
    return os;
}

}  // namespace

InferenceEngine::InferenceEngine(const CLIOptions& cli_opts)
    : cli_opts_{cli_opts},
      model_path_{cli_opts.model_name},
      label_path_{cli_opts.labels_file_name},
      verbose_{cli_opts.verbose},
      image_width_{224},
      image_height_{224},
      image_channels_{3},
      image_helper_{std::make_unique<BitmapImageHelper>()}
{
    if (!std::experimental::filesystem::exists(model_path_))
    {
        throw std::runtime_error("Failed to locate " + model_path_);
    }

    model_ = tflite::FlatBufferModel::BuildFromFile(model_path_.c_str());
    if (!model_)
    {
        throw std::runtime_error("Failed to mmap model " + model_path_);
    }
    LOG(INFO) << "Loaded model " << model_path_ << std::endl;
    model_->error_reporter();
    LOG(INFO) << "Resolved reporter" << std::endl;

    tflite::ops::builtin::BuiltinOpResolver resolver;

    tflite::InterpreterBuilder(*model_, resolver)(&interpreter_);
    if (!interpreter_)
    {
        throw std::runtime_error("Failed to construct interpreter");
    }
    if (verbose_)
    {
        LOG(INFO) << "tensors size: " << interpreter_->tensors_size() << "\n";
        LOG(INFO) << "nodes size: " << interpreter_->nodes_size() << "\n";
        LOG(INFO) << "inputs: " << interpreter_->inputs().size() << "\n";
        LOG(INFO) << "input(0) name: " << interpreter_->GetInputName(0) << "\n";

        int t_size = interpreter_->tensors_size();
        for (int i = 0; i < t_size; i++)
        {
            if (interpreter_->tensor(i)->name)
                LOG(INFO) << i << ": " << interpreter_->tensor(i)->name << ", " << interpreter_->tensor(i)->bytes
                          << ", " << interpreter_->tensor(i)->type << ", " << interpreter_->tensor(i)->params.scale
                          << ", " << interpreter_->tensor(i)->params.zero_point << "\n";
        }
    }

    if (cli_opts_.number_of_threads != -1)
    {
        interpreter_->SetNumThreads(cli_opts_.number_of_threads);
    }
}

void InferenceEngine::Init()
{
    const std::vector<std::int32_t> inputs = interpreter_->inputs();
    const std::vector<std::int32_t> outputs = interpreter_->outputs();

    if (verbose_)
    {
        LOG(INFO) << "number of inputs: " << inputs.size() << "\n";
        LOG(INFO) << "number of outputs: " << outputs.size() << "\n";
    }

    if (interpreter_->AllocateTensors() != TfLiteStatus::kTfLiteOk)
    {
        LOG(FATAL) << "Failed to allocate tensors!";
    }

    if (verbose_)
    {
        PrintInterpreterState(interpreter_.get());
    }
}

void InferenceEngine::ExecuteStep(const std::string& image_path)
{
    auto in = image_helper_->ReadImage(image_path, &image_width_, &image_height_, &image_channels_);

    SetInputData(in);

    auto profiler = absl::make_unique<tflite::profiling::Profiler>(cli_opts_.max_profiling_buffer_entries);
    interpreter_->SetProfiler(profiler.get());

    if (cli_opts_.profiling)
    {
        profiler->StartProfiling();
    }

    struct timeval start_time;
    struct timeval stop_time;
    gettimeofday(&start_time, nullptr);

    if (interpreter_->Invoke() != kTfLiteOk)
    {
        LOG(FATAL) << "Failed to invoke tflite!\n";
    }

    gettimeofday(&stop_time, nullptr);
    LOG(INFO) << "invoked \n";
    LOG(INFO) << "average time: " << (get_us(stop_time) - get_us(start_time)) / (cli_opts_.loop_count * 1000)
              << " ms \n";

    if (cli_opts_.profiling)
    {
        profiler->StopProfiling();
        auto profile_events = profiler->GetProfileEvents();
        for (std::int32_t i = 0; i < profile_events.size(); i++)
        {
            auto op_index = profile_events[i]->event_metadata;
            const auto node_and_registration = interpreter_->node_and_registration(op_index);
            const TfLiteRegistration registration = node_and_registration->second;
            PrintProfilingInfo(profile_events[i], op_index, registration);
        }
    }

    SaveIntermediateResults();

    const auto results = GetResults();

    PrintResults(results);
}

void InferenceEngine::Shutdown() {}

void InferenceEngine::SetInputData(std::vector<std::uint8_t> in)
{
    const auto input = interpreter_->inputs()[0];

    // get input dimension from the input tensor metadata
    // assuming one input only
    TfLiteIntArray* dims = interpreter_->tensor(input)->dims;
    std::int32_t wanted_height = dims->data[1];
    std::int32_t wanted_width = dims->data[2];
    std::int32_t wanted_channels = dims->data[3];

    switch (interpreter_->tensor(input)->type)
    {
        case kTfLiteFloat32:
            cli_opts_.input_floating = true;
            ResizeImage<float>(interpreter_->typed_tensor<float>(input), in.data(), image_height_, image_width_,
                               image_channels_, wanted_height, wanted_width, wanted_channels, cli_opts_.input_floating,
                               cli_opts_.input_mean, cli_opts_.input_std);
            break;
        case kTfLiteUInt8:
            ResizeImage<std::uint8_t>(interpreter_->typed_tensor<std::uint8_t>(input), in.data(), image_height_,
                                      image_width_, image_channels_, wanted_height, wanted_width, wanted_channels,
                                      cli_opts_.input_floating, cli_opts_.input_mean, cli_opts_.input_std);
            break;
        default:
            throw std::runtime_error("cannot handle input type " + std::to_string(interpreter_->tensor(input)->type) +
                                     " yet");
    }
}
std::vector<std::pair<float, std::int32_t>> InferenceEngine::GetResults() const
{
    std::vector<std::pair<float, std::int32_t>> top_results;
    const float threshold = 0.001f;

    const auto output = interpreter_->outputs()[0];
    const auto output_dims = interpreter_->tensor(output)->dims;
    // assume output dims to be something like (1, 1, ... ,size)
    const auto output_size = output_dims->data[output_dims->size - 1];
    switch (interpreter_->tensor(output)->type)
    {
        case TfLiteType::kTfLiteFloat32:
            get_top_n<float>(interpreter_->typed_output_tensor<float>(0), output_size, cli_opts_.number_of_results,
                             threshold, &top_results, true);
            break;
        case TfLiteType::kTfLiteUInt8:
            get_top_n<std::uint8_t>(interpreter_->typed_output_tensor<std::uint8_t>(0), output_size,
                                    cli_opts_.number_of_results, threshold, &top_results, false);
            break;
        default:
            throw std::runtime_error("cannot handle output type " + std::to_string(interpreter_->tensor(output)->type) +
                                     " yet");
    }
    return top_results;
}

void InferenceEngine::PrintResults(std::vector<std::pair<float, std::int32_t>> top_results)
{
    std::vector<std::string> labels;
    size_t label_count;

    ReadLabelsFile(cli_opts_.labels_file_name, &labels, &label_count);

    for (const auto& result : top_results)
    {
        const float confidence = result.first;
        const std::int32_t index = result.second;
        LOG(INFO) << confidence << ": " << labels[index] << "\n";
    }
}
// Takes a file name, and loads a list of labels from it, one per line, and
// returns a vector of the strings. It pads with empty strings so the length
// of the result is a multiple of 16, because our model expects that.
void InferenceEngine::ReadLabelsFile(const std::string& file_name, std::vector<std::string>* result,
                                     std::size_t* found_label_count)
{
    std::ifstream file(file_name);
    if (!file)
    {
        throw std::runtime_error("Labels file " + file_name + " not found\n");
    }
    result->clear();
    std::string line;
    while (std::getline(file, line))
    {
        result->push_back(line);
    }
    *found_label_count = result->size();
    const std::int32_t padding = 16;
    while (result->size() % padding)
    {
        result->emplace_back();
    }
}

void InferenceEngine::SaveIntermediateResults()
{
    const auto dirname = "intermediate_layers";
    if (!std::experimental::filesystem::exists(dirname))
    {
        if (!std::experimental::filesystem::create_directory(dirname))
        {
            throw std::runtime_error("Unable to create directory\n");
        }
    }

    for (auto idx = 0U; idx < interpreter_->tensors_size() - 1; idx++)
    {
        auto tensor = interpreter_->tensor(idx);
        if (tensor->type != TfLiteType::kTfLiteUInt8 && tensor->allocation_type == TfLiteAllocationType::kTfLiteMmapRo)
        {
            continue;
        }
        auto tensor_name = std::string{tensor->name};
        std::replace(tensor_name.begin(), tensor_name.end(), '/', '_');

        std::stringstream filename;
        filename << dirname << "/" << std::setw(3) << std::setfill('0') << idx << "_" << tensor_name << "_tensor.txt";

        auto tensor_dims = tensor->dims;
        auto tensor_channels = tensor_dims->data[tensor_dims->size - 1];

        std::ofstream f(filename.str(), std::ios::binary);
        for (auto b = 0U, ch = 0U; b < tensor->bytes; ++b, ++ch)
        {
            if (ch > tensor_channels - 1)
            {
                ch = 0;
            }
            f << +ch << " " << +static_cast<std::uint8_t>(tensor->data.raw_const[b]) << "\n";
        }
    }
}

}  // namespace perception

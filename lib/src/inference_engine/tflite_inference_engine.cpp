///
/// @file
/// @copyright Copyright (c) 2020. All Rights Reserved.
///
#include <sys/time.h>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

#include <experimental/filesystem>

#include "absl/memory/memory.h"
#include "absl/strings/match.h"
#include "tensorflow/lite/core/api/profiler.h"
#include "tensorflow/lite/delegates/nnapi/nnapi_delegate.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/optional_debug_tools.h"
#include "tensorflow/lite/string_util.h"
#include "tensorflow/lite/tools/evaluation/utils.h"
#define TFLITE_PROFILING_ENABLED
#include "tensorflow/lite/profiling/profiler.h"

#include "perception/inference_engine/tflite_inference_engine.h"
#include "perception/utils/get_top_n.h"
#include "perception/utils/resize.h"

#define LOG(x) std::cerr

namespace perception
{
namespace
{
constexpr double get_us(struct timeval t) { return (t.tv_sec * 1000000 + t.tv_usec); }

void WriteToFile(const std::string& dirname, const std::string& filename, const std::string& content)
{
    const auto filepath = std::string{dirname + "/" + filename};
    std::ofstream file(filepath, std::ios::binary);
    if (!file.is_open())
    {
        throw std::runtime_error("Unable to open " + filepath);
    }
    file << content;
}

void PrintOutput(const std::vector<std::pair<float, std::int32_t>>& top_results, const std::vector<std::string>& labels)
{
    for (const auto& result : top_results)
    {
        const float confidence = result.first;
        const std::int32_t index = result.second;
        LOG(INFO) << confidence << ": " << labels[index] << "\n";
    }
}

void PrintProfilingInfo(const tflite::profiling::ProfileEvent* e, const std::uint32_t op_index,
                        const TfLiteTensor* tensor, const TfLiteRegistration& registration)
{
    // output something like
    //  time (ms), Node xxx, OpCode xxx,     symbolic name,    dimension,   type
    //      5.352, Node   5, OpCode   4, DEPTHWISE_CONV_2D, [1 1 1 1280],  uint8
    static bool print_header = true;
    if (print_header)
    {
        LOG(INFO) << " time (ms), Node xxx, OpCode xxx,        symbolic name,              dimension,   type\n";
        print_header = false;
    }
    LOG(INFO) << std::fixed << std::setw(10) << std::setprecision(3)
              << (e->end_timestamp_us - e->begin_timestamp_us) / 1000.0 << ", Node " << std::setw(3)
              << std::setprecision(3) << op_index << ", OpCode " << std::setw(3) << std::setprecision(3)
              << registration.builtin_code << ", " << std::setw(20)
              << tflite::EnumNameBuiltinOperator(static_cast<tflite::BuiltinOperator>(registration.builtin_code))
              << ", [" << tensor->dims << "], " << std::setw(6) << tensor->type << "\n";
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
        os << " " << std::dec << std::setw(4) << v->data[k];
    }
    return os;
}

inline std::ostream& operator<<(std::ostream& os, const TfLiteType& type)
{
    switch (type)
    {
        case kTfLiteFloat32:
            os << "float32";
            break;
        case kTfLiteInt32:
            os << "int32";
            break;
        case kTfLiteUInt8:
            os << "uint8";
            break;
        case kTfLiteInt8:
            os << "int8";
            break;
        case kTfLiteInt64:
            os << "int64";
            break;
        case kTfLiteString:
            os << "string";
            break;
        case kTfLiteBool:
            os << "bool";
            break;
        case kTfLiteInt16:
            os << "int16";
            break;
        case kTfLiteComplex64:
            os << "complex64";
            break;
        case kTfLiteFloat16:
            os << "float16";
            break;
        case kTfLiteNoType:
            os << "no type";
            break;
        default:
            os << "(invalid)";
            break;
    }
    return os;
}

}  // namespace

TFLiteInferenceEngine::TFLiteInferenceEngine(const CLIOptions& cli_options) : InferenceEngineBase{cli_options} {}

void TFLiteInferenceEngine::Init()
{
    model_ = tflite::FlatBufferModel::BuildFromFile(GetModelPath().c_str());
    if (!model_)
    {
        throw std::runtime_error("Failed to mmap model " + GetModelPath());
    }
    LOG(INFO) << "Loaded model " << GetModelPath() << std::endl;
    model_->error_reporter();
    LOG(INFO) << "Resolved reporter\n";

    tflite::ops::builtin::BuiltinOpResolver resolver;

    tflite::InterpreterBuilder(*model_, resolver)(&interpreter_);
    if (!interpreter_)
    {
        throw std::runtime_error("Failed to construct interpreter");
    }
    LOG(INFO) << "Built TfLite Interpreter\n";
    if (IsVerbosityEnabled())
    {
        LOG(INFO) << "tensors size: " << interpreter_->tensors_size() << "\n";
        LOG(INFO) << "nodes size: " << interpreter_->nodes_size() << "\n";
        LOG(INFO) << "inputs: " << interpreter_->inputs().size() << "\n";
        LOG(INFO) << "input(0) name: " << interpreter_->GetInputName(0) << "\n";

        int tensor_size = interpreter_->tensors_size();
        for (int i = 0; i < tensor_size; i++)
        {
            if (interpreter_->tensor(i)->name)
            {
                LOG(INFO) << i << ": " << interpreter_->tensor(i)->name << ", " << interpreter_->tensor(i)->bytes
                          << ", " << interpreter_->tensor(i)->type << ", " << interpreter_->tensor(i)->params.scale
                          << ", " << interpreter_->tensor(i)->params.zero_point << "\n";
            }
        }
    }

    if (-1 != GetNumberOfThreads())
    {
        interpreter_->SetNumThreads(GetNumberOfThreads());
    }

    if (IsVerbosityEnabled())
    {
        const auto inputs = interpreter_->inputs();
        const auto outputs = interpreter_->outputs();
        LOG(INFO) << "number of inputs: " << inputs.size() << "\n";
        LOG(INFO) << "number of outputs: " << outputs.size() << "\n";
    }

    if (interpreter_->AllocateTensors() != TfLiteStatus::kTfLiteOk)
    {
        LOG(FATAL) << "Failed to allocate tensors!";
    }

    if (IsVerbosityEnabled())
    {
        PrintInterpreterState(interpreter_.get());
    }
}

void TFLiteInferenceEngine::Execute()
{
    SetInputData(GetImageData());

    auto profiler = absl::make_unique<tflite::profiling::Profiler>(GetMaxProfilingBufferEntries());
    interpreter_->SetProfiler(profiler.get());

    if (IsProfilingEnabled())
    {
        profiler->StartProfiling();
    }

    InvokeInference();

    if (IsProfilingEnabled())
    {
        profiler->StopProfiling();
        auto profile_events = profiler->GetProfileEvents();
        std::vector<std::int32_t> op_indices;
        for (std::int32_t i = 0; i < profile_events.size(); i++)
        {
            auto op_index = profile_events[i]->event_metadata;
            const auto node_and_registration = interpreter_->node_and_registration(op_index);
            const auto node = node_and_registration->first;
            const auto registration = node_and_registration->second;
            const auto tensor = interpreter_->tensor(node.outputs[0].data[0]);
            PrintProfilingInfo(profile_events[i], op_index, tensor, registration);
        }
    }
    const auto results = GetResults();
    const auto labels = GetLabelList();
    PrintOutput(results, labels);

    LOG(INFO) << "Retriving " << (interpreter_->tensors_size() - 1) << " tensors...\n";
    const auto intermediate_outputs = GetIntermediateOutput();
    LOG(INFO) << "Writing " << intermediate_outputs.size() << " tensors to file...\n";
    std::for_each(intermediate_outputs.begin(), intermediate_outputs.end(),
                  [&](const auto& output) { WriteToFile(GetResultDirectory(), output.first, output.second); });
    LOG(INFO) << "Completed writing " << intermediate_outputs.size() << " tensors to file!!\n";
}

void TFLiteInferenceEngine::Shutdown() {}

void TFLiteInferenceEngine::InvokeInference()
{
    struct timeval start_time;
    struct timeval stop_time;
    gettimeofday(&start_time, nullptr);

    if (interpreter_->Invoke() != TfLiteStatus::kTfLiteOk)
    {
        LOG(FATAL) << "Failed to invoke tflite!\n";
    }

    gettimeofday(&stop_time, nullptr);
    LOG(INFO) << "invoked \n";
    LOG(INFO) << "average time: " << (get_us(stop_time) - get_us(start_time)) / (GetLoopCount() * 1000) << " ms \n";
}

void TFLiteInferenceEngine::SetInputData(const std::vector<std::uint8_t>& image_data)
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
        case TfLiteType::kTfLiteFloat32:
            ResizeImage<float>(interpreter_->typed_tensor<float>(input), image_data.data(), GetImageHeight(),
                               GetImageWidth(), GetImageChannels(), wanted_height, wanted_width, wanted_channels, true,
                               GetInputMean(), GetInputStd());
            break;
        case TfLiteType::kTfLiteUInt8:
            ResizeImage<std::uint8_t>(interpreter_->typed_tensor<std::uint8_t>(input), image_data.data(),
                                      GetImageHeight(), GetImageWidth(), GetImageChannels(), wanted_height,
                                      wanted_width, wanted_channels, false, GetInputMean(), GetInputStd());
            break;
        default:
            throw std::runtime_error("cannot handle input type " + std::to_string(interpreter_->tensor(input)->type) +
                                     " yet");
    }
}

std::vector<std::pair<float, std::int32_t>> TFLiteInferenceEngine::GetResults() const
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
            get_top_n<float>(interpreter_->typed_output_tensor<float>(0), output_size, GetNumberOfThreads(), threshold,
                             &top_results, true);
            break;
        case TfLiteType::kTfLiteUInt8:
            get_top_n<std::uint8_t>(interpreter_->typed_output_tensor<std::uint8_t>(0), output_size,
                                    GetNumberOfThreads(), threshold, &top_results, false);
            break;
        default:
            throw std::runtime_error("cannot handle output type " + std::to_string(interpreter_->tensor(output)->type) +
                                     " yet");
    }
    return top_results;
}

std::vector<std::pair<std::string, std::string>> TFLiteInferenceEngine::GetIntermediateOutput() const
{
    std::vector<std::pair<std::string, std::string>> intermediate_outputs;
    for (auto tensor_index = 0; tensor_index < interpreter_->tensors_size() - 1; tensor_index++)
    {
        const auto tensor = interpreter_->tensor(tensor_index);
        const auto tensor_dims = tensor->dims;
        const auto tensor_channels = tensor_dims->data[tensor_dims->size - 1];

        std::stringstream filename;
        auto tensor_name = std::string{tensor->name};
        std::replace(tensor_name.begin(), tensor_name.end(), '/', '_');
        filename << std::setw(3) << std::setfill('0') << tensor_index << "_" << tensor_name << "_tensor.txt";

        std::stringstream output_stream;
        output_stream << "################################################################################\n"
                      << "# Tensor Details: {\n#   name: " << tensor->name << "\n#   shape: " << tensor->dims
                      << "\n#   index: " << tensor_index << "\n#   type: " << tensor->type << "\n# }\n"
                      << "#\n"
                      << "# Note: Contents are formated as (channel_index, tensor_value) pair\n"
                      << "################################################################################\n";
        for (auto b = 0U, ch = 0U; b < tensor->bytes; ++b, ++ch)
        {
            if (ch > tensor_channels - 1)
            {
                ch = 0;
            }
            output_stream << +ch << " " << +static_cast<std::uint8_t>(tensor->data.raw_const[b]) << "\n";
        }
        std::pair<std::string, std::string> intermediate_output = std::make_pair(filename.str(), output_stream.str());
        intermediate_outputs.push_back(intermediate_output);
    }
    return intermediate_outputs;
}  // namespace perception

}  // namespace perception

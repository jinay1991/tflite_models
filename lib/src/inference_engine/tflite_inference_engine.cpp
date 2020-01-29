///
/// @file tflite_inference_engine.cpp
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
#include "tensorflow/lite/builtin_op_data.h"
#include "tensorflow/lite/core/api/profiler.h"
#include "tensorflow/lite/delegates/nnapi/nnapi_delegate.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/optional_debug_tools.h"
#include "tensorflow/lite/profiling/profile_summarizer.h"
#include "tensorflow/lite/string_util.h"
#include "tensorflow/lite/tools/evaluation/utils.h"
#define TFLITE_PROFILING_ENABLED
#include "tensorflow/lite/profiling/profiler.h"

#include "perception/inference_engine/tflite_inference_engine.h"
#include "perception/logging/logging.h"
#include "perception/utils/get_top_n.h"

namespace perception
{
namespace
{
/// @brief Convert to usec (microseconds)
constexpr double get_us(struct timeval t) { return (t.tv_sec * 1000000 + t.tv_usec); }

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

/// @brief Resize Provided Image using TFLite Interpreter
template <class T>
void ResizeImage(T* out, const std::uint8_t* in, const std::int32_t image_height, const std::int32_t image_width,
                 const std::int32_t image_channels, const std::int32_t wanted_height, const std::int32_t wanted_width,
                 const std::int32_t wanted_channels, const bool input_floating, const float input_mean,
                 const float input_std)
{
    std::int32_t number_of_pixels = image_height * image_width * image_channels;
    std::unique_ptr<tflite::Interpreter> interpreter = std::make_unique<tflite::Interpreter>();

    std::int32_t base_index = 0;

    // two inputs: input and new_sizes
    interpreter->AddTensors(2, &base_index);
    // one output
    interpreter->AddTensors(1, &base_index);
    // set input and output tensors
    interpreter->SetInputs({0, 1});
    interpreter->SetOutputs({2});

    // set parameters of tensors
    TfLiteQuantizationParams quant;
    interpreter->SetTensorParametersReadWrite(0, kTfLiteFloat32, "input",
                                              {1, image_height, image_width, image_channels}, quant);
    interpreter->SetTensorParametersReadWrite(1, kTfLiteInt32, "new_size", {2}, quant);
    interpreter->SetTensorParametersReadWrite(2, kTfLiteFloat32, "output",
                                              {1, wanted_height, wanted_width, wanted_channels}, quant);

    tflite::ops::builtin::BuiltinOpResolver resolver;
    const TfLiteRegistration* resize_op = resolver.FindOp(tflite::BuiltinOperator_RESIZE_BILINEAR, 1);
    auto* params = reinterpret_cast<TfLiteResizeBilinearParams*>(malloc(sizeof(TfLiteResizeBilinearParams)));
    params->align_corners = false;
    interpreter->AddNodeWithParameters({0, 1}, {2}, nullptr, 0, params, resize_op, nullptr);

    interpreter->AllocateTensors();

    // fill input image
    // in[] are integers, cannot do memcpy() directly
    auto input = interpreter->typed_tensor<float>(0);
    for (std::int32_t i = 0; i < number_of_pixels; i++)
    {
        input[i] = in[i];
    }

    // fill new_sizes
    interpreter->typed_tensor<std::int32_t>(1)[0] = wanted_height;
    interpreter->typed_tensor<std::int32_t>(1)[1] = wanted_width;

    interpreter->Invoke();

    auto output = interpreter->typed_tensor<float>(2);
    auto output_number_of_pixels = wanted_height * wanted_width * wanted_channels;

    for (std::int32_t i = 0; i < output_number_of_pixels; i++)
    {
        if (input_floating)
        {
            out[i] = (output[i] - input_mean) / input_std;
        }
        else
        {
            out[i] = static_cast<std::uint8_t>(output[i]);
        }
    }
}

/// @brief Write given content buffer to file
void WriteToFile(const std::string& dirname, const std::string& filename, const std::string& content)
{
    const auto filepath = std::string{dirname + "/" + filename};
    std::ofstream file(filepath, std::ios::binary);
    ASSERT_CHECK(file.is_open()) << "Unable to open " << filepath;
    file << content;
}

}  // namespace

TFLiteInferenceEngine::TFLiteInferenceEngine() {}
TFLiteInferenceEngine::TFLiteInferenceEngine(const CLIOptions& cli_options) : InferenceEngineBase{cli_options} {}

TFLiteInferenceEngine::~TFLiteInferenceEngine() {}

void TFLiteInferenceEngine::Init()
{
    model_ = tflite::FlatBufferModel::BuildFromFile(GetModelPath().c_str());
    ASSERT_CHECK(model_) << "Failed to mmap model " << GetModelPath();
    LOG(INFO) << "Loaded model \"" << GetModelPath() << "\"";
    model_->error_reporter();

    tflite::ops::builtin::BuiltinOpResolver resolver;

    tflite::InterpreterBuilder(*model_, resolver)(&interpreter_);
    ASSERT_CHECK(interpreter_) << "Failed to construct interpreter";
    if (IsVerbosityEnabled())
    {
        LOG(INFO) << "tensors size: " << interpreter_->tensors_size();
        LOG(INFO) << "nodes size: " << interpreter_->nodes_size();
        LOG(INFO) << "inputs: " << interpreter_->inputs().size();
        LOG(INFO) << "input(0) name: " << interpreter_->GetInputName(0);

        int tensor_size = interpreter_->tensors_size();
        for (int i = 0; i < tensor_size; i++)
        {
            if (interpreter_->tensor(i)->name)
            {
                LOG(INFO) << i << ": " << interpreter_->tensor(i)->name << ", " << interpreter_->tensor(i)->bytes
                          << ", " << interpreter_->tensor(i)->type << ", " << interpreter_->tensor(i)->params.scale
                          << ", " << interpreter_->tensor(i)->params.zero_point;
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
        LOG(INFO) << "number of inputs: " << inputs.size();
        LOG(INFO) << "number of outputs: " << outputs.size();
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
    LOG(INFO) << "Loaded image \"" << GetImagePath() << "\"";

    auto profiler = absl::make_unique<tflite::profiling::Profiler>(GetMaxProfilingBufferEntries());
    interpreter_->SetProfiler(profiler.get());
    tflite::profiling::ProfileSummarizer summarizer;

    if (IsProfilingEnabled())
    {
        profiler->StartProfiling();
    }

    InvokeInference();

    if (IsProfilingEnabled())
    {
        profiler->StopProfiling();
        auto profile_events = profiler->GetProfileEvents();
        summarizer.ProcessProfiles(profile_events, *interpreter_);
        profiler->Reset();
        auto summary = summarizer.GetOutputString();
        LOG(INFO) << summary;
        if (IsSaveResultsEnabled())
        {
            WriteToFile(GetResultDirectory(), "performance_metrics.txt", summary);
        }
    }
    const auto results = GetResults();
    const auto labels = GetLabelList();

    std::stringstream content_stream;
    std::for_each(results.begin(), results.end(), [&](const auto& result) {
        const float confidence = result.first;
        const std::int32_t index = result.second;
        content_stream << confidence << ": " << labels[index] << "\n";
    });
    if (IsSaveResultsEnabled())
    {
        WriteToFile(GetResultDirectory(), "top_k_results.txt", content_stream.str());
    }

    LOG(INFO) << "Top " << GetNumberOfResults() << " Results: \n" << content_stream.str();

    if (IsSaveResultsEnabled())
    {
        const auto intermediate_outputs = GetIntermediateOutput();
        std::for_each(intermediate_outputs.begin(), intermediate_outputs.end(),
                      [&](const auto& output) { WriteToFile(GetResultDirectory(), output.first, output.second); });
    }
}

void TFLiteInferenceEngine::Shutdown() {}

void TFLiteInferenceEngine::InvokeInference()
{
    struct timeval start_time;
    struct timeval stop_time;
    gettimeofday(&start_time, nullptr);

    auto error_code = interpreter_->Invoke();
    ASSERT_CHECK_EQ(error_code, TfLiteStatus::kTfLiteOk) << "Failed to invoke tflite!";

    gettimeofday(&stop_time, nullptr);
    auto avg_time_in_ms = (get_us(stop_time) - get_us(start_time)) / (GetLoopCount() * 1000);
    auto images_per_sec = (1.0 / avg_time_in_ms) * 1000.0;

    LOG(INFO) << "Average time taken: " << avg_time_in_ms << " ms. (i.e. " << images_per_sec << " images/second) ";

    if (IsSaveResultsEnabled())
    {
        WriteToFile(GetResultDirectory(), "images_per_second.txt",
                    "images_per_second: " + std::to_string(images_per_sec));
    }
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
            get_top_n<float>(interpreter_->typed_output_tensor<float>(0), output_size, GetNumberOfResults(), threshold,
                             &top_results, true);
            break;
        case TfLiteType::kTfLiteUInt8:
            get_top_n<std::uint8_t>(interpreter_->typed_output_tensor<std::uint8_t>(0), output_size,
                                    GetNumberOfResults(), threshold, &top_results, false);
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
    for (std::size_t tensor_index = 0; tensor_index < interpreter_->tensors_size() - 1; tensor_index++)
    {
        const auto tensor = interpreter_->tensor(tensor_index);
        if (tensor->name == nullptr)
        {
            continue;
        }

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
        std::int32_t ch = 0;
        for (auto b = 0U; b < tensor->bytes; ++b, ++ch)
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

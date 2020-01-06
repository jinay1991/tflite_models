///
/// @file inference_engine_base.cpp
/// @copyright Copyright (c) 2020. All Rights Reserved.
///
#include <cstdint>
#include <experimental/filesystem>
#include <fstream>

#include "absl/strings/match.h"

#include "perception/image_helper/bitmap_helper.h"
#include "perception/image_helper/jpeg_helper.h"
#include "perception/inference_engine/inference_engine_base.h"
#include "perception/logging/logging.h"
#include "perception/utils/get_top_n.h"

#define ASSERT_PATH_EXISTS(path)                                   \
    do                                                             \
    {                                                              \
        if (!std::experimental::filesystem::exists(path))          \
        {                                                          \
            throw std::runtime_error(path + " does not exists!!"); \
        }                                                          \
    } while (0);

namespace perception
{
InferenceEngineBase::InferenceEngineBase() : InferenceEngineBase{CLIOptions{}} {}
InferenceEngineBase::InferenceEngineBase(const CLIOptions& cli_options)
    : cli_options_{cli_options}, channels_{3}, height_{224}, width_{224}
{
    ASSERT_PATH_EXISTS(cli_options_.model_name);
    ASSERT_PATH_EXISTS(cli_options_.labels_name);
    ASSERT_PATH_EXISTS(cli_options_.input_name);

    if (absl::EndsWith(cli_options_.input_name, ".bmp"))
    {
        image_helper_ = std::make_unique<BitmapImageHelper>();
    }
    else
    {
        image_helper_ = std::make_unique<JpegImageHelper>();
    }
}

InferenceEngineBase::~InferenceEngineBase() {}

std::vector<std::string> InferenceEngineBase::GetLabelList() const
{
    std::vector<std::string> labels;
    std::ifstream file(cli_options_.labels_name);
    ASSERT_CHECK(file.is_open()) << "Labels file " << cli_options_.labels_name << " not found";

    labels.clear();
    std::string line;
    while (std::getline(file, line))
    {
        labels.push_back(line);
    }
    /// @note It pads with empty strings so the length
    /// of the result is a multiple of 16, because our model expects that.
    const std::int32_t padding = 16;
    while (labels.size() % padding)
    {
        labels.emplace_back();
    }
    return labels;
}

std::vector<std::uint8_t> InferenceEngineBase::GetImageData()
{
    return image_helper_->ReadImage(cli_options_.input_name, &width_, &height_, &channels_);
}

std::int32_t InferenceEngineBase::GetImageWidth() const { return width_; }
std::int32_t InferenceEngineBase::GetImageHeight() const { return height_; }
std::int32_t InferenceEngineBase::GetImageChannels() const { return channels_; }
std::string InferenceEngineBase::GetModelPath() const { return cli_options_.model_name; }
std::string InferenceEngineBase::GetImagePath() const { return cli_options_.input_name; }

bool InferenceEngineBase::IsSaveResultsEnabled() const { return cli_options_.save_results; }
bool InferenceEngineBase::IsProfilingEnabled() const { return cli_options_.profiling; }
bool InferenceEngineBase::IsVerbosityEnabled() const { return cli_options_.verbose; }
std::int32_t InferenceEngineBase::GetNumberOfThreads() const { return cli_options_.number_of_threads; }
std::int32_t InferenceEngineBase::GetNumberOfResults() const { return cli_options_.number_of_results; }
std::int32_t InferenceEngineBase::GetMaxProfilingBufferEntries() const
{
    return cli_options_.max_profiling_buffer_entries;
}
std::string InferenceEngineBase::GetResultDirectory() const
{
    const auto dirname = cli_options_.result_directory;
    if (!std::experimental::filesystem::exists(dirname))
    {
        if (!std::experimental::filesystem::create_directory(dirname))
        {
            throw std::runtime_error("Unable to create directory {" + dirname + "}");
        }
    }
    return dirname;
}
float InferenceEngineBase::GetInputMean() const { return cli_options_.input_mean; }
float InferenceEngineBase::GetInputStd() const { return cli_options_.input_std; }
std::int32_t InferenceEngineBase::GetLoopCount() const { return cli_options_.loop_count; }
}  // namespace perception

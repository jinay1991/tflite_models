///
/// @file
/// @copyright Copyright (c) 2020. All Rights Reserved.
///
#include <cstdint>
#include <experimental/filesystem>
#include <fstream>

#include "absl/strings/match.h"

#include "perception/inference_engine/inference_engine_base.h"
#include "perception/utils/bitmap_helper.h"
#include "perception/utils/get_top_n.h"
#include "perception/utils/jpeg_helper.h"
#include "perception/utils/resize.h"

#define ASSERT_PATH_EXISTS(path)                                    \
    do                                                              \
    {                                                               \
        if (!std::experimental::filesystem::exists(path))           \
        {                                                           \
            throw std::runtime_error(path + "does not exists!!\n"); \
        }                                                           \
    } while (0);

namespace perception
{
InferenceEngineBase::InferenceEngineBase(const CLIOptions& cli_options)
    : cli_options_{cli_options},
      image_path_{cli_options_.input_name},
      model_path_{cli_options_.model_name},
      label_path_{cli_options_.labels_name},
      width_{224},
      height_{224},
      channels_{3}
{
    ASSERT_PATH_EXISTS(model_path_);
    ASSERT_PATH_EXISTS(image_path_);
    ASSERT_PATH_EXISTS(label_path_);

    if (absl::EndsWith(image_path_, ".bmp"))
    {
        image_helper_ = std::make_unique<BitmapImageHelper>();
    }
    else
    {
        image_helper_ = std::make_unique<JpegImageHelper>();
    }
}

std::vector<std::string> InferenceEngineBase::GetLabelList()
{
    std::vector<std::string> labels;
    std::ifstream file(label_path_);
    if (!file)
    {
        throw std::runtime_error("Labels file " + label_path_ + " not found\n");
    }
    labels.clear();
    std::string line;
    while (std::getline(file, line))
    {
        labels.push_back(line);
    }
    /// @note It pads with empty strings so the length
    /// of the result is a multiple of 16, because our model expects that.
    label_count_ = labels.size();
    const std::int32_t padding = 16;
    while (labels.size() % padding)
    {
        labels.emplace_back();
    }
    return labels;
}
std::int32_t InferenceEngineBase::GetLabelCount() const { return label_count_; }

std::vector<std::uint8_t> InferenceEngineBase::GetImageData()
{
    return image_helper_->ReadImage(image_path_, &width_, &height_, &channels_);
}

std::int32_t InferenceEngineBase::GetImageWidth() const { return width_; }
std::int32_t InferenceEngineBase::GetImageHeight() const { return height_; }
std::int32_t InferenceEngineBase::GetImageChannels() const { return channels_; }
std::string InferenceEngineBase::GetImagePath() const { return image_path_; }
std::string InferenceEngineBase::GetModelPath() const { return model_path_; }
std::string InferenceEngineBase::GetLabelPath() const { return label_path_; }

bool InferenceEngineBase::IsProfilingEnabled() const { return cli_options_.profiling; }
bool InferenceEngineBase::IsVerbosityEnabled() const { return cli_options_.verbose; }
std::int32_t InferenceEngineBase::GetNumberOfThreads() const { return cli_options_.number_of_threads; }
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
            throw std::runtime_error("Unable to create directory\n");
        }
    }
    return dirname;
}
float InferenceEngineBase::GetInputMean() const { return cli_options_.input_mean; }
float InferenceEngineBase::GetInputStd() const { return cli_options_.input_std; }
std::int32_t InferenceEngineBase::GetLoopCount() const { return cli_options_.loop_count; }
}  // namespace perception

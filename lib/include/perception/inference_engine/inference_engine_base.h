///
/// @file
/// @copyright Copyright (c) 2020. All Rights Reserved.
///
#ifndef PERCEPTION_INFERENCE_ENGINE_INFERENCE_ENGINE_BASE_H_
#define PERCEPTION_INFERENCE_ENGINE_INFERENCE_ENGINE_BASE_H_

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "perception/argument_parser/cli_options.h"
#include "perception/inference_engine/i_inference_engine.h"
#include "perception/utils/i_image_helper.h"

namespace perception
{
class InferenceEngineBase : public IInferenceEngine
{
  public:
    InferenceEngineBase();
    explicit InferenceEngineBase(const CLIOptions& cli_options);
    virtual ~InferenceEngineBase();

  protected:
    virtual std::vector<std::uint8_t> GetImageData();
    virtual std::vector<std::string> GetLabelList() const;

    virtual std::int32_t GetImageWidth() const;
    virtual std::int32_t GetImageHeight() const;
    virtual std::int32_t GetImageChannels() const;
    virtual std::string GetModelPath() const;

    virtual bool IsProfilingEnabled() const;
    virtual bool IsVerbosityEnabled() const;
    virtual std::int32_t GetNumberOfThreads() const;
    virtual std::int32_t GetMaxProfilingBufferEntries() const;
    virtual float GetInputMean() const;
    virtual float GetInputStd() const;
    virtual std::int32_t GetLoopCount() const;
    virtual std::string GetResultDirectory() const;

  private:
    CLIOptions cli_options_;
    std::int32_t channels_;
    std::int32_t height_;
    std::int32_t label_count_;
    std::int32_t width_;
    std::string image_path_;
    std::string label_path_;
    std::string model_path_;
    std::unique_ptr<IImageHelper> image_helper_;
};

}  // namespace perception
#endif  /// PERCEPTION_INFERENCE_ENGINE_INFERENCE_ENGINE_BASE_H_
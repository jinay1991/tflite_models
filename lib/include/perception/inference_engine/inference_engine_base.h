///
/// @file inference_engine_base.h
/// @brief Contains class definitions for Inference Engine base class
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
#include "perception/image_helper/i_image_helper.h"
#include "perception/inference_engine/i_inference_engine.h"

namespace perception
{
/// @brief Inference Engine Base Abstract class
class InferenceEngineBase : public IInferenceEngine
{
  public:
    /// @brief Default Constructor
    InferenceEngineBase();

    /// @brief Constructor
    /// @param [in] cli_options - Command Line Interface Options
    explicit InferenceEngineBase(const CLIOptions& cli_options);

    /// @brief Destructor
    virtual ~InferenceEngineBase();

  protected:
    /// @brief Provides Decoded Image Data
    virtual std::vector<std::uint8_t> GetImageData();

    /// @brief Provides Labels List
    virtual std::vector<std::string> GetLabelList() const;

    /// @brief Provides Image Width
    virtual std::int32_t GetImageWidth() const;

    /// @brief Provides Image Height
    virtual std::int32_t GetImageHeight() const;

    /// @brief Provides Image Channels
    virtual std::int32_t GetImageChannels() const;

    /// @brief Provides Model Path
    virtual std::string GetModelPath() const;

    /// @brief Provides Image Path
    virtual std::string GetImagePath() const;

    /// @brief Reads CLI Option for Profiling Enabled?
    /// @return true if cli arg `--profiling` is set to 1, else false
    virtual bool IsProfilingEnabled() const;

    /// @brief Reads CLI Option for Verbosity Enabled?
    /// @return true if cli arg `--verbose` is set to 1, else false
    virtual bool IsVerbosityEnabled() const;

    /// @brief Reads CLI Option for Save Results Enabled?
    /// @return true if cli arg `--save_results` is set to 1, else false.
    virtual bool IsSaveResultsEnabled() const;

    /// @brief Reads CLI Option for Number of threads to use for Inference
    virtual std::int32_t GetNumberOfThreads() const;

    /// @brief Reads CLI Option for Number of Results to produce
    virtual std::int32_t GetNumberOfResults() const;

    /// @brief Reads CLI Option for Maximum Profiling Buffer Entries.
    /// @note  Used only when `-p` is set to 1.
    virtual std::int32_t GetMaxProfilingBufferEntries() const;

    /// @brief Reads CLI Option for Input Mean
    virtual float GetInputMean() const;

    /// @brief Reads CLI Option for Input stddev
    virtual float GetInputStd() const;

    /// @brief Reads CLI Option for loop count
    virtual std::int32_t GetLoopCount() const;

    /// @brief Reads CLI Option for result directory
    virtual std::string GetResultDirectory() const;

  private:
    /// @brief Command Line Interface Options
    CLIOptions cli_options_;
    /// @brief Image Channels
    std::int32_t channels_;
    /// @brief Image Height
    std::int32_t height_;
    /// @brief Image Width
    std::int32_t width_;
    /// @brief Image Path
    std::string image_path_;
    /// @brief Label Path
    std::string label_path_;

    /// @brief Model Path
    std::string model_path_;

    /// @brief Image Reader Helper
    std::unique_ptr<IImageHelper> image_helper_;
};

}  // namespace perception
#endif  /// PERCEPTION_INFERENCE_ENGINE_INFERENCE_ENGINE_BASE_H_
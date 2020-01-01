///
/// @file
/// @copyright Copyright (c) 2020. All Rights Reserved.
///
#ifndef PERCEPTION_INFERENCE_ENGINE_TFLITE_INFERENCE_ENGINE_H_
#define PERCEPTION_INFERENCE_ENGINE_TFLITE_INFERENCE_ENGINE_H_

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/model.h"

#include "perception/argument_parser/cli_options.h"
#include "perception/image_helper/i_image_helper.h"
#include "perception/inference_engine/inference_engine_base.h"

namespace perception
{
class TFLiteInferenceEngine : public InferenceEngineBase
{
  public:
    TFLiteInferenceEngine();
    explicit TFLiteInferenceEngine(const CLIOptions& cli_options);
    virtual ~TFLiteInferenceEngine();

    /// @brief Initialise TFLite Inference Engine
    virtual void Init() override;
    
    /// @brief Execute Inference with TFLite Inference Engine
    virtual void Execute() override;
    
    /// @brief Release TFLite Inference Engine
    virtual void Shutdown() override;

  protected:
    /// @brief Obtain Intermediate Layers/Operations Output
    virtual std::vector<std::pair<std::string, std::string>> GetIntermediateOutput() const override;

    /// @brief Obtain Results for provided Image
    virtual std::vector<std::pair<float, std::int32_t>> GetResults() const override;

  private:
    /// @brief Invokes Inference with TFLite Interpreter
    virtual void InvokeInference();

    /// @brief Set Image Data to Model Input (via Interpreter)
    virtual void SetInputData(const std::vector<std::uint8_t>& image_data);

    std::unique_ptr<tflite::FlatBufferModel> model_;
    std::unique_ptr<tflite::Interpreter> interpreter_;
};

}  // namespace perception
#endif  /// PERCEPTION_INFERENCE_ENGINE_TFLITE_INFERENCE_ENGINE_H_
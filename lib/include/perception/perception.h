///
/// @file
/// @copyright Copyright (c) 2020. All Rights Reserved.
///
#ifndef PERCEPTION_PERCEPTION_H_
#define PERCEPTION_PERCEPTION_H_

#include "perception/argument_parser/i_argument_parser.h"
#include "perception/inference_engine/i_inference_engine.h"

namespace perception
{
class Perception
{
  public:
    /// @brief Supported Inference Engines
    enum class InferenceEngineType : std::int32_t
    {
        kInvalid = 0,
        kTFLiteInferenceEngine = 1,
    };

    explicit Perception(std::unique_ptr<IArgumentParser> argument_parser);
    virtual ~Perception();

    /// @brief Selects Inference Engine type and creates instance of it.
    virtual void SelectInferenceEngine(const InferenceEngineType& type);

    /// @brief Initialise Inference Engine
    virtual void Init();

    /// @brief Executes Inference Engine for given Image, n times. n=cli.loop_count
    virtual void Execute();

    /// @brief Release Inference Engine
    virtual void Shutdown();

  private:
    std::unique_ptr<IInferenceEngine> inference_engine_;
    std::unique_ptr<IArgumentParser> argument_parser_;
};

}  // namespace perception

#endif  /// PERCEPTION_PERCEPTION_H_
///
/// @file perception.h
/// @brief Contains Perception Application class definitions
/// @copyright Copyright (c) 2020. All Rights Reserved.
///
#ifndef PERCEPTION_PERCEPTION_H_
#define PERCEPTION_PERCEPTION_H_

#include "perception/argument_parser/i_argument_parser.h"
#include "perception/inference_engine/i_inference_engine.h"

namespace perception
{
/// @brief Perception application class
class Perception
{
  public:
    /// @brief Supported Inference Engines
    enum class InferenceEngineType : std::int32_t
    {
        kInvalid = 0,
        kTFLiteInferenceEngine = 1,
        kTFInferenceEngine = 2,
        kTorchInferenceEngine = 3
    };

    /// @brief Constructor
    /// @param [in] argument_parser - Instance of Argument Parser
    explicit Perception(std::unique_ptr<IArgumentParser> argument_parser);

    /// @brief Destructor
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
    /// @brief Inference Engine Instance.
    std::unique_ptr<IInferenceEngine> inference_engine_;

    /// @brief Argument Parser Instance, which contains parsed args.
    std::unique_ptr<IArgumentParser> argument_parser_;
};

}  // namespace perception

#endif  /// PERCEPTION_PERCEPTION_H_
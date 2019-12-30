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
    enum class InferenceEngineType : std::int32_t
    {
        kInvalid,
        kTFLiteInferenceEngine,
    };

    explicit Perception(std::unique_ptr<IArgumentParser> argument_parser);
    virtual ~Perception() = default;

    virtual void SelectInferenceEngine(const InferenceEngineType& type);

    virtual void Init();
    virtual void Execute();
    virtual void Shutdown();

  private:
    std::unique_ptr<IInferenceEngine> inference_engine_;
    std::unique_ptr<IArgumentParser> argument_parser_;
};

}  // namespace perception

#endif  /// PERCEPTION_PERCEPTION_H_
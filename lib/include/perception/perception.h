///
/// @file
/// @copyright Copyright (c) 2019. All Rights Reserved.
///
#ifndef PERCEPTION_PERCEPTION_H_
#define PERCEPTION_PERCEPTION_H_

#include "perception/cli.h"
#include "perception/i_inference_engine.h"

namespace perception
{
class Perception
{
  public:
    enum class InferenceType : std::int32_t
    {
        kNone = 0,
        kClassification = 1,
        kDetection = 2
    };

    explicit Perception(const CLIOptions& cli_opts);
    virtual ~Perception();

    virtual void Init();

    virtual void SetInferenceType(const InferenceType& inference_type);

    virtual void RunInference(const std::string& image_path);

  private:
    bool initialised_;
    CLIOptions cli_opts_;
    std::unique_ptr<IInferenceEngine> inference_engine_;
};

}  // namespace perception

#endif  /// PERCEPTION_PERCEPTION_H_
///
/// @file
/// @copyright Copyright (c) 2019. All Rights Reserved.
///
#ifndef PERCEPTION_I_INFERENCE_ENGINE_H_
#define PERCEPTION_I_INFERENCE_ENGINE_H_

#include <string>

namespace perception
{
class IInferenceEngine
{
  public:
    virtual void Init() = 0;
    virtual void ExecuteStep(const std::string& image_path) = 0;
    virtual void Shutdown() = 0;
};
}  // namespace perception
#endif  /// PERCEPTION_I_INFERENCE_H_
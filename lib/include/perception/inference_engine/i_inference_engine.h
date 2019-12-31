///
/// @file
/// @copyright Copyright (c) 2020. All Rights Reserved.
///
#ifndef PERCEPTION_INFERENCE_ENGINE_I_INFERENCE_ENGINE_H_
#define PERCEPTION_INFERENCE_ENGINE_I_INFERENCE_ENGINE_H_

#include <cstdint>
#include <string>

namespace perception
{
class IInferenceEngine
{
  public:
    virtual ~IInferenceEngine() = default;
    virtual void Init() = 0;
    virtual void Execute() = 0;
    virtual void Shutdown() = 0;

  protected:
    virtual std::vector<std::pair<std::string, std::string>> GetIntermediateOutput() const = 0;
    virtual std::vector<std::pair<float, std::int32_t>> GetResults() const = 0;
};
}  // namespace perception
#endif  /// PERCEPTION_INFERENCE_ENGINE_I_INFERENCE_ENGINE_H_
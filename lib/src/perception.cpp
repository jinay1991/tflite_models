///
/// @file
/// @copyright Copyright (c) 2019. All Rights Reserved.
///
#include "perception/perception.h"
#include "perception/inference_engine.h"

namespace perception
{
Perception::Perception(const CLIOptions& cli_opts)
    : initialised_{false}, cli_opts_{cli_opts}, inference_engine_{std::make_unique<InferenceEngine>(cli_opts_)}
{
}

Perception::~Perception() { inference_engine_->Shutdown(); }

void Perception::SetInferenceType(const InferenceType& inference_type)
{
    switch (inference_type)
    {
        case InferenceType::kDetection:
            throw std::runtime_error("Unsupported for Detection tasks\n");
        case InferenceType::kClassification:
        default:
            inference_engine_ = std::make_unique<InferenceEngine>(cli_opts_);
            break;
    }
}
void Perception::Init()
{
    inference_engine_->Init();
    initialised_ = true;
}

void Perception::RunInference(const std::string& image_path)
{
    if (!initialised_)
    {
        throw std::runtime_error("Inference Engine is uninitialized!!\n");
    }
    inference_engine_->ExecuteStep(image_path);
}

}  // namespace perception

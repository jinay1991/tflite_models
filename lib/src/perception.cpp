///
/// @file
/// @copyright Copyright (c) 2020. All Rights Reserved.
///
#include <string>

#include "perception/inference_engine/tflite_inference_engine.h"
#include "perception/perception.h"

namespace perception
{
Perception::Perception(std::unique_ptr<IArgumentParser> argument_parser) : argument_parser_{std::move(argument_parser)}
{
}

Perception::~Perception() {}

void Perception::SelectInferenceEngine(const InferenceEngineType& type)
{
    switch (type)
    {
        case InferenceEngineType::kTFLiteInferenceEngine:
            inference_engine_ = std::make_unique<TFLiteInferenceEngine>(argument_parser_->GetParsedArgs());
            break;
        case InferenceEngineType::kInvalid:
        default:
            throw std::runtime_error("Unsupported for InferenceEngine \n");
            break;
    }
}
void Perception::Init() { inference_engine_->Init(); }

void Perception::Execute()
{
    for (auto iter = 0; iter < argument_parser_->GetParsedArgs().loop_count; ++iter)
    {
        inference_engine_->Execute();
    }
}

void Perception::Shutdown() { inference_engine_->Shutdown(); }

}  // namespace perception

///
/// @file
/// @copyright Copyright (c) 2020. All Rights Reserved.
///
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <memory>

#include "perception/argument_parser/argument_parser.h"
#include "perception/perception.h"

namespace perception
{
namespace
{
class PerceptionTestFixture : public ::testing::Test
{
  public:
    PerceptionTestFixture()
        : argument_parser_{std::make_unique<ArgumentParser>()},
          unit_{std::make_unique<Perception>(std::move(argument_parser_))}
    {
    }

  protected:
    std::unique_ptr<IArgumentParser> argument_parser_;
    std::unique_ptr<Perception> unit_;
};

TEST_F(PerceptionTestFixture, WhenSelectInferenceEngine)
{
    EXPECT_THROW(unit_->SelectInferenceEngine(Perception::InferenceEngineType::kInvalid), std::runtime_error);
    EXPECT_NO_THROW(unit_->SelectInferenceEngine(Perception::InferenceEngineType::kTFLiteInferenceEngine));
}

TEST_F(PerceptionTestFixture, WhenInitialized)
{
    unit_->SelectInferenceEngine(Perception::InferenceEngineType::kTFLiteInferenceEngine);
    EXPECT_NO_THROW(unit_->Init());
}

TEST_F(PerceptionTestFixture, WhenExecute)
{
    unit_->SelectInferenceEngine(Perception::InferenceEngineType::kTFLiteInferenceEngine);
    unit_->Init();

    EXPECT_NO_THROW(unit_->Execute());
    EXPECT_NO_THROW(unit_->Shutdown());
}

TEST_F(PerceptionTestFixture, WhenInvalidInferenceEngine)
{
    EXPECT_THROW(unit_->SelectInferenceEngine(Perception::InferenceEngineType::kInvalid), std::runtime_error);
}

}  // namespace
}  // namespace perception
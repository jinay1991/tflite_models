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
TEST(PerceptionTest, WhenSelectInferenceEngine)
{
    auto argument_parser = std::make_unique<ArgumentParser>();
    Perception unit{std::move(argument_parser)};

    EXPECT_THROW(unit.SelectInferenceEngine(Perception::InferenceEngineType::kInvalid), std::runtime_error);
    EXPECT_NO_THROW(unit.SelectInferenceEngine(Perception::InferenceEngineType::kTFLiteInferenceEngine));
}

TEST(PerceptionTest, WhenInitialized)
{
    auto argument_parser = std::make_unique<ArgumentParser>();
    Perception unit{std::move(argument_parser)};
    EXPECT_NO_THROW(unit.SelectInferenceEngine(Perception::InferenceEngineType::kTFLiteInferenceEngine));

    EXPECT_NO_THROW(unit.Init());
}
TEST(PerceptionTest, WhenExecute)
{
    auto argument_parser = std::make_unique<ArgumentParser>();
    Perception unit{std::move(argument_parser)};
    EXPECT_NO_THROW(unit.SelectInferenceEngine(Perception::InferenceEngineType::kTFLiteInferenceEngine));
    EXPECT_NO_THROW(unit.Init());

    EXPECT_NO_THROW(unit.Execute());
    EXPECT_NO_THROW(unit.Shutdown());
}
}  // namespace
}  // namespace perception
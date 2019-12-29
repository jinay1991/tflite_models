///
/// @file
/// @copyright Copyright (c) 2020. All Rights Reserved.
///
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#define private public
#define protected public
#include "perception/inference_engine/tflite_inference_engine.h"
#define private private
#define protected protected

namespace perception
{
namespace
{
class TFLiteInferenceEngineTestFixture : public ::testing::Test
{
  public:
    TFLiteInferenceEngineTestFixture() : cli_options_{}, unit_{cli_options_} {}

  protected:
    CLIOptions cli_options_;
    TFLiteInferenceEngine unit_;
};

TEST_F(TFLiteInferenceEngineTestFixture, WhenInitialized)
{
    EXPECT_EQ(unit_.model_.get(), nullptr);
    EXPECT_EQ(unit_.interpreter_.get(), nullptr);

    EXPECT_NO_THROW(unit_.Init());

    EXPECT_NE(unit_.model_.get(), nullptr);
    EXPECT_NE(unit_.interpreter_.get(), nullptr);
}

TEST_F(TFLiteInferenceEngineTestFixture, WhenExecute)
{
    EXPECT_NO_THROW(unit_.Init());

    EXPECT_NO_THROW(unit_.Execute());

    EXPECT_EQ(unit_.GetResults().size(), 4);
}
}  // namespace
}  // namespace perception
///
/// @file
/// @copyright Copyright (c) 2020. All Rights Reserved.
///
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#define private public
#define protected public
#include "perception/inference_engine/tflite_inference_engine.h"

namespace perception
{
namespace
{
TEST(TFLiteInferenceEngineTest, WhenInitialized)
{
    TFLiteInferenceEngine unit;
    EXPECT_EQ(unit.model_.get(), nullptr);
    EXPECT_EQ(unit.interpreter_.get(), nullptr);

    EXPECT_NO_THROW(unit.Init());

    EXPECT_NE(unit.model_.get(), nullptr);
    EXPECT_NE(unit.interpreter_.get(), nullptr);
}

TEST(TFLiteInferenceEngineTest, WhenExecute)
{
    TFLiteInferenceEngine unit;
    EXPECT_NO_THROW(unit.Init());

    EXPECT_NO_THROW(unit.Execute());

    EXPECT_NO_THROW(unit.Shutdown());
    EXPECT_EQ(unit.GetResults().size(), CLIOptions().number_of_results);
}

TEST(TFLiteInferenceEngineTest, WhenExecuteWithCLIOptions)
{
    CLIOptions cli_options;
    cli_options.number_of_results = 2;
    TFLiteInferenceEngine unit{cli_options};
    EXPECT_NO_THROW(unit.Init());

    EXPECT_NO_THROW(unit.Execute());

    EXPECT_EQ(unit.GetResults().size(), cli_options.number_of_results);
}
TEST(TFLiteInferenceEngineTest, WhenInvalidModelPath)
{
    CLIOptions cli_options;
    cli_options.model_name = "test.tflite";
    EXPECT_THROW(TFLiteInferenceEngine{cli_options}, std::runtime_error);
}
TEST(TFLiteInferenceEngineTest, WhenInvalidLabelPath)
{
    TFLiteInferenceEngine unit;
    unit.label_path_ = "invalid";
    EXPECT_THROW(unit.GetLabelList(), std::runtime_error);
}
}  // namespace
}  // namespace perception
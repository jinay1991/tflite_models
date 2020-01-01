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

TEST(TFLiteInferenceEngineTest, WhenBitmapImage)
{
    CLIOptions cli_options;
    cli_options.input_name = "data/grace_hopper.bmp";
    TFLiteInferenceEngine unit{cli_options};
    EXPECT_NO_THROW(unit.Init());

    EXPECT_NO_THROW(unit.Execute());

    EXPECT_NO_THROW(unit.Shutdown());
    EXPECT_EQ(unit.GetResults().size(), CLIOptions().number_of_results);
}

TEST(TFLiteInferenceEngineTest, WhenExecuteWithCLIOptions)
{
    CLIOptions cli_options;
    cli_options.number_of_results = 2;
    cli_options.profiling = true;
    cli_options.verbose = true;
    cli_options.save_results = true;
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
    unit.cli_options_.labels_name = "test.txt";
    EXPECT_EXIT(unit.GetLabelList(), ::testing::KilledBySignal(SIGABRT), "");
}
}  // namespace
}  // namespace perception
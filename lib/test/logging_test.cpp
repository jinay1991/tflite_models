///
/// @file
/// @copyright Copyright (c) 2020. All Rights Reserved.
///
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "perception/logging/logging.h"

namespace perception
{
namespace logging
{
namespace
{
TEST(LoggingWrapperTest, AssertionMacro) { EXPECT_EXIT(ASSERT_CHECK(false), ::testing::KilledBySignal(SIGABRT), ""); }
TEST(LoggingWrapperTest, AssertionCompareMacro)
{
    EXPECT_EXIT(ASSERT_CHECK_EQ(1, 2), ::testing::KilledBySignal(SIGABRT), "");
}

TEST(LoggingWrapperTest, FATAL_SeverityLevel) { EXPECT_EXIT(LOG(FATAL), ::testing::KilledBySignal(SIGABRT), ""); }
TEST(LoggingWrapperTest, NoLogging)
{
    ::testing::internal::CaptureStderr();
    ::testing::internal::CaptureStdout();
    LoggingWrapper(LoggingWrapper::LogSeverity::INFO, false);
    LoggingWrapper(LoggingWrapper::LogSeverity::ERROR, false);
    LoggingWrapper(LoggingWrapper::LogSeverity::WARN, false);
    EXPECT_TRUE(::testing::internal::GetCapturedStdout().empty());
    EXPECT_TRUE(::testing::internal::GetCapturedStderr().empty());
}
TEST(LoggingWrapperTest, OnStandardOutput)
{
    ::testing::internal::CaptureStderr();
    ::testing::internal::CaptureStdout();
    LoggingWrapper(LoggingWrapper::LogSeverity::INFO, true);
    LoggingWrapper(LoggingWrapper::LogSeverity::WARN, true);
    EXPECT_FALSE(::testing::internal::GetCapturedStdout().empty());
    EXPECT_TRUE(::testing::internal::GetCapturedStderr().empty());
}
TEST(LoggingWrapperTest, OnStandardError)
{
    ::testing::internal::CaptureStderr();
    ::testing::internal::CaptureStdout();
    LoggingWrapper(LoggingWrapper::LogSeverity::ERROR, true);
    EXPECT_TRUE(::testing::internal::GetCapturedStdout().empty());
    EXPECT_FALSE(::testing::internal::GetCapturedStderr().empty());
}
}  // namespace
}  // namespace logging
}  // namespace perception
/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
==============================================================================*/
#ifndef PERCEPTION_LOGGING_LOGGING_H_
#define PERCEPTION_LOGGING_LOGGING_H_

#include <cstdint>
#include <iostream>
#include <sstream>

namespace perception
{
namespace logging
{
class LoggingWrapper
{
  public:
    /// @brief Log Severity Levels
    enum class LogSeverity : std::int32_t
    {
        INFO = 0,
        WARN = 1,
        ERROR = 2,
        FATAL = 3
    };

    /// @brief Constructor
    /// @param [in] severity - Logging Severity
    explicit LoggingWrapper(const LogSeverity& severity);

    /// @brief Constructor
    /// @param [in] severity - Logging Severity
    /// @param [in] should_log - Enable/Disable Logging
    explicit LoggingWrapper(const LogSeverity& severity, const bool should_log);

    /// @brief Destructor
    ~LoggingWrapper();

    /// @brief Provides Stream containing logged contents
    std::stringstream& Stream();

  private:
    /// @brief Logged String Stream
    std::stringstream stream_;

    /// @brief Severity Level for Logging
    LogSeverity severity_;

    /// @brief Enable/Disable Logging
    bool should_log_;
};
}  // namespace logging
}  // namespace perception

/// @brief Log Stream with provided severity level
/// @param [in] severity - Severity Level (INFO, WARN, ERROR, FATAL)
#define LOG(severity) \
    perception::logging::LoggingWrapper(perception::logging::LoggingWrapper::LogSeverity::severity).Stream()

/// @brief Checks for Assertion. If condition is false, Log FATAL Error and exit program.
/// @param [in] condition - condition to be evaluated
#define ASSERT_CHECK(condition)                                                                  \
    perception::logging::LoggingWrapper(perception::logging::LoggingWrapper::LogSeverity::FATAL, \
                                        (condition) ? false : true)                              \
        .Stream()

/// @brief Checks for Assertion for Comparision. If a and b are not same, Log FATAL Error and exit program.
/// @param [in] a - attribute a
/// @param [in] b - attribute b
#define ASSERT_CHECK_EQ(a, b) ASSERT_CHECK(a == b)

#endif  /// PERCEPTION_LOGGING_LOGGING_H_
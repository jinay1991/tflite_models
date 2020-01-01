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
    enum class LogSeverity : std::int32_t
    {
        INFO = 0,
        WARN = 1,
        ERROR = 2,
        FATAL = 3
    };

    explicit LoggingWrapper(const LogSeverity& severity);

    explicit LoggingWrapper(const LogSeverity& severity, const bool should_log);

    ~LoggingWrapper();

    std::stringstream& Stream();

  private:
    std::stringstream stream_;
    LogSeverity severity_;
    bool should_log_;
};
}  // namespace logging
}  // namespace perception

#define LOG(severity) \
    perception::logging::LoggingWrapper(perception::logging::LoggingWrapper::LogSeverity::severity).Stream()

#define CHECK(condition)                                                                         \
    perception::logging::LoggingWrapper(perception::logging::LoggingWrapper::LogSeverity::FATAL, \
                                        (condition) ? false : true)                              \
        .Stream()

#define CHECK_EQ(a, b) BENCHMARK_CHECK(a == b)

#endif  /// PERCEPTION_LOGGING_LOGGING_H_
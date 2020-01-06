///
/// @file i_argument_parser.h
/// @brief Contains Argument Parser Interface class
/// @copyright Copyright (c) 2020. All Rights Reserved.
///
#ifndef PERCEPTION_ARGUMENT_PARSER_I_ARGUMENT_PARSER_H_
#define PERCEPTION_ARGUMENT_PARSER_I_ARGUMENT_PARSER_H_

#include "perception/argument_parser/cli_options.h"

namespace perception
{
/// @brief Argument Parser Interface class
class IArgumentParser
{
  public:
    /// @brief Destructor
    virtual ~IArgumentParser() = default;

    /// @brief Provides Parsed Arguments
    virtual CLIOptions GetParsedArgs() const = 0;

  protected:
    /// @brief Parse Arguments from argc, argv
    virtual CLIOptions ParseArgs(int argc, char** argv) = 0;
};
}  // namespace perception

#endif  /// PERCEPTION_ARGUMENT_PARSER_I_ARGUMENT_PARSER_H_
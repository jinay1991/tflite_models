///
/// @file argument_parser.h
/// @brief Contains Argument Parser definitions
/// @copyright Copyright (c) 2020. All Rights Reserved.
///
#ifndef PERCEPTION_ARGUMENT_PARSER_ARGUMENT_PARSER_H_
#define PERCEPTION_ARGUMENT_PARSER_ARGUMENT_PARSER_H_

#include <getopt.h>
#include <unistd.h>
#include <string>
#include <vector>

#include "perception/argument_parser/cli_options.h"
#include "perception/argument_parser/i_argument_parser.h"

namespace perception
{
/// @brief Argument Parser class
class ArgumentParser : public IArgumentParser
{
  public:
    /// @brief Default Constructor
    ArgumentParser();

    /// @brief Constructor
    /// @param [in] argc - number of arguments
    /// @param [in] argv - list of arguments
    explicit ArgumentParser(int argc, char* argv[]);

    /// @brief Destructor
    virtual ~ArgumentParser();

    /// @brief Provides Parsed Arguments
    virtual CLIOptions GetParsedArgs() const override;

  protected:
    /// @brief Parse Arguments from argc, argv
    virtual CLIOptions ParseArgs(int argc, char* argv[]) override;

  private:
    /// @brief parsed arguments
    CLIOptions cli_options_;

    /// @brief long option list
    std::vector<struct option> long_options_;

    /// @brief short option list
    std::string optstring_;
};

}  // namespace perception

#endif  /// PERCEPTION_ARGUMENT_PARSER_ARGUMENT_PARSER_H_
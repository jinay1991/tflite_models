///
/// @file
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
class ArgumentParser : public IArgumentParser
{
  public:
    ArgumentParser();
    explicit ArgumentParser(int argc, char* argv[]);
    virtual ~ArgumentParser();

    virtual CLIOptions GetParsedArgs() const override;

  protected:
    virtual CLIOptions ParseArgs(int argc, char* argv[]) override;

  private:
    CLIOptions cli_options_;
    std::vector<struct option> long_options_;
    std::string optstring_;
};

}  // namespace perception

#endif  /// PERCEPTION_ARGUMENT_PARSER_ARGUMENT_PARSER_H_
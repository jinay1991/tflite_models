///
/// @file
/// @copyright Copyright (c) 2020. All Rights Reserved.
///
#include <iostream>

#include "perception/argument_parser/argument_parser.h"

#define LOG(x) std::cerr

namespace perception
{
namespace
{
void PrintUsage()
{
    LOG(INFO) << "label_image\n"
              << "--count, -c: loop interpreter->Invoke() for certain times\n"
              << "--input_mean, -b: input mean\n"
              << "--input_std, -s: input standard deviation\n"
              << "--image, -i: image_name.bmp\n"
              << "--max_profiling_buffer_entries, -e: maximum profiling buffer entries\n"
              << "--labels, -l: labels for the model\n"
              << "--tflite_model, -m: model_name.tflite\n"
              << "--profiling, -p: [0|1], profiling or not\n"
              << "--num_results, -r: number of results to show\n"
              << "--threads, -t: number of threads\n"
              << "--verbose, -v: [0|1] print more information\n"
              << "--result_directory, -d: directory path\n"
              << "\n";
}
}  // namespace

ArgumentParser::ArgumentParser(int argc, char* argv[])
    : long_options_{{"count", required_argument, nullptr, 'c'},
                    {"image", required_argument, nullptr, 'i'},
                    {"input_mean", required_argument, nullptr, 'b'},
                    {"input_std", required_argument, nullptr, 's'},
                    {"labels", required_argument, nullptr, 'l'},
                    {"max_profiling_buffer_entries", required_argument, nullptr, 'e'},
                    {"num_results", required_argument, nullptr, 'r'},
                    {"profiling", required_argument, nullptr, 'p'},
                    {"tflite_model", required_argument, nullptr, 'm'},
                    {"threads", required_argument, nullptr, 't'},
                    {"verbose", required_argument, nullptr, 'v'},
                    {"result_directory", required_argument, nullptr, 'd'},
                    {nullptr, 0, nullptr, 0}},
      optstring_{"b:c:d:e:i:l:m:p:r:s:v:t:"}
{
    cli_options_ = ParseArgs(argc, argv);
}

CLIOptions ArgumentParser::GetParsedArgs() const { return cli_options_; }

CLIOptions ArgumentParser::ParseArgs(int argc, char* argv[])
{
    while (true)
    {
        std::int32_t c = 0;
        std::int32_t optindex = 0;

        c = getopt_long(argc, argv, optstring_.c_str(), long_options_.data(), &optindex);
        /* Detect the end of the options. */
        if (c == -1)
        {
            break;
        }

        switch (c)
        {
            case 'b':
                cli_options_.input_mean = strtod(optarg, nullptr);
                break;
            case 'c':
                cli_options_.loop_count = strtol(optarg, nullptr, 10);
                break;
            case 'd':
                cli_options_.result_directory = optarg;
                break;
            case 'e':
                cli_options_.max_profiling_buffer_entries = strtol(optarg, nullptr, 10);
                break;
            case 'i':
                cli_options_.input_name = optarg;
                break;
            case 'l':
                cli_options_.labels_name = optarg;
                break;
            case 'm':
                cli_options_.model_name = optarg;
                break;
            case 'p':
                cli_options_.profiling = strtol(optarg, nullptr, 10);
                break;
            case 'r':
                cli_options_.number_of_results = strtol(optarg, nullptr, 10);
                break;
            case 's':
                cli_options_.input_std = strtod(optarg, nullptr);
                break;
            case 't':
                cli_options_.number_of_threads = strtol(optarg, nullptr, 10);
                break;
            case 'v':
                cli_options_.verbose = strtol(optarg, nullptr, 10);
                break;
            case 'h':
            case '?':
                /* getopt_long already printed an error message. */
                PrintUsage();
                exit(-1);
            default:
                exit(-1);
        }
    }
    return cli_options_;
}

}  // namespace perception
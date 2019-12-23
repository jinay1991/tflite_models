///
/// @file
/// @copyright Copyright (c) 2019. All Rights Reserved.
///
#include <getopt.h>
#include <iostream>

#include "perception/cli.h"

#define LOG(x) std::cerr

namespace perception
{
namespace
{
void PrintUsage()
{
    LOG(INFO) << "label_image\n"
              << "--accelerated, -a: [0|1], use Android NNAPI or not\n"
              << "--old_accelerated, -d: [0|1], use old Android NNAPI delegate or not\n"
              << "--allow_fp16, -f: [0|1], allow running fp32 models with fp16 or not\n"
              << "--count, -c: loop interpreter->Invoke() for certain times\n"
              << "--gl_backend, -g: use GL GPU Delegate on Android\n"
              << "--input_mean, -b: input mean\n"
              << "--input_std, -s: input standard deviation\n"
              << "--image, -i: image_name.bmp\n"
              << "--labels, -l: labels for the model\n"
              << "--tflite_model, -m: model_name.tflite\n"
              << "--profiling, -p: [0|1], profiling or not\n"
              << "--num_results, -r: number of results to show\n"
              << "--threads, -t: number of threads\n"
              << "--verbose, -v: [0|1] print more information\n"
              << "--warmup_runs, -w: number of warmup runs\n"
              << "\n";
}
}  // namespace

CLIOptions ParseCommandLineOptions(int argc, char** argv)
{
    CLIOptions cli_opts;
    while (true)
    {
        std::int32_t c = 0;
        static struct option long_options[] = {{"accelerated", required_argument, nullptr, 'a'},
                                               {"old_accelerated", required_argument, nullptr, 'd'},
                                               {"allow_fp16", required_argument, nullptr, 'f'},
                                               {"count", required_argument, nullptr, 'c'},
                                               {"verbose", required_argument, nullptr, 'v'},
                                               {"image", required_argument, nullptr, 'i'},
                                               {"labels", required_argument, nullptr, 'l'},
                                               {"tflite_model", required_argument, nullptr, 'm'},
                                               {"profiling", required_argument, nullptr, 'p'},
                                               {"threads", required_argument, nullptr, 't'},
                                               {"input_mean", required_argument, nullptr, 'b'},
                                               {"input_std", required_argument, nullptr, 's'},
                                               {"num_results", required_argument, nullptr, 'r'},
                                               {"max_profiling_buffer_entries", required_argument, nullptr, 'e'},
                                               {"warmup_runs", required_argument, nullptr, 'w'},
                                               {"gl_backend", required_argument, nullptr, 'g'},
                                               {nullptr, 0, nullptr, 0}};

        /* getopt_long stores the option index here. */
        std::int32_t option_index = 0;

        c = getopt_long(argc, argv, "a:b:c:d:e:f:g:i:l:m:p:r:s:t:v:w:", long_options, &option_index);

        /* Detect the end of the options. */
        if (c == -1)
        {
            break;
        }

        switch (c)
        {
            case 'a':
                cli_opts.accel = strtol(optarg, nullptr, 10);
                break;
            case 'b':
                cli_opts.input_mean = strtod(optarg, nullptr);
                break;
            case 'c':
                cli_opts.loop_count = strtol(optarg, nullptr, 10);
                break;
            case 'd':
                cli_opts.old_accel = strtol(optarg, nullptr, 10);
                break;
            case 'e':
                cli_opts.max_profiling_buffer_entries = strtol(optarg, nullptr, 10);
                break;
            case 'f':
                cli_opts.allow_fp16 = strtol(optarg, nullptr, 10);
                break;
            case 'g':
                cli_opts.gl_backend = strtol(optarg, nullptr, 10);
                break;
            case 'i':
                cli_opts.input_bmp_name = optarg;
                break;
            case 'l':
                cli_opts.labels_file_name = optarg;
                break;
            case 'm':
                cli_opts.model_name = optarg;
                break;
            case 'p':
                cli_opts.profiling = strtol(optarg, nullptr, 10);
                break;
            case 'r':
                cli_opts.number_of_results = strtol(optarg, nullptr, 10);
                break;
            case 's':
                cli_opts.input_std = strtod(optarg, nullptr);
                break;
            case 't':
                cli_opts.number_of_threads = strtol(optarg, nullptr, 10);
                break;
            case 'v':
                cli_opts.verbose = strtol(optarg, nullptr, 10);
                break;
            case 'w':
                cli_opts.number_of_warmup_runs = strtol(optarg, nullptr, 10);
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
    return cli_opts;
}

}  // namespace perception

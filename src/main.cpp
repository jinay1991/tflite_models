/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
#include <getopt.h>
#include <iostream>

#include "label_image/label_image.h"

#define LOG(x) std::cerr

void display_usage()
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

int main(int argc, char** argv)
{
    tflite::label_image::Settings s;

    int c;
    while (1)
    {
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
        int option_index = 0;

        c = getopt_long(argc, argv, "a:b:c:d:e:f:g:i:l:m:p:r:s:t:v:w:", long_options, &option_index);

        /* Detect the end of the options. */
        if (c == -1) break;

        switch (c)
        {
            case 'a':
                s.accel = strtol(optarg, nullptr, 10);  // NOLINT(runtime/deprecated_fn)
                break;
            case 'b':
                s.input_mean = strtod(optarg, nullptr);
                break;
            case 'c':
                s.loop_count = strtol(optarg, nullptr, 10);  // NOLINT(runtime/deprecated_fn)
                break;
            case 'd':
                s.old_accel = strtol(optarg, nullptr, 10);  // NOLINT(runtime/deprecated_fn)
                break;
            case 'e':
                s.max_profiling_buffer_entries = strtol(optarg, nullptr, 10);  // NOLINT(runtime/deprecated_fn)
                break;
            case 'f':
                s.allow_fp16 = strtol(optarg, nullptr, 10);  // NOLINT(runtime/deprecated_fn)
                break;
            case 'g':
                s.gl_backend = strtol(optarg, nullptr, 10);  // NOLINT(runtime/deprecated_fn)
                break;
            case 'i':
                s.input_bmp_name = optarg;
                break;
            case 'l':
                s.labels_file_name = optarg;
                break;
            case 'm':
                s.model_name = optarg;
                break;
            case 'p':
                s.profiling = strtol(optarg, nullptr, 10);  // NOLINT(runtime/deprecated_fn)
                break;
            case 'r':
                s.number_of_results = strtol(optarg, nullptr, 10);  // NOLINT(runtime/deprecated_fn)
                break;
            case 's':
                s.input_std = strtod(optarg, nullptr);
                break;
            case 't':
                s.number_of_threads = strtol(  // NOLINT(runtime/deprecated_fn)
                    optarg, nullptr, 10);
                break;
            case 'v':
                s.verbose = strtol(optarg, nullptr, 10);  // NOLINT(runtime/deprecated_fn)
                break;
            case 'w':
                s.number_of_warmup_runs = strtol(optarg, nullptr, 10);  // NOLINT(runtime/deprecated_fn)
                break;
            case 'h':
            case '?':
                /* getopt_long already printed an error message. */
                display_usage();
                exit(-1);
            default:
                exit(-1);
        }
    }
    RunInference(&s);
    return 0;
}
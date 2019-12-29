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

#ifndef PERCEPTION_UTILS_GET_TOP_N_H_
#define PERCEPTION_UTILS_GET_TOP_N_H_

#include <cstdint>
#include <string>
#include <vector>

#include "perception/utils/get_top_n_impl.h"

namespace perception
{
template <class T>
void get_top_n(T* prediction, std::int32_t prediction_size, std::size_t num_results, float threshold,
               std::vector<std::pair<float, std::int32_t>>* top_results, bool input_floating);

// explicit instantiation so that we can use them otherwhere
template void get_top_n<std::uint8_t>(std::uint8_t*, std::int32_t, std::size_t, float,
                                      std::vector<std::pair<float, std::int32_t>>*, bool);
template void get_top_n<float>(float*, std::int32_t, std::size_t, float, std::vector<std::pair<float, std::int32_t>>*,
                               bool);

}  // namespace perception

#endif  // PERCEPTION_UTILS_GET_TOP_N_H_

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

#ifndef PERCEPTION_UTILS_RESIZE_H_
#define PERCEPTION_UTILS_RESIZE_H_

#include <cstdint>

#include "perception/utils/resize_impl.h"

namespace perception
{
template <class T>
void ResizeImage(T* out, const std::uint8_t* in, const std::int32_t image_height, const std::int32_t image_width,
                 const std::int32_t image_channels, const std::int32_t wanted_height, const std::int32_t wanted_width,
                 const std::int32_t wanted_channels, const bool input_floating, const float input_mean,
                 const float input_std);

// explicit instantiation
template void ResizeImage<float>(float* out, const std::uint8_t* in, const std::int32_t image_height,
                                 const std::int32_t image_width, const std::int32_t image_channels,
                                 const std::int32_t wanted_height, const std::int32_t wanted_width,
                                 const std::int32_t wanted_channels, const bool input_floating, const float input_mean,
                                 const float input_std);
template void ResizeImage<std::uint8_t>(std::uint8_t* out, const std::uint8_t* in, const std::int32_t image_height,
                                        const std::int32_t image_width, const std::int32_t image_channels,
                                        const std::int32_t wanted_height, const std::int32_t wanted_width,
                                        const std::int32_t wanted_channels, const bool input_floating,
                                        const float input_mean, const float input_std);
}  // namespace perception

#endif  /// PERCEPTION_UTILS_RESIZE_H_
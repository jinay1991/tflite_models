///
/// @file
/// @copyright Copyright (c) 2019. All Rights Reserved.
///
#ifndef PERCEPTION_UTILS_RESIZE_H_
#define PERCEPTION_UTILS_RESIZE_H_

#include <cstdint>

#include "perception/utils/resize_impl.h"

namespace perception
{
template <class T>
void ResizeImage(T* out, std::uint8_t* in, const std::int32_t image_height, const std::int32_t image_width,
                 const std::int32_t image_channels, const std::int32_t wanted_height, const std::int32_t wanted_width,
                 const std::int32_t wanted_channels, const bool input_floating, const float input_mean,
                 const float input_std);

// explicit instantiation
template void ResizeImage<float>(float* out, std::uint8_t* in, const std::int32_t image_height,
                                 const std::int32_t image_width, const std::int32_t image_channels,
                                 const std::int32_t wanted_height, const std::int32_t wanted_width,
                                 const std::int32_t wanted_channels, const bool input_floating, const float input_mean,
                                 const float input_std);
template void ResizeImage<std::uint8_t>(std::uint8_t* out, std::uint8_t* in, const std::int32_t image_height,
                                        const std::int32_t image_width, const std::int32_t image_channels,
                                        const std::int32_t wanted_height, const std::int32_t wanted_width,
                                        const std::int32_t wanted_channels, const bool input_floating,
                                        const float input_mean, const float input_std);
}  // namespace perception

#endif  /// PERCEPTION_UTILS_RESIZE_H_
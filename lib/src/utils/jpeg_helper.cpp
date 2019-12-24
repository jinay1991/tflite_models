///
/// @file
/// @copyright Copyright (c) 2019. All Rights Reserved.
///
#include "perception/utils/jpeg_helper.h"

namespace perception
{
std::vector<std::uint8_t> JpegImageHelper::ReadImage(const std::string& input_bmp_name, std::int32_t* width,
                                                     std::int32_t* height, std::int32_t* channels)
{
    return std::vector<std::uint8_t>{};
}

std::vector<std::uint8_t> JpegImageHelper::DecodeImage(const std::uint8_t* input) const
{
    return std::vector<std::uint8_t>{};
}

}  // namespace perception
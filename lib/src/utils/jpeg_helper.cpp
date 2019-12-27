///
/// @file
/// @copyright Copyright (c) 2019. All Rights Reserved.
///
#include <fstream>

#include "perception/utils/jpeg_decoder.h"
#include "perception/utils/jpeg_helper.h"

namespace perception
{
std::vector<std::uint8_t> JpegImageHelper::ReadImage(const std::string& input_jpeg_name, std::int32_t* width,
                                                     std::int32_t* height, std::int32_t* channels)
{
    std::ifstream jpeg_image{input_jpeg_name};
    std::string jpeg_data((std::istreambuf_iterator<char>(jpeg_image)), std::istreambuf_iterator<char>());
    Jpeg::Decoder jpeg_decoder_{jpeg_data.c_str(), jpeg_data.size()};

    *width = jpeg_decoder_.GetWidth();
    *height = jpeg_decoder_.GetHeight();
    *channels = jpeg_decoder_.IsColor() ? 3 : 1;
    const auto data = jpeg_decoder_.GetImage();
    std::vector<uint8_t> output(jpeg_decoder_.GetImageSize());
    for (auto i = 0; i < output.size(); ++i)
    {
        output[i] = data[i];
    }
    return output;
}

std::vector<std::uint8_t> JpegImageHelper::DecodeImage(const std::uint8_t* input) const
{
    return std::vector<std::uint8_t>{};
}

}  // namespace perception
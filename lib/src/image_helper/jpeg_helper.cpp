///
/// @file
/// @copyright Copyright (c) 2020. All Rights Reserved.
///
#include <fstream>

#include "perception/image_helper/jpeg_helper.h"

namespace perception
{
JpegImageHelper::JpegImageHelper() {}
JpegImageHelper::~JpegImageHelper() {}

std::vector<std::uint8_t> JpegImageHelper::ReadImage(const std::string& input_jpeg_name, std::int32_t* width,
                                                     std::int32_t* height, std::int32_t* channels)
{
    std::ifstream jpeg_image{input_jpeg_name};
    std::string jpeg_data((std::istreambuf_iterator<char>(jpeg_image)), std::istreambuf_iterator<char>());
    jpeg_decoder_ = std::make_unique<Jpeg::Decoder>(jpeg_data.c_str(), jpeg_data.size());

    *width = jpeg_decoder_->GetWidth();
    *height = jpeg_decoder_->GetHeight();
    *channels = jpeg_decoder_->IsColor() ? 3 : 1;

    return DecodeImage(jpeg_decoder_->GetImage());
}

std::vector<std::uint8_t> JpegImageHelper::DecodeImage(const std::uint8_t* input) const
{
    std::vector<uint8_t> output(jpeg_decoder_->GetImageSize());
    for (auto i = 0; i < output.size(); ++i)
    {
        output[i] = input[i];
    }
    return output;
}

}  // namespace perception
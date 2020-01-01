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

#include <fstream>
#include <iostream>

#include "perception/image_helper/bitmap_helper.h"

namespace perception
{
BitmapImageHelper::BitmapImageHelper() : width_{224}, height_{224}, channels_{3} {}
BitmapImageHelper::~BitmapImageHelper() {}

std::vector<std::uint8_t> BitmapImageHelper::ReadImage(const std::string& image_path, std::int32_t* width,
                                                       std::int32_t* height, std::int32_t* channels)
{
    std::ifstream file(image_path, std::ios::in | std::ios::binary);
    if (!file)
    {
        throw std::runtime_error("Input file " + image_path + " not found");
    }
    auto begin = file.tellg();
    file.seekg(0, std::ios::end);
    auto end = file.tellg();
    const auto len = (end - begin);

    std::vector<std::uint8_t> img_bytes(len);
    file.seekg(0, std::ios::beg);
    file.read(reinterpret_cast<char*>(img_bytes.data()), len);

    const std::int32_t header_size = *(reinterpret_cast<const std::int32_t*>(img_bytes.data() + 10));
    width_ = *(reinterpret_cast<const std::int32_t*>(img_bytes.data() + 18));
    height_ = *(reinterpret_cast<const std::int32_t*>(img_bytes.data() + 22));
    const std::int32_t bpp = *(reinterpret_cast<const std::int32_t*>(img_bytes.data() + 28));
    channels_ = bpp / 8;

    if (!width || !height || !channels)
    {
        throw std::runtime_error("Received nullptr for width/height/channels.");
    }
    *width = width_;
    *height = height_;
    *channels = channels_;

    // Decode image, allocating tensor once the image size is known
    const uint8_t* bmp_pixels = &img_bytes[header_size];
    return DecodeImage(bmp_pixels);
}

std::vector<std::uint8_t> BitmapImageHelper::DecodeImage(const std::uint8_t* input) const
{
    // there may be padding bytes when the width is not a multiple of 4 bytes
    // 8 * channels == bits per pixel
    const std::int32_t row_size = (8 * channels_ * width_ + 31) / 32 * 4;

    // if height is negative, data layout is top down
    // otherwise, it's bottom up
    const bool top_down = (height_ < 0);

    std::vector<uint8_t> output(abs(height_) * width_ * channels_);
    for (int i = 0; i < abs(height_); i++)
    {
        int src_pos;
        int dst_pos;

        for (int j = 0; j < width_; j++)
        {
            if (!top_down)
            {
                src_pos = ((abs(height_) - 1 - i) * row_size) + j * channels_;
            }
            else
            {
                src_pos = i * row_size + j * channels_;
            }

            dst_pos = (i * width_ + j) * channels_;

            switch (channels_)
            {
                case 1:
                    output[dst_pos] = input[src_pos];
                    break;
                case 3:
                    // BGR -> RGB
                    output[dst_pos] = input[src_pos + 2];
                    output[dst_pos + 1] = input[src_pos + 1];
                    output[dst_pos + 2] = input[src_pos];
                    break;
                case 4:
                    // BGRA -> RGBA
                    output[dst_pos] = input[src_pos + 2];
                    output[dst_pos + 1] = input[src_pos + 1];
                    output[dst_pos + 2] = input[src_pos];
                    output[dst_pos + 3] = input[src_pos + 3];
                    break;
                default:
                    throw std::runtime_error("Unexpected number of channels: " + std::to_string(channels_));
            }
        }
    }
    return output;
}

}  // namespace perception

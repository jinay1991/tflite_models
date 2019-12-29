///
/// @file
/// @copyright Copyright (c) 2020. All Rights Reserved.
///
#ifndef PERCEPTION_UTILS_JPEG_HELPER_H_
#define PERCEPTION_UTILS_JPEG_HELPER_H_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "tensorflow/lite/builtin_op_data.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/string_util.h"

#include "perception/utils/i_image_helper.h"
#include "perception/utils/jpeg_decoder.h"

namespace perception
{
class JpegImageHelper : public IImageHelper
{
  public:
    JpegImageHelper() = default;
    virtual ~JpegImageHelper() = default;

    virtual std::vector<std::uint8_t> ReadImage(const std::string& input_jpeg_name, std::int32_t* width,
                                                std::int32_t* height, std::int32_t* channels) override;

  private:
    virtual std::vector<std::uint8_t> DecodeImage(const std::uint8_t* input) const override;

    std::unique_ptr<Jpeg::Decoder> jpeg_decoder_;
};
}  // namespace perception
#endif  /// PERCEPTION_UTILS_JPEG_HELPER_H_
///
/// @file
/// @copyright Copyright (c) 2020. All Rights Reserved.
///
#ifndef PERCEPTION_IMAGE_HELPER_JPEG_HELPER_H_
#define PERCEPTION_IMAGE_HELPER_JPEG_HELPER_H_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "perception/image_helper/i_image_helper.h"
#include "perception/image_helper/jpeg_decoder.h"

namespace perception
{
class JpegImageHelper : public IImageHelper
{
  public:
    JpegImageHelper();
    virtual ~JpegImageHelper();

    virtual std::vector<std::uint8_t> ReadImage(const std::string& input_jpeg_name, std::int32_t* width,
                                                std::int32_t* height, std::int32_t* channels) override;

  private:
    virtual std::vector<std::uint8_t> DecodeImage(const std::uint8_t* input) const override;

    std::unique_ptr<Jpeg::Decoder> jpeg_decoder_;
};
}  // namespace perception
#endif  /// PERCEPTION_IMAGE_HELPER_JPEG_HELPER_H_
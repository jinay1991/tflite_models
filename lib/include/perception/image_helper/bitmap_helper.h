///
/// @file
/// @copyright Copyright (c) 2020. All Rights Reserved.
///
#ifndef PERCEPTION_IMAGE_HELPER_BITMAP_HELPER_H_
#define PERCEPTION_IMAGE_HELPER_BITMAP_HELPER_H_

#include <cstdint>
#include <string>
#include <vector>

#include "perception/image_helper/i_image_helper.h"

namespace perception
{
class BitmapImageHelper : public IImageHelper
{
  public:
    BitmapImageHelper();
    virtual ~BitmapImageHelper();

    virtual std::vector<std::uint8_t> ReadImage(const std::string& input_name, std::int32_t* width,
                                                std::int32_t* height, std::int32_t* channels) override;

  private:
    virtual std::vector<std::uint8_t> DecodeImage(const std::uint8_t* input) const override;

    std::int32_t width_;
    std::int32_t height_;
    std::int32_t channels_;
};
}  // namespace perception
#endif  /// PERCEPTION_IMAGE_HELPER_BITMAP_HELPER_H_
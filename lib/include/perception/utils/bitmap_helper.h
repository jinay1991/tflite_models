///
/// @file
/// @copyright Copyright (c) 2020. All Rights Reserved.
///
#ifndef PERCEPTION_UTILS_BITMAP_HELPER_H_
#define PERCEPTION_UTILS_BITMAP_HELPER_H_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "perception/utils/i_image_helper.h"

namespace perception
{
class BitmapImageHelper : public IImageHelper
{
  public:
    BitmapImageHelper() = default;
    virtual ~BitmapImageHelper() = default;

    virtual std::vector<std::uint8_t> ReadImage(const std::string& input_name, std::int32_t* width,
                                                std::int32_t* height, std::int32_t* channels) override;

  private:
    virtual std::vector<std::uint8_t> DecodeImage(const std::uint8_t* input) const override;

    std::int32_t width_;
    std::int32_t height_;
    std::int32_t channels_;
};
}  // namespace perception
#endif  /// PERCEPTION_UTILS_BITMAP_HELPER_H_
///
/// @file
/// @copyright Copyright (c) 2020. All Rights Reserved.
///
#ifndef PERCEPTION_IMAGE_HELPER_I_IMAGE_HELPERS_H_
#define PERCEPTION_IMAGE_HELPER_I_IMAGE_HELPERS_H_

#include <cstdint>
#include <string>
#include <vector>

namespace perception
{
class IImageHelper
{
  public:
    virtual ~IImageHelper() = default;
    virtual std::vector<std::uint8_t> ReadImage(const std::string& image_path, std::int32_t* width,
                                                std::int32_t* height, std::int32_t* channels) = 0;

  private:
    virtual std::vector<std::uint8_t> DecodeImage(const std::uint8_t* input_data) const = 0;
};

}  // namespace perception

#endif  /// PERCEPTION_IMAGE_HELPER_I_IMAGE_HELPERS_H_
///
/// @file bitmap_helper.h
/// @brief Contains Bitmap Image Helper
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
    /// @brief Constructor
    BitmapImageHelper();

    /// @brief Destructor
    virtual ~BitmapImageHelper();

    /// @brief Read Bitmap (BMP) Image file.
    /// @param [in] image_path - Path to BMP Image
    /// @param [out] width - Image Width
    /// @param [out] height - Image Height
    /// @param [out] channels - Image Channels
    /// @return data - Image Data (vector<uint8_t>)
    virtual std::vector<std::uint8_t> ReadImage(const std::string& image_path, std::int32_t* width,
                                                std::int32_t* height, std::int32_t* channels) override;

  private:
    /// @brief Decodes Image Data from given data buffer
    virtual std::vector<std::uint8_t> DecodeImage(const std::uint8_t* input) const override;

    /// @brief Image Width
    std::int32_t width_;

    /// @brief Image Height
    std::int32_t height_;

    /// @brief Image Channels
    std::int32_t channels_;
};
}  // namespace perception
#endif  /// PERCEPTION_IMAGE_HELPER_BITMAP_HELPER_H_
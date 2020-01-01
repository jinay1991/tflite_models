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
    /// @brief Constructor
    JpegImageHelper();

    /// @brief Destructor
    virtual ~JpegImageHelper();

    /// @brief Read JPG Image file.
    /// @param [in] image_path - Path to JPG Image
    /// @param [out] width - Image Width
    /// @param [out] height - Image Height
    /// @param [out] channels - Image Channels
    /// @return data - Image Data (vector<uint8_t>)
    virtual std::vector<std::uint8_t> ReadImage(const std::string& image_path, std::int32_t* width,
                                                std::int32_t* height, std::int32_t* channels) override;

  private:
    /// @brief Decode Image Data from provided image buffer
    virtual std::vector<std::uint8_t> DecodeImage(const std::uint8_t* input) const override;

    std::unique_ptr<Jpeg::Decoder> jpeg_decoder_;
};
}  // namespace perception
#endif  /// PERCEPTION_IMAGE_HELPER_JPEG_HELPER_H_
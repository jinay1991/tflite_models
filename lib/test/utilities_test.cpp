///
/// @file
/// @copyright Copyright (c) 2020. All Rights Reserved.
///
#include <gtest/gtest.h>
#include <cstdint>
#include <memory>
#include <vector>

#include "perception/utils/bitmap_helper.h"
#include "perception/utils/get_top_n.h"
#include "perception/utils/i_image_helper.h"
#include "perception/utils/resize.h"

namespace perception
{
class UtilitiesTestFixture : public ::testing::Test
{
  public:
    UtilitiesTestFixture()
        : image_helper_{std::make_unique<BitmapImageHelper>()},
          test_image_path_{"data/grace_hopper.bmp"},
          width_{224},
          height_{224},
          channels_{3},
          test_image_height_{606},
          test_image_width_{517},
          test_image_channels_{3}
    {
    }
    virtual ~UtilitiesTestFixture() = default;

  protected:
    std::unique_ptr<IImageHelper> image_helper_;
    std::string test_image_path_;
    std::int32_t width_;
    std::int32_t height_;
    std::int32_t channels_;
    const std::int32_t test_image_height_;
    const std::int32_t test_image_width_;
    const std::int32_t test_image_channels_;
};

TEST_F(UtilitiesTestFixture, GivenBitmapImagePath_WhenReadImage_ExpectDecodedImageBuffer)
{
    auto in = image_helper_->ReadImage(test_image_path_, &width_, &height_, &channels_);

    EXPECT_GT(in.size(), 0);
    ASSERT_EQ(height_, test_image_height_);
    ASSERT_EQ(width_, test_image_width_);
    ASSERT_EQ(channels_, test_image_channels_);
}

TEST_F(UtilitiesTestFixture, GivenInvalidImagePath_WhenReadImage_ExpectException)
{
    std::string invalid_image_path{"invalid_file"};
    EXPECT_THROW(image_helper_->ReadImage(invalid_image_path, &width_, &height_, &channels_), std::runtime_error);
}

TEST_F(UtilitiesTestFixture, GivenNullImageAttributes_WhenReadImage_ExpectException)
{
    EXPECT_THROW(image_helper_->ReadImage(test_image_path_, nullptr, nullptr, nullptr), std::runtime_error);
}

TEST_F(UtilitiesTestFixture, GivenImageBuffer_WhenResizeImage_ExpectResizedImageBuffer)
{
    const std::int32_t wanted_height{214};
    const std::int32_t wanted_width{214};
    const std::int32_t wanted_channels{3};
    const bool input_floating{false};
    const float input_mean = 127.5f;
    const float input_std = 127.5f;

    auto input = image_helper_->ReadImage(test_image_path_, &width_, &height_, &channels_);
    ASSERT_EQ(height_, test_image_height_);
    ASSERT_EQ(width_, test_image_width_);
    ASSERT_EQ(channels_, test_image_channels_);

    std::vector<std::uint8_t> output(test_image_height_ * test_image_width_ * test_image_channels_);
    ResizeImage<std::uint8_t>(output.data(), input.data(), test_image_height_, test_image_width_, test_image_channels_,
                              wanted_height, wanted_width, wanted_channels, input_floating, input_mean, input_std);

    ASSERT_EQ(output[0], 0x15);
    ASSERT_EQ(output[wanted_height * wanted_width * wanted_channels - 1], 0x11);
}

TEST_F(UtilitiesTestFixture, GetTopN)
{
    std::vector<std::uint8_t> in{1, 1, 2, 2, 4, 4, 16, 32, 128, 64};

    std::vector<std::pair<float, std::int32_t>> top_results;
    get_top_n<std::uint8_t>(in.data(), 10, 5, 0.025, &top_results, false);
    ASSERT_EQ(top_results.size(), 4);
    ASSERT_EQ(top_results[0].second, 8);
}

}  // namespace perception

///
/// @file utilities.cpp
/// @brief Contains unit tests for utility functions
/// @copyright Copyright (c) 2020. All Rights Reserved.
///
#include <gtest/gtest.h>
#include <cstdint>
#include <memory>
#include <vector>

#include "perception/image_helper/bitmap_helper.h"
#include "perception/image_helper/i_image_helper.h"
#include "perception/utils/get_top_n.h"

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

TEST_F(UtilitiesTestFixture, GetTopN)
{
    std::vector<std::uint8_t> in{1, 1, 2, 2, 4, 4, 16, 32, 128, 64};

    std::vector<std::pair<float, std::int32_t>> top_results;
    get_top_n<std::uint8_t>(in.data(), 10, 5, 0.025, &top_results, false);
    ASSERT_EQ(top_results.size(), 4);
    ASSERT_EQ(top_results[0].second, 8);
}

}  // namespace perception

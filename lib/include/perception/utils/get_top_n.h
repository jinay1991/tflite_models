///
/// @file
/// @copyright Copyright (c) 2019. All Rights Reserved.
///
#ifndef PERCEPTION_GET_TOP_N_H_
#define PERCEPTION_GET_TOP_N_H_

#include <cstdint>
#include <string>
#include <vector>

#include "perception/utils/get_top_n_impl.h"

namespace perception
{
template <class T>
void get_top_n(T* prediction, std::int32_t prediction_size, std::size_t num_results, float threshold,
               std::vector<std::pair<float, std::int32_t>>* top_results, bool input_floating);

// explicit instantiation so that we can use them otherwhere
template void get_top_n<std::uint8_t>(std::uint8_t*, std::int32_t, std::size_t, float,
                                      std::vector<std::pair<float, std::int32_t>>*, bool);
template void get_top_n<float>(float*, std::int32_t, std::size_t, float, std::vector<std::pair<float, std::int32_t>>*,
                               bool);

}  // namespace perception

#endif  // PERCEPTION_GET_TOP_N_H_

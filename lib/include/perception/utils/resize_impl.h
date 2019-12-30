/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef PERCEPTION_UTILS_RESIZE_IMPL_H_
#define PERCEPTION_UTILS_RESIZE_IMPL_H_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "tensorflow/lite/builtin_op_data.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"

namespace perception
{
template <class T>
void ResizeImage(T* out, const std::uint8_t* in, const std::int32_t image_height, const std::int32_t image_width,
                 const std::int32_t image_channels, const std::int32_t wanted_height, const std::int32_t wanted_width,
                 const std::int32_t wanted_channels, const bool input_floating, const float input_mean,
                 const float input_std)
{
    std::int32_t number_of_pixels = image_height * image_width * image_channels;
    std::unique_ptr<tflite::Interpreter> interpreter = std::make_unique<tflite::Interpreter>();

    std::int32_t base_index = 0;

    // two inputs: input and new_sizes
    interpreter->AddTensors(2, &base_index);
    // one output
    interpreter->AddTensors(1, &base_index);
    // set input and output tensors
    interpreter->SetInputs({0, 1});
    interpreter->SetOutputs({2});

    // set parameters of tensors
    TfLiteQuantizationParams quant;
    interpreter->SetTensorParametersReadWrite(0, kTfLiteFloat32, "input",
                                              {1, image_height, image_width, image_channels}, quant);
    interpreter->SetTensorParametersReadWrite(1, kTfLiteInt32, "new_size", {2}, quant);
    interpreter->SetTensorParametersReadWrite(2, kTfLiteFloat32, "output",
                                              {1, wanted_height, wanted_width, wanted_channels}, quant);

    tflite::ops::builtin::BuiltinOpResolver resolver;
    const TfLiteRegistration* resize_op = resolver.FindOp(tflite::BuiltinOperator_RESIZE_BILINEAR, 1);
    auto* params = reinterpret_cast<TfLiteResizeBilinearParams*>(malloc(sizeof(TfLiteResizeBilinearParams)));
    params->align_corners = false;
    interpreter->AddNodeWithParameters({0, 1}, {2}, nullptr, 0, params, resize_op, nullptr);

    interpreter->AllocateTensors();

    // fill input image
    // in[] are integers, cannot do memcpy() directly
    auto input = interpreter->typed_tensor<float>(0);
    for (std::int32_t i = 0; i < number_of_pixels; i++)
    {
        input[i] = in[i];
    }

    // fill new_sizes
    interpreter->typed_tensor<std::int32_t>(1)[0] = wanted_height;
    interpreter->typed_tensor<std::int32_t>(1)[1] = wanted_width;

    interpreter->Invoke();

    auto output = interpreter->typed_tensor<float>(2);
    auto output_number_of_pixels = wanted_height * wanted_width * wanted_channels;

    for (std::int32_t i = 0; i < output_number_of_pixels; i++)
    {
        if (input_floating)
        {
            out[i] = (output[i] - input_mean) / input_std;
        }
        else
        {
            out[i] = static_cast<std::uint8_t>(output[i]);
        }
    }
}

}  // namespace perception

#endif  /// PERCEPTION_UTILS_RESIZE_IMPL_H_
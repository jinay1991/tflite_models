///
/// @file
///
#include <experimental/filesystem>
#include <iostream>

#include "tensorflow/lite/interpreter.h"

int main(int argc, char** argv)
{
    if (std::experimental::filesystem::exists("external/mobilenet_v2_1.0_224_quant/mobilenet_v2_1.0_224_quant.tflite"))
    {
        std::cout << "found model" << std::endl;
    }
    else
    {
        std::cout << "not found model" << std::endl;
    }

    return 0;
}
///
/// @file
/// @copyright Copyright (c) 2019. All Rights Reserved.
///
#include <iostream>

#include "perception/cli.h"
#include "perception/perception.h"

int main(int argc, char** argv)
{
    try
    {
        auto cli_opts = perception::ParseCommandLineOptions(argc, argv);
        auto perception_app = std::make_unique<perception::Perception>(cli_opts);
        perception_app->SetInferenceType(perception::Perception::InferenceType::kClassification);

        perception_app->Init();
        perception_app->RunInference(cli_opts.input_bmp_name);
    }
    catch (std::exception& e)
    {
        std::cerr << "Caught Exception!! " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
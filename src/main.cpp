///
/// @file
/// @copyright Copyright (c) 2020. All Rights Reserved.
///
#include <iostream>
#include <memory>

#include "perception/argument_parser/argument_parser.h"
#include "perception/argument_parser/i_argument_parser.h"
#include "perception/perception.h"

int main(int argc, char** argv)
{
    try
    {
        std::unique_ptr<perception::IArgumentParser> argument_parser =
            std::make_unique<perception::ArgumentParser>(argc, argv);
        auto perception = std::make_unique<perception::Perception>(std::move(argument_parser));
        perception->SelectInferenceEngine(perception::Perception::InferenceEngineType::kTFLiteInferenceEngine);

        perception->Init();
        perception->Execute();
        perception->Shutdown();
    }
    catch (std::exception& e)
    {
        std::cerr << "Caught Exception!! " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
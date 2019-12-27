///
/// @file
/// @copyright Copyright (c) 2019. All Rights Reserved.
///
#ifndef PERCEPTION_INFERENCE_H_
#define PERCEPTION_INFERENCE_H_

#include <cstdint>
#include <memory>
#include <string>

#include "tensorflow/lite/core/api/profiler.h"
#include "tensorflow/lite/delegates/nnapi/nnapi_delegate.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/optional_debug_tools.h"
#include "tensorflow/lite/string_util.h"
#include "tensorflow/lite/tools/evaluation/utils.h"

#define TFLITE_PROFILING_ENABLED
#include "tensorflow/lite/profiling/profiler.h"

#include "perception/cli.h"
#include "perception/i_inference_engine.h"
#include "perception/utils/get_top_n.h"
#include "perception/utils/i_image_helper.h"

namespace perception
{
class InferenceEngine : public IInferenceEngine
{
  public:
    explicit InferenceEngine(const CLIOptions& cli_opts);

    virtual ~InferenceEngine() = default;

    virtual void Init() override;

    virtual void ExecuteStep(const std::string& image_path) override;

    virtual void Shutdown() override;

  private:
    void SetInputData(std::vector<std::uint8_t> in);
    std::vector<std::pair<float, std::int32_t>> GetResults() const;
    void PrintOutput(std::vector<std::pair<float, std::int32_t>> top_results);
    void PrintProfilingInfo(const tflite::profiling::ProfileEvent* e, const std::uint32_t op_index,
                            const TfLiteNode& node, const TfLiteRegistration& registration);
    void ReadLabelsFile(const std::string& file_name, std::vector<std::string>* result, std::size_t* found_label_count);
    void SaveIntermediateResults();

    CLIOptions cli_opts_;

    std::string model_path_;
    std::string label_path_;
    bool verbose_;

    std::unique_ptr<tflite::FlatBufferModel> model_;
    std::unique_ptr<tflite::Interpreter> interpreter_;

    std::int32_t image_width_;
    std::int32_t image_height_;
    std::int32_t image_channels_;

    std::unique_ptr<IImageHelper> image_helper_;
};

}  // namespace perception
#endif  /// PERCEPTION_INFERENCE_H_
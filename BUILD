# Description:
# TensorFlow Lite Example Label Image.

load("@org_tensorflow//tensorflow/lite:build_def.bzl", "tflite_linkopts")

package(default_visibility = ["//visibility:public"])

exports_files(glob([
    "data/*.bmp",
]))

cc_library(
    name = "label_image_helpers",
    srcs = [
        "include/label_image/get_top_n.h",
        "include/label_image/get_top_n_impl.h",
        "src/bitmap_helpers.cc",
        "src/label_image.cc",
    ],
    hdrs = [
        "include/label_image/bitmap_helpers.h",
        "include/label_image/bitmap_helpers_impl.h",
        "include/label_image/get_top_n.h",
        "include/label_image/get_top_n_impl.h",
        "include/label_image/label_image.h",
    ],
    strip_include_prefix = "include",
    deps = [
        "@com_google_absl//absl/memory",
        "@org_tensorflow//tensorflow/lite:builtin_op_data",
        "@org_tensorflow//tensorflow/lite:framework",
        "@org_tensorflow//tensorflow/lite:string",
        "@org_tensorflow//tensorflow/lite:string_util",
        "@org_tensorflow//tensorflow/lite/delegates/nnapi:nnapi_delegate",
        "@org_tensorflow//tensorflow/lite/kernels:builtin_ops",
        "@org_tensorflow//tensorflow/lite/profiling:profiler",
        "@org_tensorflow//tensorflow/lite/schema:schema_fbs",
        "@org_tensorflow//tensorflow/lite/tools/evaluation:utils",
    ],
)

cc_binary(
    name = "label_image",
    srcs = [
        "src/main.cpp",
    ],
    data = [
        "data/grace_hopper.bmp",
        "data/labels.txt",
        "@mobilenet_v2_1.0_224_quant//:tflite",
    ],
    deps = [
        ":label_image_helpers",
    ],
)

cc_test(
    name = "label_image_test",
    srcs = [
        "test/label_image_test.cc",
    ],
    data = [
        "data/grace_hopper.bmp",
    ],
    deps = [
        ":label_image_helpers",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

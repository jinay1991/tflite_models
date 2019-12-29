package(default_visibility = ["//visibility:public"])

config_setting(
    name = "darwin",
    constraint_values = ["@bazel_tools//platforms:osx"],
)

cc_library(
    name = "utils",
    srcs = glob(["src/utils/*.cpp"]),
    hdrs = glob(["include/perception/utils/*.h"]),
    linkopts = select({
        ":darwin": [],
        "//conditions:default": ["-lstdc++fs"],
    }),
    strip_include_prefix = "include",
    deps = [
        # TODO: Remove dependencies from TensorFlow
        "@org_tensorflow//tensorflow/lite:framework",
        "@org_tensorflow//tensorflow/lite/kernels:builtin_ops",
    ],
)

cc_library(
    name = "argument_parser",
    srcs = glob(["src/argument_parser/*.cpp"]),
    hdrs = glob(["include/perception/argument_parser/*.h"]),
    strip_include_prefix = "include",
)

cc_library(
    name = "inference_engine",
    srcs = glob(["src/inference_engine/*.cpp"]),
    hdrs = glob(["include/perception/inference_engine/*.h"]),
    linkopts = select({
        ":darwin": [],
        "//conditions:default": ["-lstdc++fs"],
    }),
    strip_include_prefix = "include",
    deps = [
        ":argument_parser",
        ":utils",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@org_tensorflow//tensorflow/lite:builtin_op_data",
        "@org_tensorflow//tensorflow/lite:framework",
        "@org_tensorflow//tensorflow/lite:string",
        "@org_tensorflow//tensorflow/lite:string_util",
        "@org_tensorflow//tensorflow/lite/kernels:builtin_ops",
        "@org_tensorflow//tensorflow/lite/profiling:profiler",
        "@org_tensorflow//tensorflow/lite/schema:schema_fbs",
        "@org_tensorflow//tensorflow/lite/tools/evaluation:utils",
    ],
)

cc_library(
    name = "perception",
    srcs = ["src/perception.cpp"],
    hdrs = ["include/perception/perception.h"],
    strip_include_prefix = "include",
    deps = [
        ":argument_parser",
        ":inference_engine",
    ],
)

cc_test(
    name = "perception_tests",
    srcs = glob(["test/*.cpp"]),
    data = [
        "//:testdata",
        "@mobilenet_v2_1.0_224_quant//:tflite",
    ],
    deps = [
        ":perception",
        "@com_google_googletest//:gtest_main",
    ],
)

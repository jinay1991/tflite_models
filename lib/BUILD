package(default_visibility = ["//visibility:public"])

cc_library(
    name = "jpeg_decoder",
    srcs = ["include/perception/image_helper/jpeg_decoder.h"],
    hdrs = ["include/perception/image_helper/jpeg_decoder.h"],
    copts = [
        "-Wno-error=deprecated-register",
    ],
    licenses = ["notice"],
    linkstatic = True,
    strip_include_prefix = "include",
)

cc_library(
    name = "logging",
    srcs = glob(["src/logging/*.cpp"]),
    hdrs = glob(["include/perception/logging/*.h"]),
    strip_include_prefix = "include",
)

cc_library(
    name = "image_helpers",
    srcs = glob(["src/image_helper/*.cpp"]),
    hdrs = glob(
        ["include/perception/image_helper/*.h"],
        exclude = ["include/perception/image_helper/jpeg_decoder.h"],
    ),
    copts = [
        "-Wall",
        "-Werror",
    ],
    strip_include_prefix = "include",
    deps = [
        ":jpeg_decoder",
        ":logging",
    ],
)

cc_library(
    name = "utils",
    srcs = glob(["src/utils/*.cpp"]),
    hdrs = glob(
        ["include/perception/utils/*.h"],
        exclude = ["include/perception/utils/jpeg_decoder.h"],
    ),
    copts = [
        "-Wall",
        "-Werror",
    ],
    linkopts = select({
        "//bazel/platforms:macos": [],
        "//conditions:default": ["-lstdc++fs"],
    }),
    strip_include_prefix = "include",
)

cc_library(
    name = "argument_parser",
    srcs = glob(["src/argument_parser/*.cpp"]),
    hdrs = glob(["include/perception/argument_parser/*.h"]),
    copts = [
        "-Wall",
        "-Werror",
    ],
    strip_include_prefix = "include",
    deps = [
        ":logging",
    ],
)

cc_library(
    name = "inference_engine",
    srcs = glob(["src/inference_engine/*.cpp"]),
    hdrs = glob(["include/perception/inference_engine/*.h"]),
    copts = [
        "-Wall",
        "-Werror",
    ],
    linkopts = select({
        "//bazel/platforms:macos": [],
        "//conditions:default": ["-lstdc++fs"],
    }),
    strip_include_prefix = "include",
    deps = [
        ":argument_parser",
        ":image_helpers",
        ":logging",
        ":utils",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@org_tensorflow//tensorflow/lite:builtin_op_data",
        "@org_tensorflow//tensorflow/lite:framework",
        "@org_tensorflow//tensorflow/lite:string",
        "@org_tensorflow//tensorflow/lite:string_util",
        "@org_tensorflow//tensorflow/lite/kernels:builtin_ops",
        "@org_tensorflow//tensorflow/lite/profiling:profile_summarizer",
        "@org_tensorflow//tensorflow/lite/profiling:profiler",
        "@org_tensorflow//tensorflow/lite/schema:schema_fbs",
        "@org_tensorflow//tensorflow/lite/tools/evaluation:utils",
    ],
)

cc_library(
    name = "perception",
    srcs = ["src/perception.cpp"],
    hdrs = ["include/perception/perception.h"],
    copts = [
        "-Wall",
        "-Werror",
    ],
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
        "@googletest//:gtest_main",
    ],
)

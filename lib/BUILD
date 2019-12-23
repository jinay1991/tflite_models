package(default_visibility = ["//visibility:public"])

config_setting(
    name = "darwin",
    constraint_values = ["@bazel_tools//platforms:osx"],
)

cc_library(
    name = "utils",
    srcs = [
        "src/utils/bitmap_helper.cpp",
        "src/utils/jpeg_helper.cpp",
    ],
    hdrs = [
        "include/perception/utils/bitmap_helper.h",
        "include/perception/utils/get_top_n.h",
        "include/perception/utils/get_top_n_impl.h",
        "include/perception/utils/i_image_helper.h",
        "include/perception/utils/jpeg_helper.h",
        "include/perception/utils/resize.h",
        "include/perception/utils/resize_impl.h",
    ],
    linkopts = select({
        ":darwin": [],
        "//conditions:default": ["-lstdc++fs"],
    }),
    strip_include_prefix = "include",
    deps = [
        "@org_tensorflow//tensorflow/lite:framework",
        "@org_tensorflow//tensorflow/lite/kernels:builtin_ops",
    ],
)

cc_library(
    name = "cli",
    srcs = ["src/cli.cpp"],
    hdrs = ["include/perception/cli.h"],
    strip_include_prefix = "include",
    deps = [
        "@org_tensorflow//tensorflow/lite:framework",
    ],
)

cc_library(
    name = "inference_engine",
    srcs = ["src/inference_engine.cpp"],
    hdrs = [
        "include/perception/i_inference_engine.h",
        "include/perception/inference_engine.h",
    ],
    linkopts = select({
        ":darwin": [],
        "//conditions:default": ["-lstdc++fs"],
    }),
    strip_include_prefix = "include",
    deps = [
        ":cli",
        ":utils",
        "@com_google_absl//absl/memory",
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
        ":inference_engine",
    ],
)

cc_test(
    name = "perception_test",
    srcs = glob([
        "test/*.cpp",
    ]),
    data = [
        "//:testdata",
    ],
    deps = [
        ":perception",
        "@com_google_googletest//:gtest_main",
    ],
)

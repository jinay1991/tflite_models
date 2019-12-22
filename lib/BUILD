cc_library(
    name = "interpreter",
    srcs = glob(["src/*.cpp"]),
    hdrs = glob(["include/**/*.h"]),
    copts = [
        "-std=c++14",
        "-Wall",
        "-Werror",
    ],
    strip_include_prefix = "include",
    visibility = ["//visibility:public"],
    deps = [
        "@org_tensorflow//tensorflow/lite:framework",
    ],
)

cc_test(
    name = "interpreter_tests",
    srcs = glob([
        "test/src/*.cpp",
        "test/include/**/*.h",
    ]),
    includes = ["test/include"],
    visibility = ["//visibility:public"],
    deps = [
        ":interpreter",
        "@googletest//:gtest_main",
    ],
)

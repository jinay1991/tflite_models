cc_binary(
    name = "tflite-app",
    srcs = ["src/main.cpp"],
    copts = [
        "-std=c++14",
        "-Wall",
        "-Werror",
    ],
    data = [
        "@mobilenet_v2_1.0_224_quant//:tflite",
    ],
    includes = ["include"],
    linkstatic = True,
    visibility = [
        "//visibility:public",
    ],
    # deps = [
    #     "@tensorflow//tensorflow/lite:framework",
    # ],
)

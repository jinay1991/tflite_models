package(default_visibility = ["//visibility:public"])

exports_files(glob([
    "data/*.bmp",
    "data/*.jpg",
    "data/*.txt",
]))

filegroup(
    name = "testdata",
    srcs = [
        "data/grace_hopper.bmp",
        "data/grace_hopper.jpg",
        "data/labels.txt",
    ],
)

cc_binary(
    name = "label_image",
    srcs = [
        "src/main.cpp",
    ],
    data = [
        ":testdata",
        "@mobilenet_v2_1.0_224_quant//:tflite",
    ],
    deps = [
        "//lib:perception",
    ],
)

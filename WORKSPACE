load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository", "new_git_repository")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

new_git_repository(
    name = "googletest",
    build_file = "//bazel:gtest.BUILD",
    remote = "https://github.com/google/googletest",
    tag = "release-1.8.1",
)

http_archive(
    name = "mobilenet_v2_1.0_224_quant",
    build_file = "//bazel:mobilenet_v2_1.0_224_quant.BUILD",
    sha256 = "d6a04d780f76f656c902413be432eb349ec4a458240e3739119eb44977f77a79",
    url = "https://storage.googleapis.com/download.tensorflow.org/models/tflite_11_05_08/mobilenet_v2_1.0_224_quant.tgz",
)

git_repository(
    name = "tensorflow",
    remote = "https://github.com/tensorflow/tensorflow",
    tag = "v2.0.0",
)

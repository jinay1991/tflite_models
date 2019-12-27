# TensorFlow Lite C++ image classification

This repository contains source code demonstrating how to load a pre-trained and converted
TensorFlow Lite model and use it to recognize objects in images.

You also need to [install Bazel 26.1](https://docs.bazel.build/versions/master/install.html)
in order to build this example code. And be sure you have the Python `future`
module installed:

```
pip install future --user
```

Note that, this repository uses `tensorflow` as bazel external dependency.


## Build 

Build it for desktop machines (tested on Ubuntu and OS X):

```
bazel build -c opt --cxxopt="-std=c++14" //:label_image
```

## Tests

To run unit tests first pull LFS filesystems

```
git lfs pull
```

Run unit tests

```
bazel test -c opt --cxxopt="-std=c++14" //...
```

## Download sample model and image

You can use any compatible model, but the following MobileNet v1 model offers
a good demonstration of a model trained to recognize 1,000 different objects.

```
# Get model
curl https://storage.googleapis.com/download.tensorflow.org/models/tflite_11_05_08/mobilenet_v2_1.0_224_quant.tgz | tar xzv -C /tmp
```

## Run the sample

```
bazel-bin/label_image \
  --tflite_model /tmp/mobilenet_v2_1.0_224_quant.tflite \
  --labels data/labels.txt \
  --image data/grace_hopper.bmp
```

You should see results like this:

```
Loaded model /tmp/mobilenet_v2_1.0_224_quant.tflite
resolved reporter
invoked
average time: 68.12 ms
0.860174: 653 653:military uniform
0.0481017: 907 907:Windsor tie
0.00786704: 466 466:bulletproof vest
0.00644932: 514 514:cornet, horn, trumpet, trump
0.00608029: 543 543:drumstick
```

See the `lib/src/cli.cpp` source code for other command line options.


## On CentOS 7 or later

Due to `libstdc++` dependencies `bazel` sometimes doesn't link to appropriate library hence it is advisiable to export variables as given below: 

```
export BAZEL_LINKOPTS=-static-libstdc++
export BAZEL_LINKLIBS=-l%:libstdc++.a
```

Use `devtoolset-8` for compiling source with `bazel`. (Refer: `docker/centos/Dockerfile` for more details on dependencies on CentOS 7 or refer below section to configure manually.)

### Prepare CentOS 7 Environment

Install dependencies...

```
sudo yum install -y centos-release-scl
sudo yum install -y devtoolset-8
sudo yum install -y git wget 
sudo yum install -y libtool clang-format-6.0
sudo yum install -y epel-release
sudo yum install -y python python-dev python-pip

sudo python -m pip install -U pip 
sudo python -m pip install -U future
sudo python -m pip install -U tensorflow 

wget https://copr.fedorainfracloud.org/coprs/vbatts/bazel/repo/epel-7/vbatts-bazel-epel-7.repo && sudo mv vbatts-bazel-epel-7.repo /etc/yum.repos.d
sudo yum install -y bazel
```

Before running/compiling source, run following commands

```
scl enable devtoolset-8 bash

export BAZEL_LINKOPTS=-static-libstdc++
export BAZEL_LINKLIBS=-l%:libstdc++.a
```
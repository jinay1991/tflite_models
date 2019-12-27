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

## Profiling

To enable profiling, provide command line arg `-p 1` or `--profiling 1`. 
As a result, following will be received...

```
$ bazel run //:label_image -- -p 1
Loaded model external/mobilenet_v2_1.0_224_quant/mobilenet_v2_1.0_224_quant.tflite
Resolved reporter
INFO: Initialized TensorFlow Lite runtime.
invoked 
average time: 15.888 ms 
 time (ms), Node xxx, OpCode xxx,         symblic name,              dimension,   type
     1.390, Node   0, OpCode   3,              CONV_2D, [    1  112  112   32],  uint8
     0.608, Node   1, OpCode   4,    DEPTHWISE_CONV_2D, [    1  112  112   32],  uint8
     0.257, Node   2, OpCode   3,              CONV_2D, [    1  112  112   16],  uint8
     2.223, Node   3, OpCode   3,              CONV_2D, [    1  112  112   96],  uint8
     0.475, Node   4, OpCode   4,    DEPTHWISE_CONV_2D, [    1   56   56   96],  uint8
     0.168, Node   5, OpCode   3,              CONV_2D, [    1   56   56   24],  uint8
     0.505, Node   6, OpCode   3,              CONV_2D, [    1   56   56  144],  uint8
     0.527, Node   7, OpCode   4,    DEPTHWISE_CONV_2D, [    1   56   56  144],  uint8
     0.210, Node   8, OpCode   3,              CONV_2D, [    1   56   56   24],  uint8
     0.848, Node   9, OpCode   0,                  ADD, [    1   56   56   24],  uint8
     0.560, Node  10, OpCode   3,              CONV_2D, [    1   56   56  144],  uint8
     0.136, Node  11, OpCode   4,    DEPTHWISE_CONV_2D, [    1   28   28  144],  uint8
     0.073, Node  12, OpCode   3,              CONV_2D, [    1   28   28   32],  uint8
     0.188, Node  13, OpCode   3,              CONV_2D, [    1   28   28  192],  uint8
     0.178, Node  14, OpCode   4,    DEPTHWISE_CONV_2D, [    1   28   28  192],  uint8
     0.110, Node  15, OpCode   3,              CONV_2D, [    1   28   28   32],  uint8
     0.372, Node  16, OpCode   0,                  ADD, [    1   28   28   32],  uint8
     0.339, Node  17, OpCode   3,              CONV_2D, [    1   28   28  192],  uint8
     0.263, Node  18, OpCode   4,    DEPTHWISE_CONV_2D, [    1   28   28  192],  uint8
     0.090, Node  19, OpCode   3,              CONV_2D, [    1   28   28   32],  uint8
     0.277, Node  20, OpCode   0,                  ADD, [    1   28   28   32],  uint8
     0.191, Node  21, OpCode   3,              CONV_2D, [    1   28   28  192],  uint8
     0.052, Node  22, OpCode   4,    DEPTHWISE_CONV_2D, [    1   14   14  192],  uint8
     0.072, Node  23, OpCode   3,              CONV_2D, [    1   14   14   64],  uint8
     0.131, Node  24, OpCode   3,              CONV_2D, [    1   14   14  384],  uint8
     0.102, Node  25, OpCode   4,    DEPTHWISE_CONV_2D, [    1   14   14  384],  uint8
     0.108, Node  26, OpCode   3,              CONV_2D, [    1   14   14   64],  uint8
     0.139, Node  27, OpCode   0,                  ADD, [    1   14   14   64],  uint8
     0.115, Node  28, OpCode   3,              CONV_2D, [    1   14   14  384],  uint8
     0.101, Node  29, OpCode   4,    DEPTHWISE_CONV_2D, [    1   14   14  384],  uint8
     0.100, Node  30, OpCode   3,              CONV_2D, [    1   14   14   64],  uint8
     0.138, Node  31, OpCode   0,                  ADD, [    1   14   14   64],  uint8
     0.115, Node  32, OpCode   3,              CONV_2D, [    1   14   14  384],  uint8
     0.102, Node  33, OpCode   4,    DEPTHWISE_CONV_2D, [    1   14   14  384],  uint8
     0.097, Node  34, OpCode   3,              CONV_2D, [    1   14   14   64],  uint8
     0.138, Node  35, OpCode   0,                  ADD, [    1   14   14   64],  uint8
     0.115, Node  36, OpCode   3,              CONV_2D, [    1   14   14  384],  uint8
     0.101, Node  37, OpCode   4,    DEPTHWISE_CONV_2D, [    1   14   14  384],  uint8
     0.168, Node  38, OpCode   3,              CONV_2D, [    1   14   14   96],  uint8
     0.215, Node  39, OpCode   3,              CONV_2D, [    1   14   14  576],  uint8
     0.150, Node  40, OpCode   4,    DEPTHWISE_CONV_2D, [    1   14   14  576],  uint8
     0.248, Node  41, OpCode   3,              CONV_2D, [    1   14   14   96],  uint8
     0.207, Node  42, OpCode   0,                  ADD, [    1   14   14   96],  uint8
     0.217, Node  43, OpCode   3,              CONV_2D, [    1   14   14  576],  uint8
     0.153, Node  44, OpCode   4,    DEPTHWISE_CONV_2D, [    1   14   14  576],  uint8
     0.206, Node  45, OpCode   3,              CONV_2D, [    1   14   14   96],  uint8
     0.209, Node  46, OpCode   0,                  ADD, [    1   14   14   96],  uint8
     0.214, Node  47, OpCode   3,              CONV_2D, [    1   14   14  576],  uint8
     0.041, Node  48, OpCode   4,    DEPTHWISE_CONV_2D, [    1    7    7  576],  uint8
     0.099, Node  49, OpCode   3,              CONV_2D, [    1    7    7  160],  uint8
     0.155, Node  50, OpCode   3,              CONV_2D, [    1    7    7  960],  uint8
     0.064, Node  51, OpCode   4,    DEPTHWISE_CONV_2D, [    1    7    7  960],  uint8
     0.154, Node  52, OpCode   3,              CONV_2D, [    1    7    7  160],  uint8
     0.086, Node  53, OpCode   0,                  ADD, [    1    7    7  160],  uint8
     0.156, Node  54, OpCode   3,              CONV_2D, [    1    7    7  960],  uint8
     0.068, Node  55, OpCode   4,    DEPTHWISE_CONV_2D, [    1    7    7  960],  uint8
     0.157, Node  56, OpCode   3,              CONV_2D, [    1    7    7  160],  uint8
     0.086, Node  57, OpCode   0,                  ADD, [    1    7    7  160],  uint8
     0.155, Node  58, OpCode   3,              CONV_2D, [    1    7    7  960],  uint8
     0.072, Node  59, OpCode   4,    DEPTHWISE_CONV_2D, [    1    7    7  960],  uint8
     0.289, Node  60, OpCode   3,              CONV_2D, [    1    7    7  320],  uint8
     0.371, Node  61, OpCode   3,              CONV_2D, [    1    7    7 1280],  uint8
     0.018, Node  62, OpCode   1,      AVERAGE_POOL_2D, [    1    1    1 1280],  uint8
     0.209, Node  63, OpCode   3,              CONV_2D, [    1    1    1 1001],  uint8
     0.000, Node  64, OpCode  22,              RESHAPE, [    1 1001],  uint8
0.706: 653:military uniform
0.541: 835:suit, suit of clothes
0.533: 753:racket, racquet
0.506: 907:Windsor tie
0.502: 458:bow tie, bow-tie, bowtie
```

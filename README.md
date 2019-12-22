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

First run `$TENSORFLOW_ROOT/configure`. 

Build it for desktop machines (tested on Ubuntu and OS X):

```
bazel build -c opt //:label_image
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

See the `label_image.cc` source code for other command line options.

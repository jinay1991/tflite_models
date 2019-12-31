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
Loaded model data/mobilenet_v2_1.0_224_quant.tflite
Resolved reporter
INFO: Initialized TensorFlow Lite runtime.
Built TfLite Interpreter
invoked 
average time: 20.885 ms. (i.e. 47.8813 images/second) 
============================== Run Order ==============================
	             [node type]	          [start]	  [first]	 [avg ms]	     [%]	  [cdf%]	  [mem KB]	[times called]	[Name]
	                 CONV_2D	            0.000	    1.504	    1.504	  7.205%	  7.205%	     0.000	        1	[MobilenetV2/Conv/Relu6]
	       DEPTHWISE_CONV_2D	            1.504	    0.689	    0.689	  3.301%	 10.506%	     0.000	        1	[MobilenetV2/expanded_conv/depthwise/Relu6]
	                 CONV_2D	            2.194	    0.464	    0.464	  2.223%	 12.729%	     0.000	        1	[MobilenetV2/expanded_conv/project/add_fold]
	                 CONV_2D	            2.658	    2.111	    2.111	 10.114%	 22.843%	     0.000	        1	[MobilenetV2/expanded_conv_1/expand/Relu6]
	       DEPTHWISE_CONV_2D	            4.769	    0.440	    0.440	  2.108%	 24.951%	     0.000	        1	[MobilenetV2/expanded_conv_1/depthwise/Relu6]
	                 CONV_2D	            5.210	    0.175	    0.175	  0.838%	 25.789%	     0.000	        1	[MobilenetV2/expanded_conv_1/project/add_fold]
	                 CONV_2D	            5.386	    0.870	    0.870	  4.168%	 29.957%	     0.000	        1	[MobilenetV2/expanded_conv_2/expand/Relu6]
	       DEPTHWISE_CONV_2D	            6.257	    0.826	    0.826	  3.957%	 33.915%	     0.000	        1	[MobilenetV2/expanded_conv_2/depthwise/Relu6]
	                 CONV_2D	            7.083	    0.300	    0.300	  1.437%	 35.352%	     0.000	        1	[MobilenetV2/expanded_conv_2/project/add_fold]
	                     ADD	            7.383	    0.901	    0.901	  4.317%	 39.668%	     0.000	        1	[MobilenetV2/expanded_conv_2/add]
	                 CONV_2D	            8.284	    0.590	    0.590	  2.827%	 42.495%	     0.000	        1	[MobilenetV2/expanded_conv_3/expand/Relu6]
	       DEPTHWISE_CONV_2D	            8.874	    0.145	    0.145	  0.695%	 43.190%	     0.000	        1	[MobilenetV2/expanded_conv_3/depthwise/Relu6]
	                 CONV_2D	            9.019	    0.077	    0.077	  0.369%	 43.559%	     0.000	        1	[MobilenetV2/expanded_conv_3/project/add_fold]
	                 CONV_2D	            9.096	    0.198	    0.198	  0.949%	 44.507%	     0.000	        1	[MobilenetV2/expanded_conv_4/expand/Relu6]
	       DEPTHWISE_CONV_2D	            9.294	    0.189	    0.189	  0.905%	 45.413%	     0.000	        1	[MobilenetV2/expanded_conv_4/depthwise/Relu6]
	                 CONV_2D	            9.483	    0.096	    0.096	  0.460%	 45.873%	     0.000	        1	[MobilenetV2/expanded_conv_4/project/add_fold]
	                     ADD	            9.579	    0.293	    0.293	  1.404%	 47.276%	     0.000	        1	[MobilenetV2/expanded_conv_4/add]
	                 CONV_2D	            9.872	    0.198	    0.198	  0.949%	 48.225%	     0.000	        1	[MobilenetV2/expanded_conv_5/expand/Relu6]
	       DEPTHWISE_CONV_2D	           10.070	    0.210	    0.210	  1.006%	 49.231%	     0.000	        1	[MobilenetV2/expanded_conv_5/depthwise/Relu6]
	                 CONV_2D	           10.280	    0.141	    0.141	  0.676%	 49.907%	     0.000	        1	[MobilenetV2/expanded_conv_5/project/add_fold]
	                     ADD	           10.421	    0.393	    0.393	  1.883%	 51.789%	     0.000	        1	[MobilenetV2/expanded_conv_5/add]
	                 CONV_2D	           10.814	    0.415	    0.415	  1.988%	 53.778%	     0.000	        1	[MobilenetV2/expanded_conv_6/expand/Relu6]
	       DEPTHWISE_CONV_2D	           11.230	    0.096	    0.096	  0.460%	 54.238%	     0.000	        1	[MobilenetV2/expanded_conv_6/depthwise/Relu6]
	                 CONV_2D	           11.326	    0.122	    0.122	  0.584%	 54.822%	     0.000	        1	[MobilenetV2/expanded_conv_6/project/add_fold]
	                 CONV_2D	           11.448	    0.222	    0.222	  1.064%	 55.886%	     0.000	        1	[MobilenetV2/expanded_conv_7/expand/Relu6]
	       DEPTHWISE_CONV_2D	           11.670	    0.181	    0.181	  0.867%	 56.753%	     0.000	        1	[MobilenetV2/expanded_conv_7/depthwise/Relu6]
	                 CONV_2D	           11.851	    0.214	    0.214	  1.025%	 57.778%	     0.000	        1	[MobilenetV2/expanded_conv_7/project/add_fold]
	                     ADD	           12.065	    0.177	    0.177	  0.848%	 58.626%	     0.000	        1	[MobilenetV2/expanded_conv_7/add]
	                 CONV_2D	           12.242	    0.238	    0.238	  1.140%	 59.766%	     0.000	        1	[MobilenetV2/expanded_conv_8/expand/Relu6]
	       DEPTHWISE_CONV_2D	           12.480	    0.187	    0.187	  0.896%	 60.662%	     0.000	        1	[MobilenetV2/expanded_conv_8/depthwise/Relu6]
	                 CONV_2D	           12.667	    0.186	    0.186	  0.891%	 61.553%	     0.000	        1	[MobilenetV2/expanded_conv_8/project/add_fold]
	                     ADD	           12.853	    0.183	    0.183	  0.877%	 62.430%	     0.000	        1	[MobilenetV2/expanded_conv_8/add]
	                 CONV_2D	           13.037	    0.162	    0.162	  0.776%	 63.206%	     0.000	        1	[MobilenetV2/expanded_conv_9/expand/Relu6]
	       DEPTHWISE_CONV_2D	           13.199	    0.130	    0.130	  0.623%	 63.829%	     0.000	        1	[MobilenetV2/expanded_conv_9/depthwise/Relu6]
	                 CONV_2D	           13.329	    0.130	    0.130	  0.623%	 64.452%	     0.000	        1	[MobilenetV2/expanded_conv_9/project/add_fold]
	                     ADD	           13.459	    0.185	    0.185	  0.886%	 65.338%	     0.000	        1	[MobilenetV2/expanded_conv_9/add]
	                 CONV_2D	           13.644	    0.174	    0.174	  0.834%	 66.172%	     0.000	        1	[MobilenetV2/expanded_conv_10/expand/Relu6]
	       DEPTHWISE_CONV_2D	           13.818	    0.144	    0.144	  0.690%	 66.861%	     0.000	        1	[MobilenetV2/expanded_conv_10/depthwise/Relu6]
	                 CONV_2D	           13.962	    0.266	    0.266	  1.274%	 68.136%	     0.000	        1	[MobilenetV2/expanded_conv_10/project/add_fold]
	                 CONV_2D	           14.228	    0.446	    0.446	  2.137%	 70.273%	     0.000	        1	[MobilenetV2/expanded_conv_11/expand/Relu6]
	       DEPTHWISE_CONV_2D	           14.675	    0.312	    0.312	  1.495%	 71.767%	     0.000	        1	[MobilenetV2/expanded_conv_11/depthwise/Relu6]
	                 CONV_2D	           14.987	    0.512	    0.512	  2.453%	 74.220%	     0.000	        1	[MobilenetV2/expanded_conv_11/project/add_fold]
	                     ADD	           15.500	    0.330	    0.330	  1.581%	 75.801%	     0.000	        1	[MobilenetV2/expanded_conv_11/add]
	                 CONV_2D	           15.831	    0.457	    0.457	  2.189%	 77.991%	     0.000	        1	[MobilenetV2/expanded_conv_12/expand/Relu6]
	       DEPTHWISE_CONV_2D	           16.289	    0.290	    0.290	  1.389%	 79.380%	     0.000	        1	[MobilenetV2/expanded_conv_12/depthwise/Relu6]
	                 CONV_2D	           16.579	    0.474	    0.474	  2.271%	 81.651%	     0.000	        1	[MobilenetV2/expanded_conv_12/project/add_fold]
	                     ADD	           17.053	    0.252	    0.252	  1.207%	 82.858%	     0.000	        1	[MobilenetV2/expanded_conv_12/add]
	                 CONV_2D	           17.305	    0.398	    0.398	  1.907%	 84.765%	     0.000	        1	[MobilenetV2/expanded_conv_13/expand/Relu6]
	       DEPTHWISE_CONV_2D	           17.703	    0.070	    0.070	  0.335%	 85.100%	     0.000	        1	[MobilenetV2/expanded_conv_13/depthwise/Relu6]
	                 CONV_2D	           17.773	    0.171	    0.171	  0.819%	 85.920%	     0.000	        1	[MobilenetV2/expanded_conv_13/project/add_fold]
	                 CONV_2D	           17.944	    0.279	    0.279	  1.337%	 87.256%	     0.000	        1	[MobilenetV2/expanded_conv_14/expand/Relu6]
	       DEPTHWISE_CONV_2D	           18.223	    0.113	    0.113	  0.541%	 87.798%	     0.000	        1	[MobilenetV2/expanded_conv_14/depthwise/Relu6]
	                 CONV_2D	           18.337	    0.286	    0.286	  1.370%	 89.168%	     0.000	        1	[MobilenetV2/expanded_conv_14/project/add_fold]
	                     ADD	           18.623	    0.128	    0.128	  0.613%	 89.781%	     0.000	        1	[MobilenetV2/expanded_conv_14/add]
	                 CONV_2D	           18.751	    0.211	    0.211	  1.011%	 90.792%	     0.000	        1	[MobilenetV2/expanded_conv_15/expand/Relu6]
	       DEPTHWISE_CONV_2D	           18.962	    0.104	    0.104	  0.498%	 91.290%	     0.000	        1	[MobilenetV2/expanded_conv_15/depthwise/Relu6]
	                 CONV_2D	           19.066	    0.274	    0.274	  1.313%	 92.603%	     0.000	        1	[MobilenetV2/expanded_conv_15/project/add_fold]
	                     ADD	           19.340	    0.088	    0.088	  0.422%	 93.024%	     0.000	        1	[MobilenetV2/expanded_conv_15/add]
	                 CONV_2D	           19.428	    0.272	    0.272	  1.303%	 94.328%	     0.000	        1	[MobilenetV2/expanded_conv_16/expand/Relu6]
	       DEPTHWISE_CONV_2D	           19.700	    0.103	    0.103	  0.493%	 94.821%	     0.000	        1	[MobilenetV2/expanded_conv_16/depthwise/Relu6]
	                 CONV_2D	           19.803	    0.480	    0.480	  2.300%	 97.121%	     0.000	        1	[MobilenetV2/expanded_conv_16/project/add_fold]
	                 CONV_2D	           20.283	    0.372	    0.372	  1.782%	 98.903%	     0.000	        1	[MobilenetV2/Conv_1/Relu6]
	         AVERAGE_POOL_2D	           20.655	    0.019	    0.019	  0.091%	 98.994%	     0.000	        1	[MobilenetV2/Logits/AvgPool]
	                 CONV_2D	           20.674	    0.210	    0.210	  1.006%	100.000%	     0.000	        1	[MobilenetV2/Logits/Conv2d_1c_1x1/BiasAdd]
	                 RESHAPE	           20.884	    0.000	    0.000	  0.000%	100.000%	     0.000	        1	[output]

============================== Top by Computation Time ==============================
	             [node type]	          [start]	  [first]	 [avg ms]	     [%]	  [cdf%]	  [mem KB]	[times called]	[Name]
	                 CONV_2D	            2.658	    2.111	    2.111	 10.114%	 10.114%	     0.000	        1	[MobilenetV2/expanded_conv_1/expand/Relu6]
	                 CONV_2D	            0.000	    1.504	    1.504	  7.205%	 17.319%	     0.000	        1	[MobilenetV2/Conv/Relu6]
	                     ADD	            7.383	    0.901	    0.901	  4.317%	 21.636%	     0.000	        1	[MobilenetV2/expanded_conv_2/add]
	                 CONV_2D	            5.386	    0.870	    0.870	  4.168%	 25.804%	     0.000	        1	[MobilenetV2/expanded_conv_2/expand/Relu6]
	       DEPTHWISE_CONV_2D	            6.257	    0.826	    0.826	  3.957%	 29.761%	     0.000	        1	[MobilenetV2/expanded_conv_2/depthwise/Relu6]
	       DEPTHWISE_CONV_2D	            1.504	    0.689	    0.689	  3.301%	 33.062%	     0.000	        1	[MobilenetV2/expanded_conv/depthwise/Relu6]
	                 CONV_2D	            8.284	    0.590	    0.590	  2.827%	 35.888%	     0.000	        1	[MobilenetV2/expanded_conv_3/expand/Relu6]
	                 CONV_2D	           14.987	    0.512	    0.512	  2.453%	 38.341%	     0.000	        1	[MobilenetV2/expanded_conv_11/project/add_fold]
	                 CONV_2D	           19.803	    0.480	    0.480	  2.300%	 40.641%	     0.000	        1	[MobilenetV2/expanded_conv_16/project/add_fold]
	                 CONV_2D	           16.579	    0.474	    0.474	  2.271%	 42.912%	     0.000	        1	[MobilenetV2/expanded_conv_12/project/add_fold]

Number of nodes executed: 65
============================== Summary by node type ==============================
	             [Node type]	  [count]	  [avg ms]	    [avg %]	    [cdf %]	  [mem KB]	[times called]
	                 CONV_2D	       36	    13.695	    65.611%	    65.611%	     0.000	       36
	       DEPTHWISE_CONV_2D	       17	     4.229	    20.261%	    85.872%	     0.000	       17
	                     ADD	       10	     2.930	    14.037%	    99.909%	     0.000	       10
	         AVERAGE_POOL_2D	        1	     0.019	     0.091%	   100.000%	     0.000	        1
	                 RESHAPE	        1	     0.000	     0.000%	   100.000%	     0.000	        1

Timings (microseconds): count=1 curr=20873
Memory (bytes): count=0
65 nodes observed

0.705882: 653:military uniform
0.541176: 835:suit, suit of clothes
0.533333: 753:racket, racquet
0.505882: 907:Windsor tie
Retriving 173 tensors...
```

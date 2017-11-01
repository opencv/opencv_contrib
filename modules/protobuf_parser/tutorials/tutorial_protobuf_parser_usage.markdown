# How to parse protobuf files using OpenCV {#tutorial_protobuf_parser_usage}

## Introduction
Protocol buffer is an efficient way to serialize structured data. The main
thing is a `.proto` file that defines single values as `fields` and their sets
as `messages`.
@code
message Values {
  required int32 value_int32 = 1;
  repeated float value_float = 2;
};
@endcode

Using protobuf compiler called `protoc` it can be compiled into pair of C++
source and header files. Then link it into the target project and use generated
C++ interface for writing or reading `.pb` files with data.

@code
Values myMessage;
myMessage.set_value_int32(-1023);
myMessage.add_value_float(0.15f);
myMessage.add_value_float(9.01f);
@endcode

Generated `.pb.cc` and `.pb.h` depends on protobuf library. So it also should be
linked in project.

## Protobuf parser module
If your project use OpenCV you may enable protobuf_parser module and use it for
parsing serialized protocol buffers. One and only preparing is a compilation of
textual `.proto` file into binary one. Use protobuf compiler once:
@code
protoc --descriptor_set_out=/path/to/output.pb /path/to/input.proto
@endcode

## Sample
@include samples/parse_caffe_model.cpp

## Explanation

-# Parse command line arguments.
   @snippet samples/parse_caffe_model.cpp Parse arguments

   Arguments `--proto` and `--caffemodel` are required.
   They are used to determine paths to compiled binary `.proto` file and
   `.caffemodel` network from the Caffe framework respectively.

-# Initialize parser using binary `.proto` file.
   @snippet samples/parse_caffe_model.cpp Initialize parser

   Initialize parser by `.proto` file to be ready for parsing `.caffemodel`.
   For every protocol buffer one of messages is a root message. In case of
   networks from Caffe it will be `caffe.NetParameter`.

-# Parse `.caffemodel`.
   @snippet samples/parse_caffe_model.cpp Parse model

-# Print the name of network if it exists.
   @snippet samples/parse_caffe_model.cpp Print name of network

-# For every layer print it's name, type and number of trainable parameters
stored in protobuf.
   @snippet samples/parse_caffe_model.cpp Print layers data

   `caffe.proto` defines trainable parameters in different ways.
   For differentiate between deprecated and actual styles we check what kind of
   field is not empty.

Output of application for AlexNet model will be:
@code
Network: AlexNet
Layer data of type DATA with 0 parameters
Layer conv1 of type CONVOLUTION with 34944 parameters
Layer relu1 of type RELU with 0 parameters
Layer norm1 of type LRN with 0 parameters
Layer pool1 of type POOLING with 0 parameters
Layer conv2 of type CONVOLUTION with 307456 parameters
Layer relu2 of type RELU with 0 parameters
Layer norm2 of type LRN with 0 parameters
Layer pool2 of type POOLING with 0 parameters
Layer conv3 of type CONVOLUTION with 885120 parameters
Layer relu3 of type RELU with 0 parameters
Layer conv4 of type CONVOLUTION with 663936 parameters
Layer relu4 of type RELU with 0 parameters
Layer conv5 of type CONVOLUTION with 442624 parameters
Layer relu5 of type RELU with 0 parameters
Layer pool5 of type POOLING with 0 parameters
Layer fc6 of type INNER_PRODUCT with 37752832 parameters
Layer relu6 of type RELU with 0 parameters
Layer drop6 of type DROPOUT with 0 parameters
Layer fc7 of type INNER_PRODUCT with 16781312 parameters
Layer relu7 of type RELU with 0 parameters
Layer drop7 of type DROPOUT with 0 parameters
Layer fc8 of type INNER_PRODUCT with 4097000 parameters
Layer loss of type SOFTMAX_LOSS with 0 parameters
@endcode

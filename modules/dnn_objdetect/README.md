# Object Detection using Convolutional Neural Networks

This module uses Convolutional Neural Networks for detecting objects in an image

## Dependencies
- opencv dnn module
- Google Protobuf

## Building this module
Run the following command to build this module:

```make
cmake -DOPENCV_EXTRA_MODULES_PATH=<opencv_contrib>/modules -Dopencv_dnn_objdetect=ON <opencv_source_dir>
```

## Models

There are two models which are trained.
#### SqueezeNet model trained for Image Classification.
- This model was trained for 1500000 iterations with a batch size of 16
- Size of Model: 4.9MB
- Top-1 Accuracy on ImageNet 2012 DataSet: 56.10%
- Top-5 Accuracy on ImageNet 2012 DataSet: 79.54%
- Link to trained weights: [here]()

#### SqueezeDet model trained for Object Detection
- This model was trained for 180000 iterations with a batch size of 16
- Size of the Model: 14.9MB
- Link to the trained weights: [here]()

## Usage

#### With Caffe
For details pertaining to the usage of the model, have a look at [this repository](https://github.com/kvmanohar22/caffe)

You can infact train your own object detection models with the loss function which is implemented.

#### Without Caffe, using `opencv's dnn module`
`test/core_detect.cpp` gives an example of how to use the model to predict the bounding boxes.
`test/image_classification.cpp` gives an example of how to use the model to classify an image.

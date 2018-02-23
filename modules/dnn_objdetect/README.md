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
- Link to trained weights: [here](https://github.com/kvmanohar22/caffe/blob/obj_detect_loss/proto/SqueezeNet.caffemodel) ([copy](https://github.com/opencv/opencv_3rdparty/tree/dnn_objdetect_20170827))

#### SqueezeDet model trained for Object Detection

- This model was trained for 180000 iterations with a batch size of 16
- Size of the Model: 14.2MB
- Link to the trained weights: [here](https://github.com/kvmanohar22/caffe/blob/obj_detect_loss/proto/SqueezeDet.caffemodel) ([copy](https://github.com/opencv/opencv_3rdparty/tree/dnn_objdetect_20170827))

## Usage

#### With Caffe

For details pertaining to the usage of the model, have a look at [this repository](https://github.com/kvmanohar22/caffe)

You can infact train your own object detection models with the loss function which is implemented.

#### Without Caffe, using `opencv's dnn module`
`tutorials/core_detect.cpp` gives an example of how to use the model to predict the bounding boxes.
`tutorials/image_classification.cpp` gives an example of how to use the model to classify an image.

Here's the brief summary of examples. For detailed usage and testing, refer `tutorials` directory.

## Examples:

### Image Classification

```c++
// Read the net along with it's trained weights
cv::dnn::net = cv::dnn::readNetFromCaffe(model_defn, model_weights);

// Read an image
cv::Mat image = cv::imread(image_file);

// Convert the image into blob
cv::Mat image_blob = cv::net::blobFromImage(image);

// Get the output of "predictions" layer
cv::Mat probs = net.forward("predictions");

```
`probs` is a 4-d tensor of shape `[1, 1000, 1, 1]` which is obtained after the application of `softmax` activation.

### Object Detection

```c++
// Reading the network and weights, converting image to blob is same as Image Classification example.

// Forward through the network and collect blob data
cv::Mat delta_bboxs = net.forward("slice")[0];
cv::Mat conf_scores = net.forward("softmax");
cv::Mat class_scores = net.forward("sigmoid");
```
Three blobs aka `delta_bbox`, `conf_scores`, `class_scores` are post-processed in `cv::dnn_objdetect::InferBbox` class and the bounding boxes predicted.

```c++
InferBbox infer(delta_bbox, class_scores, conf_scores);
infer.filter();
```

`infer.filter()` returns vector of `cv::dnn_objdetect::object` of predictions. Here `cv::dnn_objdetect::object` is a structure containing the following elements.

```c++
typedef struct {
  int xmin, xmax;
  int ymin, ymax;
  int class_idx;
  std::string label_name;
  double class_prob;
} object;

```
For further details on post-processing refer this detailed [blog-post](https://kvmanohar22.github.io/GSoC/).

## Results from Object Detection

Refer `tutorials` directory for results.

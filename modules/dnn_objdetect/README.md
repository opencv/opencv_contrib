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

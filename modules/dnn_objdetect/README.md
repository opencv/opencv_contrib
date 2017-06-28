# Object Detection using Convolutional Neural Networks

This module uses Convolutional Neural Networks for detecting objects in an image

## Dependencies
- Caffe
- Google Protobuf
- Glog

## Building this module
Set the variable `Caffe_DIR` in the file `FindCaffe.cmake` or alternately set this variable while running `cmake` as :

```make
cmake -DOPENCV_EXTRA_MODULES_PATH=<opencv_contrib>/modules -DCaffe_DIR=<caffe_root_dir> <opencv_source_dir>
```

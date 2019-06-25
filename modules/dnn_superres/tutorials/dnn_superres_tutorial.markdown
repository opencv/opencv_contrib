Super Resolution using CNNs {#tutorial_dnn_superres}
===========================

# Building

Run the following command to build this module:

```make
cmake -DOPENCV_EXTRA_MODULES_PATH=<opencv_contrib>/modules -Dopencv_dnn_superres=ON <opencv_source_dir>
```

# Super resolution sample code

See the "dnn_superres" in the samples for an idea of how to run it. For example:

```dnn_superres/samples/dnn_superres.cpp ./butterfly.png edsr 2
```
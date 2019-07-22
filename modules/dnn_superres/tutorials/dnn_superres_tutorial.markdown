Super Resolution using CNNs {#tutorial_dnn_superres}
===========================

# Building

Run the following command to build this module:

```make
cmake -DOPENCV_EXTRA_MODULES_PATH=<opencv_contrib>/modules -Dopencv_dnn_superres=ON <opencv_source_dir>
```

# Super resolution sample code

See the "dnn_superres" in the samples for an idea of how to run it. For example:

```
dnn_superres/samples/dnn_superres.cpp ./butterfly.png edsr 2
```

# Single output

Run the sample code to do single output super-resolution with the implemented models.\
ESPCN model can now support 2x, 3x, and 4x super resolution.

```
./bin/example_dnn_superres_dnn_superres path/to/image.png espcn 2 \
/path/to/opencv_contrib/modules/dnn_superres/models/ESPCN_x2.pb
```

# Multiple output

LapSRN supports multiple outputs with one forward pass. It can now support 2x, 4x, 8x, and (2x, 4x) and (2x, 4x, 8x) super-resolution.\
The uploaded trained model files have the following output node names:
- 2x model: NCHW_output
- 4x model: NCHW_output_2x, NCHW_output_4x
- 8x model: NCHW_output_2x, NCHW_output_4x, NCHW_output_8x

```
./bin/example_dnn_superres_dnn_superres_multioutput path/to/image.png 2,4 NCHW_output_2x,NCHW_output_4x \
path/to/opencv_contrib/modules/dnn_superres/models/LapSRN_x4.pb
```
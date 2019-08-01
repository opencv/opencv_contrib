# Super Resolution using Convolutional Neural Networks

This module contains several learning-based algorithms for upscaling an image.

## Dependencies
-
-

## Building this module

Run the following command to build this module:

```make
cmake -DOPENCV_EXTRA_MODULES_PATH=<opencv_contrib>/modules -Dopencv_dnn_superres=ON <opencv_source_dir>
```

## Models

There are four models which are trained. (Not yet implemented!!)

#### EDSR

- Size of the model:
- This model was trained for <> iterations with a batch size of <>
- Link to model:
- Advantage:
- Disadvantage:
- Speed:

#### ESPCN

Trained models can be downloaded from [here](https://github.com/fannymonori/TF-ESPCN/tree/master/export).

- Size of the model: ~100kb
- This model was trained for 100 iterations with a batch size of 32
- Link to implementation code: https://github.com/fannymonori/TF-ESPCN
- x2, x3, x4 trained models available
- Advantage: It is tiny, and fast, and still perform well.
- Disadvantage: Perform worse visually than newer, more robust models.
- Speed:

#### FSRCNN

- Size of the model:
- This model was trained for <> iterations with a batch size of <>
- Link to model:
- Advantage:
- Disadvantage:
- Speed:

#### LapSRN

Trained models can be downloaded from [here](https://github.com/fannymonori/TF-LapSRN/tree/master/export).

- Size of the model: between 1-5Mb
- This model was trained for ~50 iterations with a batch size of 32
- Link to implementation code: https://github.com/fannymonori/TF-LAPSRN
- x2, x4, x8 trained models available
- Advantage: The model can do multi-scale super-resolution with one forward pass. It can now support 2x, 4x, 8x, and [2x, 4x] and [2x, 4x, 8x] super-resolution.
- Disadvantage: It is a slower model.
- Speed
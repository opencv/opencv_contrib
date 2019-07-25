# Super Resolution using Convolutional Neural Networks

This module contains several learning-based algorithms for upscaling an image.

## Building this module

Run the following command to build this module:

```make
cmake -DOPENCV_EXTRA_MODULES_PATH=<opencv_contrib>/modules -Dopencv_dnn_superres=ON <opencv_source_dir>
```

## Models

There are four models which are trained. (Not all are implemented yet!!)

#### EDSR

- Size of the model:
- This model was trained for <> iterations with a batch size of <>
- Link to model code:
- Link to model:
- Advantage: Highly accurate
- Disadvantage: Slow and big
- Speed:

#### ESPCN

- Size of the model:
- This model was trained for <> iterations with a batch size of <>
- Link to model:
- Advantage:
- Disadvantage:
- Speed:

#### FSRCNN

- Size of the model: ~40KB
- This model was trained for ~30 iterations with a batch size of 1
- Link to model code: https://github.com/Saafke/FSRCNN_Tensorflow
- Link to model: /models/FSRCNN_x2.pb, /models/FSRCNN_x3.pb, /models/FSRCNN_x4.pb
- Advantage: Fast, small and accurate
- Disadvantage: Not state-of-the-art accuracy
- Speed: ~0.06sec for a 256x256 image | ~0.2sec for a 512x512 image

#### LapSRN

- Size of the model:
- This model was trained for <> iterations with a batch size of <>
- Link to model:
- Advantage:
- Disadvantage:
- Speed:
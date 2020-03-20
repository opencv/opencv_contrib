Information Flow AlphaMatting {#tutorial_alphamat_information_flow}
=============================

Alphamatting is the problem of extracting the foreground from an image. Given the input of image and its corresponding trimap, we try to extract the foreground from the background. Following is an example -

Input Image: ![](images/net_input_image.jpg)
Input Trimap: ![](images/net_trimap.jpg)
Output alpha Matte: ![](images/net_result.jpg)

Build the module using -
```
cmake -DOPENCV_EXTRA_MODULES_PATH=<opencv_contrib>/modules -Dopencv_alphamat=ON <opencv_source_dir>
make
```

Usage Example
-------------

@includelineno alphamat/samples/information_flow_matting.cpp

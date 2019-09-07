Information Flow AlphaMatting {#tutorial_alphamat_information_flow}
=======================

Alphamatting is the problem of extracting the foreground from an image. Given the input of image and its corresponding trimap, we try to extract the foreground from the background. Following is an example -

Input Image: ![](images/net_input_image.jpg)
Input Trimap: ![](images/net_trimap.jpg)
Output alpha Matte: ![](images/net_result.jpg)

Source Code of the sample
-----------

@includelineno alphamat/samples/alphamat_information_flow.cpp

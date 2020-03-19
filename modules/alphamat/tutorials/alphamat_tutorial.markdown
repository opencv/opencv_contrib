Information Flow Alpha Matting {#tutorial_alphamat}
============================

This project was part of Google Summer of Code 2019.

*Student:* Muskaan Kularia

*Mentor:* Sunita Nayak

Alphamatting is the problem of extracting the foreground from an image. The extracted foreground can be used for further operations like changing the background in an image.

Given an input image and its corresponding trimap, we try to extract the foreground from the background. Following is an example:

Input Image: ![](samples/input_images/plant.jpg)
Input Trimap: ![](samples/trimaps/plant.png)
Output alpha Matte: ![](samples/output_mattes/plant_result.jpg)

This project is implementation of @cite aksoy2017designing . It required implementation of parts of other papers [2,3,4].

# Building

This module uses the Eigen package.

Build the sample code of the alphamat module using the following two cmake commands run inside the build folder:
```
cmake -DOPENCV_EXTRA_MODULES_PATH=<path to opencv_contrib modules> -DBUILD_EXAMPLES=ON ..

cmake --build . --config Release --target example_alphamat_information_flow_matting
```
Please refer to OpenCV building tutorials for further details, if needed.

# Testing

The built target can be tested as follows:
```
example_alphamat_information_flow_matting -img=<path to input image file> -tri=<path to the corresponding trimap> -out=<path to save output matte file>
```
# Source Code of the sample

@includelineno alphamat/samples/information_flow_matting.cpp


# References

[1] Yagiz Aksoy, Tunc Ozan Aydin, Marc Pollefeys, "[Designing Effective Inter-Pixel Information Flow for Natural Image Matting](http://people.inf.ethz.ch/aksoyy/ifm/)", CVPR, 2017.

[2] Roweis, Sam T., and Lawrence K. Saul. "[Nonlinear dimensionality reduction by locally linear embedding](https://science.sciencemag.org/content/290/5500/2323)" Science 290.5500 (2000): 2323-2326.

[3] Anat Levin, Dani Lischinski, Yair Weiss, "[A Closed Form Solution to Natural Image Matting](https://www.researchgate.net/publication/5764820_A_Closed-Form_Solution_to_Natural_Image_Matting)", IEEE TPAMI, 2008.

[4] Qifeng Chen, Dingzeyu Li, Chi-Keung Tang, "[KNN Matting](http://dingzeyu.li/files/knn-matting-tpami.pdf)", IEEE TPAMI, 2013.

[5] Yagiz Aksoy, "[Affinity Based Matting Toolbox](https://github.com/yaksoy/AffinityBasedMattingToolbox)".

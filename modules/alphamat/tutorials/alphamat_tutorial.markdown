Information Flow Alpha Matting {#tutorial_alphamat}
============================

This project was part of Google Summer of Code 2019.

*Student:* Muskaan Kularia

*Mentor:* Sunita Nayak

Alphamatting is the problem of extracting the foreground with soft boundaries from a background image. The extracted foreground can be used for further operations like changing the background in an image.

Given an input image and its corresponding trimap, we try to extract the foreground from the background. Following is an example:

Input Image: ![](alphamat/samples/input_images/plant.jpg)
Input image should be preferably a RGB image.

Input Trimap: ![](alphamat/samples/trimaps/plant.png)
The trimap image is a greyscale image that contains information about the foreground(white pixels), background(black pixels) and unknown(grey) pixels.

Output alpha Matte: ![](alphamat/samples/output_mattes/plant_result.png)
The computed alpha matte is saved as a greyscale image where the pixel values indicate the opacity of the extracted foreground object. These opacity values can be used to blend the foreground object into a diffferent backgound, as shown below:
![](plant_new_backgrounds.jpg)

Following are some more results.
![](matting_results.jpg)

The first column is input RGB image, the second column is input trimap, third column is the extracted alpha matte and the last two columns show the foreground object blended on new backgrounds.

This project is implementation of @cite aksoy2017designing . It also required implementation of parts of other papers [2,3,4].

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
<path to your opencv build directory>/bin/example_alphamat_information_flow_matting -img=<path to input image file> -tri=<path to the corresponding trimap> -out=<path to save output matte file>
```
# Source Code of the sample

@includelineno alphamat/samples/information_flow_matting.cpp

# References

[1] Yagiz Aksoy, Tunc Ozan Aydin, Marc Pollefeys, [Designing Effective Inter-Pixel Information Flow for Natural Image Matting](https://www.researchgate.net/publication/318489370_Designing_Effective_Inter-Pixel_Information_Flow_for_Natural_Image_Matting), CVPR, 2017.

[2] Roweis, Sam T., and Lawrence K. Saul. [Nonlinear dimensionality reduction by locally linear embedding](https://science.sciencemag.org/content/290/5500/2323), Science 290.5500 (2000): 2323-2326.

[3] Anat Levin, Dani Lischinski, Yair Weiss, [A Closed Form Solution to Natural Image Matting](https://www.researchgate.net/publication/5764820_A_Closed-Form_Solution_to_Natural_Image_Matting), IEEE TPAMI, 2008.

[4] Qifeng Chen, Dingzeyu Li, Chi-Keung Tang, [KNN Matting](http://dingzeyu.li/files/knn-matting-tpami.pdf), IEEE TPAMI, 2013.

[5] Yagiz Aksoy, [Affinity Based Matting Toolbox](https://github.com/yaksoy/AffinityBasedMattingToolbox).

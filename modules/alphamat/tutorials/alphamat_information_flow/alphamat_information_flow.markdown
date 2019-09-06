Information Flow AlphaMatting {#tutorial_alphamat_information_flow}
======================

Goal
----

In this chapter,

-   We will familiarize with the computer vision based alphamatting in OpenCV.

Basics
------

Alphamatting is the problem of extracting the foreground from an image. Given the input of image and its corresponding trimap, we try to extract the foreground from the background. Following is an example -

Input Image                | Input trimap              | Ouput Alpha matte
:-------------------------:|:-------------------------:|:-------------------------:
<img src="https://github.com/muskaankularia/opencv_contrib/blob/alphamatting/modules/alphamat/img/net.png" alt="alt text" width="300" height="200"> | <img src="https://github.com/muskaankularia/opencv_contrib/blob/alphamatting/modules/alphamat/trimap/net.png" alt="alt text" width="300" height="200"> | <img src="https://github.com/muskaankularia/opencv_contrib/blob/alphamatting/modules/alphamat/Result/result_net.png" alt="alt text" width="300" height="200">

This project is implementation of [[Information-Flow Matting](https://arxiv.org/pdf/1707.05055.pdf)] by Aksoy et al.[1]. It required implementation of some parts of other papers [2,3].

This is a pixel-affinity based alpha matting algorithm which solves a linear system of equations using preconditioned conjugate gradient method. Affinity-based methods operate by propagating opacity information from known opacity regions(K) into unknown opacity regions(U) using a variety of affinity definitions mentioned as -
* Color mixture information flow - Opacity transitions in a matte occur as a result of the original colors in the image getting mixed with each other due to transparency or intricate parts of an object. They make use of this fact by representing each pixel in U as a mixture of similarly-colored pixels and the difference is the energy term ECM,  which is to be reduced.
* K-to-U information flow - Connections from every pixel in U to both F(foreground pixels) and B(background pixels) are made to facilitate direct information flow from known-opacity regions to even the most remote opacity-transition regions in the image.
* Intra U information flow - They distribute the information inside U effectively by encouraging pixels with similar colors inside U to have similar opacity.
* Local information flow - Spatial connectivity is one of the main cues for information flow which is achieved by connecting each pixel in U to its immediate neighbors to ensure spatially smooth mattes.

Using these information flow, energy/error(E) is obtained as a weighted local composite of E<sub>CM</sub>, E<sub>KU</sub>(K-to-U information flow), E<sub>UU</sub>(Intra U information flow), E<sub>L</sub>(Local information flow).
E represents the deviation of unknown pixels opacity or colour from what we predict it to be using other pixels. So, the algorithm aims at minimizing this error.

Results
-------
Input Image             | Ouput Alpha matte
:-------------------------:|:-------------------------:
<img src="https://github.com/muskaankularia/opencv_contrib/blob/alphamatting/modules/alphamat/img/net.png" alt="alt text" width="200" height="155">  |  <img src="https://github.com/muskaankularia/opencv_contrib/blob/alphamatting/modules/alphamat/Result/result_net.png" alt="alt text" width="200" height="155">
<img src="https://github.com/muskaankularia/opencv_contrib/blob/alphamatting/modules/alphamat/img/elephant.png" alt="alt text" width="200" height="155">  |  <img src="https://github.com/muskaankularia/opencv_contrib/blob/alphamatting/modules/alphamat/Result/result_elephant.png" alt="alt text" width="200" height="155">
<img src="https://github.com/muskaankularia/opencv_contrib/blob/alphamatting/modules/alphamat/img/pineapple.png" alt="alt text" width="200" height="155">  |  <img src="https://github.com/muskaankularia/opencv_contrib/blob/alphamatting/modules/alphamat/Result/result_pineapple.png" alt="alt text" width="200" height="155">
<img src="https://github.com/muskaankularia/opencv_contrib/blob/alphamatting/modules/alphamat/img/plasticbag.png" alt="alt text" width="200" height="155">  |  <img src="https://github.com/muskaankularia/opencv_contrib/blob/alphamatting/modules/alphamat/Result/result_plasticbag.png" alt="alt text" width="200" height="155">

References
--------------------

[1] Yagiz Aksoy, Tunc Ozan Aydin, Marc Pollefeys, "Designing Effective Inter-Pixel Information Flow for Natural Image Matting", CVPR, 2017. [[link](http://people.inf.ethz.ch/aksoyy/ifm/)]

[2] Roweis, Sam T., and Lawrence K. Saul. "Nonlinear dimensionality reduction by locally linear embedding." science 290.5500 (2000): 2323-2326.[[link](https://science.sciencemag.org/content/290/5500/2323)]

[3] Ehsan Shahrian, Deepu Rajan, Brian Price, Scott Cohen, "Improving Image Matting using Comprehensive Sampling Sets", CVPR 2013 [[paper](http://www.cv-foundation.org/openaccess/content_cvpr_2013/papers/Shahrian_Improving_Image_Matting_2013_CVPR_paper.pdf)]

[4] Affinity Based Matting Toolbox by Yagiz Aksoy[[link](https://github.com/yaksoy/AffinityBasedMattingToolbox)]

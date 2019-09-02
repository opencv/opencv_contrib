# Designing Effective Inter-Pixel Information Flow for Natural Image Matting:

Alphamatting is the problem of extracting the foreground from an image. Given the input of image and its corresponding trimap, we try to extract the foreground from the background. Following is an example -

Input Image                | Input trimap              | Ouput Alpha matte
:-------------------------:|:-------------------------:|:-------------------------:
<img src="https://github.com/muskaankularia/opencv_contrib/blob/alphamatting/modules/alphamat/img/net.png" alt="alt text" width="300" height="200"> | <img src="https://github.com/muskaankularia/opencv_contrib/blob/alphamatting/modules/alphamat/trimap/net.png" alt="alt text" width="300" height="200"> | <img src="https://github.com/muskaankularia/opencv_contrib/blob/alphamatting/modules/alphamat/Result/result_net.png" alt="alt text" width="300" height="200">

This project is implementation of Information-Flow Matting [Yağız Aksoy, Tunç Ozan Aydın, Marc Pollefeys] [1]. It required implementation of some parts of other papers [2,3].

This is a pixel-affinity based alpha matting algorithm which solves a linear system of equations using preconditioned conjugate gradient method. Affinity-based methods operate by propagating opacity information from known opacity regions(K) into unknown opacity regions(U) using a variety of affinity definitions mentioned as -
* Color mixture information flow - Opacity transitions in a matte occur as a result of the original colors in the image getting mixed with each other due to transparency or intricate parts of an object. They make use of this fact by representing each pixel in U as a mixture of similarly-colored pixels and the difference is the energy term ECM,  which is to be reduced. This is coded in **cm.hpp**
* K-to-U information flow - Connections from every pixel in U to both F(foreground pixels) and B(background pixels) are made to facilitate direct information flow from known-opacity regions to even the most remote opacity-transition regions in the image. This is coded in **KtoU.hpp**
* Intra U information flow - They distribute the information inside U effectively by encouraging pixels with similar colors inside U to have similar opacity. This is coded in **intraU.hpp**
* Local information flow - Spatial connectivity is one of the main cues for information flow which is achieved by connecting each pixel in U to its immediate neighbors to ensure spatially smooth mattes. This is coded in **local_info.hpp**

Using these information flow, energy/error(E) is obtained as a weighted local composite of E<sub>CM</sub>, E<sub>KU</sub>(K-to-U information flow), E<sub>UU</sub>(Intra U information flow), E<sub>L</sub>(Local information flow).
E represents the deviation of unknown pixels opacity or colour from what we predict it to be using other pixels. So, the algorithm aims at minimizing this error. This is coded in **alphac.cpp**

Pre-processing and post-processing is implemented in **trimming.hpp**

To run the code -
1. **g++ -std=c++11 alphac.cpp \`pkg-config --cflags --libs opencv\`**
1. **./a.out \<path to image> \<path to corresponding trimap>**

Sample image and trimap are in opencv_contrib/modules/alphamat/src/img and opencv_contrib/modules/alphamat/src/trimap

## Results

Results for input_lowres are available here -
https://docs.google.com/document/d/1BJG4633_U5K-Z0QLp3RTi43q25NI0hrTw-Q4w_85NrA/edit?usp=sharing

Input Image             | Ouput Alpha matte
:-------------------------:|:-------------------------:
<img src="https://github.com/muskaankularia/opencv_contrib/blob/alphamatting/modules/alphamat/img/net.png" alt="alt text" width="200" height="155">  |  <img src="https://github.com/muskaankularia/opencv_contrib/blob/alphamatting/modules/alphamat/Result/result_net.png" alt="alt text" width="200" height="155">
<img src="https://github.com/muskaankularia/opencv_contrib/blob/alphamatting/modules/alphamat/img/doll.png" alt="alt text" width="200" height="155">  |  <img src="https://github.com/muskaankularia/opencv_contrib/blob/alphamatting/modules/alphamat/Result/result_doll.png" alt="alt text" width="200" height="155">
<img src="https://github.com/muskaankularia/opencv_contrib/blob/alphamatting/modules/alphamat/img/donkey.png" alt="alt text" width="200" height="155">  |  <img src="https://github.com/muskaankularia/opencv_contrib/blob/alphamatting/modules/alphamat/Result/result_donkey.png" alt="alt text" width="200" height="155">
<img src="https://github.com/muskaankularia/opencv_contrib/blob/alphamatting/modules/alphamat/img/elephant.png" alt="alt text" width="200" height="155">  |  <img src="https://github.com/muskaankularia/opencv_contrib/blob/alphamatting/modules/alphamat/Result/result_elephant.png" alt="alt text" width="200" height="155">
<img src="https://github.com/muskaankularia/opencv_contrib/blob/alphamatting/modules/alphamat/img/pineapple.png" alt="alt text" width="200" height="155">  |  <img src="https://github.com/muskaankularia/opencv_contrib/blob/alphamatting/modules/alphamat/Result/result_pineapple.png" alt="alt text" width="200" height="155">
<img src="https://github.com/muskaankularia/opencv_contrib/blob/alphamatting/modules/alphamat/img/plant.png" alt="alt text" width="200" height="155">  |  <img src="https://github.com/muskaankularia/opencv_contrib/blob/alphamatting/modules/alphamat/Result/result_plant.png" alt="alt text" width="200" height="155">
<img src="https://github.com/muskaankularia/opencv_contrib/blob/alphamatting/modules/alphamat/img/plasticbag.png" alt="alt text" width="200" height="155">  |  <img src="https://github.com/muskaankularia/opencv_contrib/blob/alphamatting/modules/alphamat/Result/result_plasticbag.png" alt="alt text" width="200" height="155">
<img src="https://github.com/muskaankularia/opencv_contrib/blob/alphamatting/modules/alphamat/img/troll.png" alt="alt text" width="200" height="155">  |  <img src="https://github.com/muskaankularia/opencv_contrib/blob/alphamatting/modules/alphamat/Result/result_troll.png" alt="alt text" width="200" height="155">

Average time taken to compute the different flows is 40s, but solving of linear equations using preconditioned conjugate gradient method takes another 2-3 min, which can be lessened by allowing lesser iterations.

## TO DO

* Results need to be improved by extensively comparing each flow's matrix with yaksoy MATLAB implementation [4].
* Runtime needs improvement.
* Third part library(Eigen, nanoflann) dependencies can be removed.

## References

[1] Yagiz Aksoy, Tunc Ozan Aydin, Marc Pollefeys, "Designing Effective Inter-Pixel Information Flow for Natural Image Matting", CVPR, 2017. [[link](http://people.inf.ethz.ch/aksoyy/ifm/)]

[2] Roweis, Sam T., and Lawrence K. Saul. "Nonlinear dimensionality reduction by locally linear embedding." science 290.5500 (2000): 2323-2326.[[link](https://science.sciencemag.org/content/290/5500/2323)]

[3] Ehsan Shahrian, Deepu Rajan, Brian Price, Scott Cohen, "Improving Image Matting using Comprehensive Sampling Sets", CVPR 2013 [[paper](http://www.cv-foundation.org/openaccess/content_cvpr_2013/papers/Shahrian_Improving_Image_Matting_2013_CVPR_paper.pdf)]

[4] Affinity Based Matting Toolbox by Yagiz Aksoy[[link](https://github.com/yaksoy/AffinityBasedMattingToolbox)]

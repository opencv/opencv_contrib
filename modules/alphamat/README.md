## Designing Effective Inter-Pixel Information Flow for Natural Image Matting: 
This is a pixel-affinity based alpha matting algorithm which solves a linear system of equations using preconditioned conjugate gradient method. Affinity-based methods operate by propagating opacity information from known opacity regions(K) into unknown opacity regions(U) using a variety of affinity definitions mentioned as - 
* Color mixture information flow - Opacity transitions in a matte occur as a result of the original colors in the image getting mixed with each other due to transparency or intricate parts of an object. They make use of this fact by representing each pixel in U as a mixture of similarly-colored pixels and the difference is the energy term ECM,  which is to be reduced. This is coded in **cm.hpp**
* K-to-U information flow - Connections from every pixel in U to both F(foreground pixels) and B(background pixels) are made to facilitate direct information flow from known-opacity regions to even the most remote opacity-transition regions in the image. This is coded in **KtoU.hpp**
* Intra U information flow - They distribute the information inside U effectively by encouraging pixels with similar colors inside U to have similar opacity. This is coded in **intraU.hpp** 
* Local information flow - Spatial connectivity is one of the main cues for information flow which is achieved by connecting each pixel in U to its immediate neighbors to ensure spatially smooth mattes. This is coded in **local_info.hpp**

Using these information flow, energy/error(E) is obtained as a weighted local composite of E<sub>CM</sub>, E<sub>KU</sub>(K-to-U information flow), E<sub>UU</sub>(Intra U information flow), E<sub>L</sub>(Local information flow.
E represents the deviation of unknown pixels opacity or colour from what we predict it to be using other pixels. So, the algorithm aims at minimizing this error. This is coded in **alphac.cpp**

To run the code - 
**g++ -std=c++11 alphac.cpp \`pkg-config --cflags --libs opencv\`**


F-transform theory {#tutorial_fuzzy_theory}
=============

Goal
====
In this tutorial, the basic concept of fuzzy transform is presented. You will learn:

-   mathematic background,
-   how to apply concept of fuzziness to image processing.

The presented explanation demands knowledge of basic math. All related papers are cited and mostly accessible on https://www.researchgate.net/.

Introduction
====
In the last years, the theory of F-transforms has been intensively developed in many directions. In image processing, it has had successful applications in image compression and reduction, image fusion, edge detection and image reconstruction @cite Perf:FT @cite MSLP:cod-decod @cite Fusion:AFS12 @cite IPMU2012 @cite Perf:rec @cite vlavsanek2015patch. The F-transform is a technique that places a continuous/discrete function in correspondence with a finite vector of its F-transform components. In image processing, where images are identified by intensity functions of two arguments, the F-transform of the latter is given by a matrix of components.

Let me introduce F-transform of a 2D grayscale image \f$I\f$ that is considered as a function \f$I:[0,M]\times [0,N]\to [0,255]\f$ where \f$[0,M]=\{0,1,2,\ldots,M\}; [0,N]=\{0,1,2,\ldots,N\}\f$. It is assumed that the image is defined at points (pixels) that belong to the set \f$P\f$, where \f$P=\{(x,y)\mid x=0,1,\ldots, M;y=0,1,\ldots, N\}\f$.

Let \f$A_0, \dots ,A_m\f$ and \f$B_0, \dots ,B_n\f$ be basic functions, \f$A_0, \dots ,A_m : [0,M] \to [0, 1]\f$ be fuzzy partition of \f$[0,M]\f$ and \f$B_0, \dots ,B_n :[0,N]\to [0, 1]\f$ be fuzzy partition of \f$[0,N]\f$. Assume that the set of pixels \f$P\f$ is _sufficiently dense with respect to the chosen partitions_. This means that for all \f$k\in{0,\dots, m}(\exists x\in [0,M]) \ A_k(x)>0\f$, and for all \f$l\in{0,\dots, n}(\exists y\in [0,N])\ B_l(y)>0\f$.

\f$F^0\f$-transform
====
We say that the \f$m\times n\f$-matrix of real numbers  \f$F^0_{mn}[I] = (F^0_{kl})\f$ is called _the (discrete) F-transform_ of \f$I\f$ with respect to \f$\{A_0, \dots,A_m\}\f$ and \f$\{B_0, \dots,B_n\}\f$ if for all \f$k=0,\dots,m,\ l=0,\dots,n\f$:

\f[
    F^0_{kl}=\frac{\sum_{y=0}^{N}\sum_{x=0}^{M} I(x,y)A_k(x)B_l(y)}{\sum_{y=0}^{N}\sum_{x=0}^{M} A_k(x)B_l(y)}.
\f]

The coefficients \f$F^0_{kl}\f$ are called _components_ of the \f$F^0\f$-transform.

\f$F^1\f$-transform
====
\f$F^1\f$-transform has been presented in @cite perfilieva2014differentiation. We say that matrix \f$F^1_{mn}[I] = (F^1_{kl}), k=0,\ldots, m, l=0,\ldots, n\f$, is the \f$F^1\f$-transform of \f$I\f$ with respect to \f$\{A_k\times B_l\mid k=0,\ldots, m, l=0,\ldots, n\}\f$, and \f$F^1_{kl}\f$ is the corresponding \f$F^1\f$-transform component.

The \f$F^1\f$-transform components of \f$I\f$ are linear polynomials in the form

\f[
    F^1_{kl}(x,y)= c^{00}_{kl} + c^{10}_{kl}(x-x_k) + c^{01}_{kl}(y-y_l),
\f]

where the coefficients are given by

\f[
	c_{kl}^{00} =\frac{\sum_{y=0}^{N}\sum_{x=0}^{M} I(x,y)A_k(x)B_l(y)}{\sum_{y=0}^{N}\sum_{x=0}^{M} A_k(x)B_l(y)}, \\
	c_{kl}^{10} =\frac{\sum_{y=0}^{N}\sum_{x=0}^{M} I(x,y)(x - x_k)A_k(x)B_l(y)}{\sum_{y=0}^{N}\sum_{x=0}^{M} (x - x_k)^2A_k(x)B_l(y)}, \\
	c_{kl}^{01} =\frac{\sum_{y=0}^{N}\sum_{x=0}^{M} I(x,y)(y - y_l)A_k(x)B_l(y)}{\sum_{y=0}^{N}\sum_{x=0}^{M} (y - y_l)^2A_k(x)B_l(y)}.
\f]

Application to image processing
====
The technique of F-transforms uses two steps: _direct and inverse_. The direct step is described in the previous section whereas the inverse is as follows

\f[
    O(x,y)=\sum_{k=0}^{m}\sum_{l=0}^{n} F^d_{kl}A_k(x)B_l(y),
\f]

where \f$O\f$ is the output (reconstructed) image and \f$d\f$ is F-transform degree. In fact, the algorithm computes the F-transform components of the input image \f$I\f$ and spreads the components afterwards to the size of \f$I\f$. For details see @cite Perf:rec. Application to image processing is possible to take from two different views.

From pixel point of view
----
The pixels are processed one by one in a way that appropriate basic functions are found for each of them. It will be exactly four, two in each direction. We need some helper structure in the memory for collecting their values. The values will be used in the nominator of the related fuzzy component. Implementation of this approach uses keyword `FL` as __fast__ processing (because of more optimizations) and __linear basic function__.

![Pixel point of view with marked basic functions related to processed pixel.](images/fuzzy_pixel_view.jpg)

From fuzzy component point of view
----
In this way, image is divided to the regular areas. Each area is processed separately using kernel window. This approach benefits from easy to understand, matrix based processing with straight forward parallelization.

![Fuzzy component point of view with marked basic functions related to processed area.](images/fuzzy_BF_view.jpg)

This approach uses kernel \f$g\f$. Let us show linear case with radius \f$h = 2\f$ as an example.

\f[
    A   = (0, 0.5, 1, 0.5, 0) \\
    B^T = (0, 0.5, 1, 0.5, 0) \\
    g   = AB^T=\left(
             \begin{array}{ccccc}
               0 & 0    & 0   & 0    & 0 \\
               0 & 0.25 & 0.5 & 0.25 & 0 \\
               0 & 0.5  & 1   & 0.5  & 0 \\
               0 & 0.25 & 0.5 & 0.25 & 0 \\
               0 & 0    & 0   & 0    & 0 \\
             \end{array}
           \right)
\f]

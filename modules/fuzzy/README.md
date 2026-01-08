Fuzzy image processing
=======================

Author and maintainer: Pavel Vlasanek
                       pavel.vlasanek@osu.cz

This module is focused on the image processing using fuzzy mathematics, namely fuzzy (F)-transform. The F-transform technique approximates input function, where only few input values are known. The technique of F-transform takes local areas as areas with some additional structure. This structure is characterized by fuzzy predicates that may express any information which is relevant for a problem. In image processing, this can be, for example, a distance from a certain point, a relationship between points, color/intensity, texture, etc.

The F-transform is a technique putting a continuous/discrete function into a correspondence with a finite vector of its F-transform components. In image processing, where images are identified with intensity functions of two arguments, the F-transform of the latter is given by a matrix of components. The module currently covering F0-trasnform, where components are scalars.

The components can be used for inverse F-transform, where approximated input function is obtained. If input function (image) includes some damaged or missing areas, these areas are recomputed and restored after invesre F-transform processing.

Let me give you two related papers:

Perfilieva, Irina, and Pavel Vlašánek. "Image Reconstruction by means of F-transform." Knowledge-Based Systems 70 (2014): 55-63.

Perfilieva, Irina. "Fuzzy transforms: Theory and applications." Fuzzy sets and systems 157.8 (2006): 993-1023.

Investigation of the F-transform technique leads to several applications in image processing. Currently investigated are image inpainting, filtering, resampling, edge detection, compression and image fusion.

The module covers:

* F0 processing (fuzzy_F0_math.cpp): Functions for computation of the image F0 components and inverse F0-transform.
* Fuzzy image processing (fuzzy_image.cpp): Functions aimed to image processing currently including image inpainting and image filtering.

There are also tests in test_image.cpp using resources from opencv_extra, and samples in fuzzy_inpainting.cpp and fuzzy_filtering.cpp.

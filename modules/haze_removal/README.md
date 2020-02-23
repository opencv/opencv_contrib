Haze Removal algorithms
========================

This module contains functions which can dehaze hazy/foggy images.

Currently the following algorithms are implemented:

* Single image haze removal using dark channel priors


Note that the " Single image haze removal using dark channel priors " algorithm was patented and its use may be restricted by following (but not limited to) list of patents:

* _US8340461B2_ Single image haze removal using dark channel priors


Since OpenCV's license imposes different restrictions on usage please consult a legal advisor before using this algorithm any way.

That's why you need to set the OPENCV_ENABLE_NONFREE option in CMake to use this algorithm.

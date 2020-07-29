// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html
// Copyright (C) 2020 by Archit Rungta

#ifndef OPENCV_JULIA_HPP
#define OPENCV_JULIA_HPP

#include "opencv2/core.hpp"

/**
@defgroup julia Julia bindings for OpenCV

Julia (https://julialang.org) is a programming language for scientific community with growing popularity.
These are bindings for a subset of OpenCV functionality, based on libcxxwrap-julia and CxxWrap packages.

For installation instructions, see README.md in this module or OpenCV wiki (https://github.com/opencv/opencv/wiki)
*/

namespace cv
{
namespace julia
{

    //! @addtogroup julia
    //! @{

    // initializes Julia bindings module
    CV_WRAP void initJulia(int argc, char **argv);

    //! @} julia

} // namespace julia
} // namespace cv

#endif

/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2009, Willow Garage, Inc.
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of Willow Garage, Inc. nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 */

#ifndef __OPENCV_SFM_HPP__
#define __OPENCV_SFM_HPP__

#include <opencv2/sfm/conditioning.hpp>
#include <opencv2/sfm/fundamental.hpp>
#include <opencv2/sfm/io.hpp>
#include <opencv2/sfm/numeric.hpp>
#include <opencv2/sfm/projection.hpp>
#include <opencv2/sfm/triangulation.hpp>
#if CERES_FOUND
#include <opencv2/sfm/reconstruct.hpp>
#include <opencv2/sfm/simple_pipeline.hpp>
#endif

/** @defgroup sfm Structure From Motion

The opencv_sfm module contains algorithms to perform 3d reconstruction
from 2d images.\n
The core of the module is based on a light version of
[Libmv](https://developer.blender.org/project/profile/59) originally
developed by Sameer Agarwal and Keir Mierle.

__Whats is libmv?__ \n
libmv, also known as the Library for Multiview Reconstruction (or LMV),
is the computer vision backend for Blender's motion tracking abilities.
Unlike other vision libraries with general ambitions, libmv is focused
on algorithms for match moving, specifically targeting [Blender](https://developer.blender.org) as the
primary customer. Dense reconstruction, reconstruction from unorganized
photo collections, image recognition, and other tasks are not a focus
of libmv.

__Development__ \n
libmv is officially under the Blender umbrella, and so is developed
on developer.blender.org. The [source repository](https://developer.blender.org/diffusion/LMV) can get checked out
independently from Blender.

This module has been originally developed as a project for Google Summer of Code 2012-2015.

@note
  - Notice that it is compiled only when Eigen, GLog and GFlags are correctly installed.\n
    Check installation instructions in the following tutorial: @ref tutorial_sfm_installation

  @{
    @defgroup conditioning Conditioning
    @defgroup fundamental Fundamental
    @defgroup io Input/Output
    @defgroup numeric Numeric
    @defgroup projection Projection
    @defgroup robust Robust Estimation
    @defgroup triangulation Triangulation

    @defgroup reconstruction Reconstruction
      @note
        - Notice that it is compiled only when Ceres Solver is correctly installed.\n
          Check installation instructions in the following tutorial: @ref tutorial_sfm_installation


    @defgroup simple_pipeline Simple Pipeline
      @note
          - Notice that it is compiled only when Ceres Solver is correctly installed.\n
            Check installation instructions in the following tutorial: @ref tutorial_sfm_installation

  @}

*/

#endif

/* End of file. */

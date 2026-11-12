/*

By downloading, copying, installing or using the software you agree to this
license. If you do not agree to this license, do not download, install,
copy or use the software.


                          License Agreement
               For Open Source Computer Vision Library
                       (3-clause BSD License)

Copyright (C) 2014, OpenCV Foundation, all rights reserved.
Third party copyrights are property of their respective owners.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

  * Redistributions of source code must retain the above copyright notice,
    this list of conditions and the following disclaimer.

  * Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions and the following disclaimer in the documentation
    and/or other materials provided with the distribution.

  * Neither the names of the copyright holders nor the names of the contributors
    may be used to endorse or promote products derived from this software
    without specific prior written permission.

This software is provided by the copyright holders and contributors "as is" and
any express or implied warranties, including, but not limited to, the implied
warranties of merchantability and fitness for a particular purpose are
disclaimed. In no event shall copyright holders or contributors be liable for
any direct, indirect, incidental, special, exemplary, or consequential damages
(including, but not limited to, procurement of substitute goods or services;
loss of use, data, or profits; or business interruption) however caused
and on any theory of liability, whether in contract, strict liability,
or tort (including negligence or otherwise) arising in any way out of
the use of this software, even if advised of the possibility of such damage.

*/

#ifndef __OPENCV_SURFACE_MATCHING_PRECOMP_HPP__
#define __OPENCV_SURFACE_MATCHING_PRECOMP_HPP__

#include "opencv2/surface_matching/ppf_match_3d.hpp"
#include "opencv2/surface_matching/icp.hpp"
#include "opencv2/surface_matching/ppf_helpers.hpp"

#include <string>
#include <cstdio>
#include <cstdlib>
#include <math.h>
#include <ctime>

#include <fstream>
#include <iostream>
#include <algorithm>

#if defined (_OPENMP)
#include<omp.h>
#endif

#include <sstream>  // flann dependency, needed in precomp now
#include "opencv2/flann.hpp"

#include "c_utils.hpp"

namespace cv
{
namespace ppf_match_3d
{

// Accepts NxC (C >= 6) CV_32FC1, Nx1/1xN CV_32FC6 (a (N,1,6) NumPy array) and {N,1,6}/{1,N,6} CV_32FC1,
// returns a CV_32FC1 header with one point per row (copy only for a non-continuous multi-channel or 3D input).
static inline Mat normalizePointCloud(const Mat& pc)
{
  if (pc.dims == 2 && pc.channels() == 1)
  {
    CV_CheckTypeEQ(pc.type(), CV_32FC1, "Point cloud must be CV_32F");
    CV_Check(pc.cols, pc.rows > 0 && pc.cols >= 6, "Point cloud must be Nx6 (x, y, z, nx, ny, nz)");
    return pc;
  }
  const int n = pc.checkVector(6, CV_32F, false);
  CV_Check(n, n > 0, "Point cloud must be Nx6 CV_32FC1 or Nx1/1xN CV_32FC6 (x, y, z, nx, ny, nz)");
  return (pc.isContinuous() ? pc : pc.clone()).reshape(1, n);
}

} // namespace ppf_match_3d
} // namespace cv

#endif /* __OPENCV_SURFACE_MATCHING_PRECOMP_HPP__ */

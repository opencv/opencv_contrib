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

static inline Mat normalizePointCloud(const Mat& pc, bool allow3Cols = false, int errCode = Error::StsBadSize)
{
  CV_CheckFalse(pc.empty(), "Point cloud must not be empty");
  CV_CheckTypeEQ(pc.depth(), CV_32F, "Point cloud must have CV_32F depth");

  if (pc.dims == 2 && pc.channels() == 1)
  {
    if (pc.cols != 6 && !(allow3Cols && pc.cols == 3))
      CV_Error(errCode, allow3Cols
          ? "Point cloud must have 3 or 6 elements per point"
          : "Point cloud must have 6 elements per point (x, y, z, nx, ny, nz)");
    return pc;
  }

  if (pc.dims == 2 && (pc.cols == 1 || pc.rows == 1))
  {
    const int ch = pc.channels();
    if (ch != 6 && !(allow3Cols && ch == 3))
      CV_Error(errCode, allow3Cols
          ? "Point cloud must have 3 or 6 elements per point"
          : "Point cloud must have 6 elements per point (x, y, z, nx, ny, nz)");
    const int numPoints = pc.rows * pc.cols;
    Mat cont = pc.isContinuous() ? pc : pc.clone();
    return cont.reshape(1, numPoints);
  }

  if (pc.dims == 3 && pc.channels() == 1 && (pc.size[0] == 1 || pc.size[1] == 1))
  {
    const int elemCols = pc.size[2];
    if (elemCols != 6 && !(allow3Cols && elemCols == 3))
      CV_Error(errCode, allow3Cols
          ? "Point cloud must have 3 or 6 elements per point"
          : "Point cloud must have 6 elements per point (x, y, z, nx, ny, nz)");
    const int numPoints = pc.size[0] * pc.size[1];
    Mat cont = pc.isContinuous() ? pc : pc.clone();
    return cont.reshape(1, numPoints);
  }

  CV_Error(errCode, allow3Cols
      ? "Point cloud must be Nx3/Nx6 (CV_32FC1) or Nx1/1xN with 3 or 6 channels"
      : "Point cloud must be Nx6 (CV_32FC1) or Nx1/1xN with 6 channels");
}

} // namespace ppf_match_3d
} // namespace cv

#endif /* __OPENCV_SURFACE_MATCHING_PRECOMP_HPP__ */

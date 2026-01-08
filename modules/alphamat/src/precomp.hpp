// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef __OPENCV_PRECOMP_H__
#define __OPENCV_PRECOMP_H__

#include <vector>
#include <unordered_set>
#include <set>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/utils/logger.hpp>

#include <opencv2/alphamat.hpp>

#include "3rdparty/nanoflann.hpp"
#include "3rdparty/KDTreeVectorOfVectorsAdaptor.h"

#ifdef HAVE_EIGEN
#include <Eigen/Eigen>
#include <opencv2/core/eigen.hpp>
#include <Eigen/IterativeLinearSolvers>
#endif

#include "intraU.hpp"
#include "cm.hpp"
#include "local_info.hpp"

#endif

// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef __OPENCV_PRECOMP_H__
#define __OPENCV_PRECOMP_H__

#include <iostream>
#include <vector>
#include <unordered_set>
#include <set>

#include <opencv2/core/base.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>
// #include <opencv2/highgui.hpp>

#include "nanoflann.hpp"
#include "KDTreeVectorOfVectorsAdaptor.h"
#include <Eigen/Sparse>

namespace cv{
	namespace alphamat{
		const int dim = 5;  // dimension of feature vectors
	}
}

#include "KtoU.hpp"
#include "intraU.hpp"
#include "cm.hpp"
#include "local_info.hpp"
#include <Eigen/IterativeLinearSolvers>
#include "trimming.hpp"

#endif

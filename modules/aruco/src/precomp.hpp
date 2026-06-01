// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

// BUG FIX 16: Include guard was named __OPENCV_CCALIB_PRECOMP__ (from the ccalib module).
// This is the aruco module — wrong guard name could cause silent double-inclusion
// conflicts if ccalib and aruco are both compiled in the same translation unit.
// FIX: Renamed to __OPENCV_ARUCO_PRECOMP__.
#ifndef __OPENCV_ARUCO_PRECOMP__
#define __OPENCV_ARUCO_PRECOMP__

#include <opencv2/core.hpp>
#include <vector>

#endif

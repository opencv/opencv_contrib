// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_CANNOPS_PYOPENCV_CANN_HPP
#define OPENCV_CANNOPS_PYOPENCV_CANN_HPP

#ifdef HAVE_OPENCV_CORE

#include "opencv2/cann.hpp"

typedef std::vector<cann::AscendMat> vector_AscendMat;
typedef cann::AscendMat::Allocator AscendMat_Allocator;

CV_PY_TO_CLASS(cann::AscendMat);
CV_PY_TO_CLASS(cann::AscendStream);

CV_PY_TO_CLASS_PTR(cann::AscendMat);
CV_PY_TO_CLASS_PTR(cann::AscendMat::Allocator);

CV_PY_FROM_CLASS(cann::AscendMat);
CV_PY_FROM_CLASS(cann::AscendStream);

CV_PY_FROM_CLASS_PTR(cann::AscendMat::Allocator);

#endif // HAVE_OPENCV_CORE

#endif // OPENCV_CANNOPS_PYOPENCV_CANN_HPP

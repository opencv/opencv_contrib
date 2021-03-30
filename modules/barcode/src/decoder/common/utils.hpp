// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (c) 2020-2021 darkliang wangberlinT Certseeds

#ifndef __OPENCV_BARCODE_UTILS_HPP__
#define __OPENCV_BARCODE_UTILS_HPP__


namespace cv {
namespace barcode {

constexpr int OTSU = 0;
constexpr int HYBRID = 1;

void preprocess(Mat &src, Mat &dst);

Mat binarize(const Mat &src, int mode);
}
}
#endif //__OPENCV_BARCODE_UTILS_HPP__

// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_IMG_HASH_PRECOMP_H
#define OPENCV_IMG_HASH_PRECOMP_H

#include "opencv2/core.hpp"
#include "opencv2/core/base.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgproc/types_c.h"
#include "opencv2/img_hash.hpp"

#include <bitset>
#include <iostream>

namespace cv{ namespace img_hash {

class ImgHashBase::ImgHashImpl
{
public:
    virtual void compute(cv::InputArray inputArr, cv::OutputArray outputArr) = 0;
    virtual double compare(cv::InputArray hashOne, cv::InputArray hashTwo) const = 0;
    virtual ~ImgHashImpl() {}
};

}} // cv::img_hash::

#endif // OPENCV_IMG_HASH_PRECOMP_H

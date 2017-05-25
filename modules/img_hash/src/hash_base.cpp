// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"

namespace cv {
namespace img_hash{

ImgHashBase::ImgHashBase()
{
    pImpl = 0;
}

ImgHashBase::~ImgHashBase()
{
    if (pImpl)
        delete getImpl(pImpl);
}

void ImgHashBase::compute(cv::InputArray inputArr, cv::OutputArray outputArr)
{
    getImpl(pImpl)->compute(inputArr, outputArr);
}

double ImgHashBase::compare(cv::InputArray hashOne, cv::InputArray hashTwo) const
{
    return getImpl(pImpl)->compare(hashOne, hashTwo);
}

} } // cv::img_hash::

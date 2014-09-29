#include "precomp.hpp"

namespace cv { namespace photoeffects {

int warmify(cv::InputArray src, cv::OutputArray dst, uchar delta)
{
    CV_Assert(src.type() == CV_8UC3);
    Mat imgSrc = src.getMat();
    CV_Assert(imgSrc.data);
    dst.create(src.size(), CV_8UC3);
    Mat imgDst = dst.getMat();

    imgDst = imgSrc + Scalar(0, delta, delta);
    return 0;
}

}}
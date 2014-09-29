#include "precomp.hpp"

namespace cv { namespace photoeffects {

int boostColor(cv::InputArray src, cv::OutputArray dst, float intensity)
{
    const int MAX_INTENSITY = 255;

    Mat srcImg = src.getMat();

    CV_Assert(srcImg.channels() == 3);
    CV_Assert(intensity >= 0.0f && intensity <= 1.0f);

    if (srcImg.type() != CV_8UC3)
    {
        srcImg.convertTo(srcImg, CV_8UC3);
    }

    Mat srcHls;
    cvtColor(srcImg, srcHls, CV_BGR2HLS);

    int intensityInt = intensity * MAX_INTENSITY;
    srcHls += Scalar(0, 0, intensityInt);

    cvtColor(srcHls, dst, CV_HLS2BGR);

    dst.getMat().convertTo(dst, srcImg.type());
    return 0;
}

}}
// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"

// Random Generator
Mat randomMat(int w, int h, int dtype, float min, float max)
{
    Mat rnMat(w, h, dtype);
    RNG rng(getTickCount());
    rng.fill(rnMat, RNG::UNIFORM, min, max);
    return rnMat;
}
Scalar randomScalar()
{
    RNG rng(getTickCount());
    Scalar sc;
    rng.fill(sc, RNG::UNIFORM, 1.0, 5.0);
    return sc;
}
float randomNum()
{
    RNG rng(getTickCount());
    float rdnNum = float(rng.uniform(1.0, 5.0));
    return rdnNum;
}

int randomInterger()
{
    RNG rng(getTickCount());
    float rdnNum = float(rng.uniform(1, 5));
    return rdnNum;
}

Mat genMask()
{
    Mat mask = Mat::zeros(Size(10, 10), CV_8UC1);
    rectangle(mask, cv::Rect(5, 5, 3, 3), Scalar(255), -1);
    return mask;
}

AscendMat genNpuMask()
{
    cv::Mat mask = genMask();
    cv::cann::AscendMat npuMask;
    npuMask.upload(mask);
    return npuMask;
}

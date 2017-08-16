// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef _OPENCV_VIGNETTEREMOVER_HPP
#define _OPENCV_VIGNETTEREMOVER_HPP

#include "opencv2/core.hpp"

namespace cv { namespace photometric_calib {

class CV_EXPORTS VignetteRemover
{
public:
    VignetteRemover(const std::string &vignettePath, int w_, int h_);
    ~VignetteRemover();

    Mat getUnVignetteImageMat(std::vector<float> &unGammaImVec);
    void getUnVignetteImageVec(const std::vector<float> &unGammaImVec, std::vector<float> &outImVec);

private:
    float* vignetteMap;
    float* vignetteMapInv;
    int w,h;
    bool validVignette;
};

}}


#endif //_OPENCV_VIGNETTEREMOVER_HPP

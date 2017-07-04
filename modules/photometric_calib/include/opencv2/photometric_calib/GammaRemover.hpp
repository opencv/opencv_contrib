// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef _OPENCV_GAMMAREMOVER_HPP
#define _OPENCV_GAMMAREMOVER_HPP

#include "opencv2/photometric_calib.hpp"

namespace cv { namespace photometric_calib {

class CV_EXPORTS GammaRemover
{
public:
    GammaRemover(const std::string &gammaPath, int w_, int h_);

    Mat getUnGammaImageMat(Mat inputIm);
    void getUnGammaImageVec(Mat inputIm, std::vector<float> &outImVec);

    inline float* getG()
    {
        if(!validGamma) return 0;
        else return G;
    };
    inline float* getGInv()
    {
        if(!validGamma) return 0;
        else return GInv;
    };

private:
    float G[256];
    float GInv[256];
    int w,h;
    bool validGamma;
};


}}

#endif //_OPENCV__GAMMAREMOVER_HPP

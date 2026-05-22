// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"
#include "opencv2/photometric_calib/VignetteRemover.hpp"
#include "opencv2/photometric_calib/GammaRemover.hpp"

namespace cv {
namespace photometric_calib {

VignetteRemover::VignetteRemover(const std::string &vignettePath, const std::string &pcalibPath, int w_, int h_)
{
    CV_Assert(vignettePath != "");
    CV_Assert(pcalibPath != "");

    validVignette = false;
    vignetteMap = 0;
    vignetteMapInv = 0;
    w = w_;
    h = h_;


    Mat vignetteMat = imread(vignettePath, IMREAD_UNCHANGED);
    vignetteMap = new float[w * h];
    vignetteMapInv = new float[w * h];

    CV_Assert(vignetteMat.rows == h && vignetteMat.cols == w);

    CV_Assert(vignetteMat.type() == CV_8U || vignetteMat.type() == CV_16U);
    if (vignetteMat.type() == CV_8U)
    {
        float maxV = 0;
        for (int i = 0; i < w * h; i++)
        {
            if (vignetteMat.at<unsigned char>(i) > maxV) maxV = vignetteMat.at<unsigned char>(i);
        }

        for (int i = 0; i < w * h; i++)
        {
            vignetteMap[i] = vignetteMat.at<unsigned char>(i) / maxV;
        }
    }

    else
    {
        float maxV = 0;
        for (int i = 0; i < w * h; i++)
        {
            if (vignetteMat.at<ushort>(i) > maxV) maxV = vignetteMat.at<ushort>(i);
        }

        for (int i = 0; i < w * h; i++)
        {
            vignetteMap[i] = vignetteMat.at<ushort>(i) / maxV;
        }
    }

    for (int i = 0; i < w * h; i++)
    {
        vignetteMapInv[i] = 1.0f / vignetteMap[i];
    }

    _pcalibPath = pcalibPath;
    validVignette = true;

}

Mat VignetteRemover::getUnVignetteImageMat(Mat oriImMat)
{
    std::vector<float> _outImVec(w * h);
    getUnVignetteImageVec(oriImMat, _outImVec);

    Mat _outIm(h, w, CV_32F, &_outImVec[0]);
    Mat outIm = _outIm * (1 / 255.0f);
    return outIm;
}

void VignetteRemover::getUnVignetteImageVec(Mat oriImMat, std::vector<float> &outImVec)
{
    CV_Assert(validVignette);
    CV_Assert(outImVec.size() == (unsigned long) w * h);
    photometric_calib::GammaRemover gammaRemover(_pcalibPath, w, h);
    std::vector<float> unGammaImVec(w * h);
    gammaRemover.getUnGammaImageVec(oriImMat, unGammaImVec);
    for (int i = 0; i < w * h; ++i)
    {
        outImVec[i] = unGammaImVec[i] * vignetteMapInv[i];
    }
}

VignetteRemover::~VignetteRemover()
{
    if (vignetteMap != 0)
    {
        delete[] vignetteMap;
    }
    if (vignetteMapInv != 0)
    {
        delete[] vignetteMapInv;
    }
}

} // namespace photometric_calib
} // namespace cv

// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef _OPENCV_VIGNETTECALIB_HPP
#define _OPENCV_VIGNETTECALIB_HPP

#include "opencv2/core.hpp"
#include "opencv2/photometric_calib/Reader.hpp"
#include "opencv2/photometric_calib/GammaRemover.hpp"

namespace cv { namespace photometric_calib {


class CV_EXPORTS VignetteCalib
{
public:
    VignetteCalib(std::string folderPath, std::string timePath, std::string cameraFile, std::string gammaFile, std::string imageFormat);
    VignetteCalib(std::string folderPath, std::string timePath, std::string cameraFile, std::string gammaFile, int imageSkip, int maxIterations, int outlierTh,
                  int gridWidth, int gridHeight, float facW, float facH, int maxAbsGrad, std::string imageFormat);

    virtual ~VignetteCalib();

    //EIGEN_ALWAYS_INLINE float getInterpolatedElement(const float* const mat, const float x, const float y, const int width)
    float getInterpolatedElement(const float* const mat, const float x, const float y, const int width);
    void displayImage(float* I, int w, int h, std::string name);
    void displayImageV(float* I, int w, int h, std::string name);
    void calib();

private:
    int _imageSkip;
    int _maxIterations;
    int _outlierTh;

    // grid width for template image.
    int _gridWidth;
    int _gridHeight;

    // width of grid relative to marker (fac times marker size)
    float _facW;
    float _facH;

    // remove pixel with absolute gradient larger than this from the optimization.
    int _maxAbsGrad;

    Mat _cameraMatrix;
    Mat _distCoeffs;

    Reader *imageReader;
    GammaRemover *gammaRemover;
};

}}


#endif //_OPENCV_VIGNETTECALIB_HPP

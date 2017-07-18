// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef _OPENCV_VIGNETTECALIB_HPP
#define _OPENCV_VIGNETTECALIB_HPP

#include "opencv2/photometric_calib.hpp"
#include "opencv2/photometric_calib/Reader.hpp"

namespace cv { namespace photometric_calib {


class CV_EXPORTS VignetteCalib
{
public:
    VignetteCalib(std::string folderPath, std::string timePath);
    VignetteCalib(std::string folderPath, std::string timePath, int imageSkip, int maxIterations, int outlierTh,
                  int gridWidth, int gridHeight, float facW, float facH, int maxAbsGrad);

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

    Reader *imageReader;
};

}}


#endif //_OPENCV_VIGNETTECALIB_HPP

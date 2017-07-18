// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"
#include "opencv2/photometric_calib/VignetteCalib.hpp"

#include <fstream>
#include <iostream>

namespace cv { namespace photometric_calib{


VignetteCalib::VignetteCalib(std::string folderPath, std::string timePath) :
        _imageSkip(1), _maxIterations(20), _outlierTh(15), _gridWidth(1000), _gridHeight(1000), _facW(5), _facH(5),
        _maxAbsGrad(255)
{
    imageReader = new Reader(folderPath, "png", timePath);
}

VignetteCalib::VignetteCalib(std::string folderPath, std::string timePath, int imageSkip, int maxIterations,
                             int outlierTh, int gridWidth, int gridHeight, float facW, float facH, int maxAbsGrad) :
        _imageSkip(imageSkip), _maxIterations(maxIterations), _outlierTh(outlierTh), _gridWidth(gridWidth), _gridHeight(gridHeight),
        _facW(facW), _facH(facH), _maxAbsGrad(maxAbsGrad)
{
    imageReader = new Reader(folderPath, "png", timePath);
}
}}// namespace photometric_calib, cv